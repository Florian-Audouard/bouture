"""``BackgroundAgents`` : bouture sur un thread et une boucle à lui (D123-D125).

Tests synchrones : chaque test choisit la boucle de l'hôte (aucune, ou ``Selector`` forcée).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from bouture import (
    AgentDefinition,
    Agents,
    FakeModelClient,
    FakeReply,
    MemoryStorage,
    Storage,
    ToolContext,
)
from bouture.hosting import BackgroundAgents

from .conftest import make_definitions

OWNER = "alice"
StorageFactory = Callable[[], Storage]


@pytest.fixture(params=["memory", "sqlite"])
def storage_factory(request: pytest.FixtureRequest, tmp_path: Path) -> StorageFactory:
    """Le stockage est ouvert par la boucle de bouture, pas par celle du test. Deux appels
    donnent le même contenu (même objet en mémoire, même fichier SQLite)."""
    if request.param == "memory":
        shared = MemoryStorage()
        return lambda: shared
    from bouture.storage.sqlite import SQLiteStorage

    return lambda: SQLiteStorage(tmp_path / "bouture.db")


class Studio(BaseModel):
    last: str = ""


def studio() -> AgentDefinition[Studio, Any]:
    d = AgentDefinition[Studio, Any]("studio", model="fake", state_type=Studio)

    @d.tool(timeout=None)
    async def render(ctx: ToolContext[Studio, Any], prompt: str) -> str:
        """Lance un vrai sous-processus (faux moteur de rendu)."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; print('rendu:' + sys.argv[1])",
            prompt,
            stdout=subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        ctx.state.last = out.decode().strip()
        return ctx.state.last

    return d


def make(fake: FakeModelClient, storage: Storage) -> Agents[Any]:
    a = Agents[Any](storage=storage, model_client=fake, retry_base_delay=0, poll_interval=0.05)
    a.register(*make_definitions(), studio())
    return a


# -- pont synchrone -------------------------------------------------------------------
def test_sync_bridge(storage_factory: StorageFactory) -> None:
    fake = FakeModelClient(
        [FakeReply("", tool_calls=[("set_title", {"title": "T"})]), FakeReply("ok")]
    )
    bg = BackgroundAgents(make(fake, storage_factory())).start()
    assert bg.running and bg.loop is not None
    a = bg.agents
    s = bg.run_sync(a.create_session("note", OWNER))
    h = bg.run_sync(a.send_message(s.id, OWNER, "titre"))
    events = list(bg.stream_sync(h.events()))
    assert events[0].type == "turn_start" and events[-1].data["reason"] == "completed"
    assert bg.run_sync(a.get_state(s.id, OWNER)).state.title == "T"

    bg.shutdown()
    bg.shutdown()  # idempotent
    assert not bg.running
    with pytest.raises(RuntimeError, match="arrêté"):
        bg.run_sync(a.get_session(s.id, OWNER))


def test_run_sync_timeout_cancels(storage_factory: StorageFactory) -> None:
    with BackgroundAgents(make(FakeModelClient(), storage_factory())) as bg:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            bg.run_sync(asyncio.sleep(10), timeout=0.05)
        assert time.monotonic() - started < 2


def test_start_failure_is_raised() -> None:
    class Broken(MemoryStorage):
        async def setup(self, *, auto_migrate: bool = True) -> None:
            raise RuntimeError("base illisible")

    bg = BackgroundAgents(make(FakeModelClient(), Broken()))
    with pytest.raises(RuntimeError, match="base illisible"):
        bg.start()
    assert not bg.running


def test_bridges_from_inside_bouture_loop(storage_factory: StorageFactory) -> None:
    with BackgroundAgents(make(FakeModelClient(), storage_factory())) as bg:

        async def inside() -> list[str]:
            out = [await bg.run(asyncio.sleep(0, "await direct"))]
            try:
                bg.run_sync(asyncio.sleep(0))
            except RuntimeError:
                out.append("run_sync refusé")
            return out

        assert bg.run_sync(inside()) == ["await direct", "run_sync refusé"]


# -- boucle Selector forcée chez l'hôte -------------------------------------------------
def test_selector_host_runs_subprocess_tool(storage_factory: StorageFactory) -> None:
    fake = FakeModelClient(
        [FakeReply("", tool_calls=[("render", {"prompt": "chat"})]), FakeReply("Voilà.")]
    )
    agents = make(fake, storage_factory())

    async def host() -> None:
        loop = asyncio.get_running_loop()
        assert isinstance(loop, asyncio.SelectorEventLoop)
        if sys.platform == "win32":  # la boucle de l'hôte ne sait pas lancer de sous-processus
            with pytest.raises(NotImplementedError):
                await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")
        async with BackgroundAgents(agents) as bg:
            if sys.platform == "win32":
                assert isinstance(bg.loop, asyncio.ProactorEventLoop)
            a = bg.aio
            s = await a.create_session("studio", OWNER)
            h = await a.send_message(s.id, OWNER, "dessine un chat")
            kinds = [ev.type async for ev in h.events()]
            assert kinds[-1] == "turn_end"
            end = await (await a.send_message(s.id, OWNER, "encore")).wait()
            assert end.data["reason"] == "completed"
            results = [e for e in await a.get_branch(s.id, OWNER) if e.type == "tool_result"]
            assert results[0].content["content"] == "rendu:chat"
            assert (await a.get_state(s.id, OWNER)).state.last == "rendu:chat"

    fake.push(FakeReply("Encore."))
    asyncio.run(host(), loop_factory=asyncio.SelectorEventLoop)


def test_closing_a_stream_stops_the_reader(storage_factory: StorageFactory) -> None:
    fake = FakeModelClient([FakeReply("un"), FakeReply("deux")])

    async def tasks() -> int:
        return len(asyncio.all_tasks())

    async def host(bg: BackgroundAgents[Any]) -> None:
        a = bg.aio
        s = await a.create_session("note", OWNER)
        baseline = bg.run_sync(tasks())  # cette tâche-ci comprise
        await (await a.send_message(s.id, OWNER, "1")).wait()
        sub = a.subscribe(s.id, OWNER, after=0)
        first = await anext(sub)
        assert first.type == "turn_start"
        await sub.aclose()  # le client se déconnecte
        for _ in range(50):
            if bg.run_sync(tasks()) <= baseline:
                break
            await asyncio.sleep(0.02)
        assert bg.run_sync(tasks()) <= baseline

    with BackgroundAgents(make(fake, storage_factory())) as bg:
        asyncio.run(host(bg), loop_factory=asyncio.SelectorEventLoop)


# -- arrêt ----------------------------------------------------------------------------
def test_shutdown_closes_running_turns(storage_factory: StorageFactory) -> None:
    fake = FakeModelClient([FakeReply("", tool_calls=[("slow", {})])])
    bg = BackgroundAgents(make(fake, storage_factory())).start()
    a = bg.agents
    s = bg.run_sync(a.create_session("note", OWNER))
    h = bg.run_sync(a.send_message(s.id, OWNER, "x"))
    events = bg.stream_sync(h.events())
    for ev in events:
        if ev.type == "tool_call":
            break
    events.close()
    bg.shutdown(timeout=2)
    assert bg._thread is not None and not bg._thread.is_alive()  # pyright: ignore[reportPrivateUsage]

    with BackgroundAgents(make(FakeModelClient(), storage_factory())) as bg2:
        branch = bg2.run_sync(bg2.agents.get_branch(s.id, OWNER))
        assert branch[-1].type == "tool_result" and branch[-1].content["is_error"]
        assert bg2.run_sync(bg2.agents.get_session(s.id, OWNER)).status == "idle"


# -- routes FastAPI -------------------------------------------------------------------
def test_router_with_background_agents(storage_factory: StorageFactory) -> None:
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, Header
    from fastapi.testclient import TestClient

    from bouture.fastapi import make_router

    def current_user(x_user: str = Header()) -> str:
        return x_user

    fake = FakeModelClient([FakeReply("Bonjour.")])
    with BackgroundAgents(make(fake, storage_factory())) as bg:
        app = FastAPI()
        app.include_router(make_router(bg, current_user), prefix="/agents")
        with TestClient(app) as client:
            headers = {"x-user": OWNER}
            r = client.post("/agents/sessions", json={"kind": "note"}, headers=headers)
            assert r.status_code == 201, r.text
            sid = r.json()["id"]
            r = client.post(
                f"/agents/sessions/{sid}/messages", json={"content": "x"}, headers=headers
            )
            assert r.status_code == 200 and '"reason": "completed"' in r.text
            assert client.get(f"/agents/sessions/{sid}", headers=headers).json()["status"] == "idle"
            r = client.get("/agents/sessions/00000000-0000-0000-0000-000000000000", headers=headers)
            assert r.status_code == 404

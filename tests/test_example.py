"""L'exemple personnage/monde, piloté par ses routes avec le modèle de démo."""

from __future__ import annotations

import httpx
import pytest

from examples.character_world.app import build_agents, create_app

from .conftest import PG_URL
from .test_routes import parse_sse

H = {"x-user": "alice"}


@pytest.mark.parametrize("backend", ["memory", "postgres"])
async def test_character_reads_published_world(backend: str) -> None:
    if backend == "postgres" and not PG_URL:
        pytest.skip("BOUTURE_TEST_PG_URL non défini")
    agents = build_agents(PG_URL if backend == "postgres" else None)
    app = create_app(agents)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
    ):

        async def say(sid: str, content: str) -> list[dict]:
            r = await c.post(
                f"/agents/sessions/{sid}/messages", json={"content": content}, headers=H
            )
            assert r.status_code == 200, r.text
            return parse_sse(r.text)

        # Un monde, édité puis publié.
        world = (await c.post("/agents/sessions", json={"kind": "world"}, headers=H)).json()["id"]
        await say(world, "nom: Aldéra")
        await say(world, "lieu: Port-Brume")
        r = await c.post(f"/agents/sessions/{world}/publish", headers=H)
        assert r.status_code == 200, r.text
        pub = r.json()

        # Un personnage lié à ce monde (version épinglée).
        r = await c.post(
            "/agents/sessions",
            json={
                "kind": "character",
                "inputs": {"world": {"kind": "world", "id": pub["object_id"]}},
            },
            headers=H,
        )
        assert r.status_code == 201, r.text
        char = r.json()["id"]

        # Le monde évolue ensuite : le personnage garde la version 1.
        await say(world, "lieu: Hautes-Terres")
        await c.post(f"/agents/sessions/{world}/publish", headers=H)

        evs = await say(char, "monde")
        result = next(e for e in evs if e["type"] == "tool_result")
        assert "Aldéra" in result["data"]["content"] and "Port-Brume" in result["data"]["content"]
        assert "Hautes-Terres" not in result["data"]["content"]

        # Pause (AskUser) puis reprise.
        evs = await say(char, "portrait: aquarelle")
        assert evs[-1]["data"]["reason"] == "awaiting_input"
        r = await c.post(f"/agents/sessions/{char}/answer", json={"answer": "oui"}, headers=H)
        assert parse_sse(r.text)[-1]["data"]["reason"] == "completed"
        state = (await c.get(f"/agents/sessions/{char}/state", headers=H)).json()["state"]
        assert state["portrait_style"] == "aquarelle"

        # Bob ne voit pas les publications d'Alice.
        r = await c.get(f"/agents/publications/world/{pub['object_id']}", headers={"x-user": "bob"})
        assert r.json() == []

    if backend == "postgres":
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(PG_URL)  # type: ignore[arg-type]
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT version, name FROM app_worlds WHERE id = :id ORDER BY version"),
                {"id": pub["object_id"]},
            )
            assert [tuple(r) for r in rows] == [(1, "Aldéra"), (2, "Aldéra")]
        await engine.dispose()

"""Outils longs et bail (D121-D122), sous horloge simulée : aucune attente réelle du bail."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

import bouture.agents
import bouture.turn
from bouture import (
    AgentDefinition,
    Agents,
    FakeModelClient,
    FakeReply,
    SessionBusy,
    ToolContext,
)

from .conftest import AgentsFactory, Note

OWNER = "alice"
TTL = 30.0  # bail par défaut ; le temps est simulé


class Clock:
    """Horloge murale et ``sleep`` du battement, avancées à la main par le test."""

    def __init__(self) -> None:
        self.t = datetime(2026, 1, 1, tzinfo=UTC)
        self.sleepers: list[tuple[datetime, asyncio.Future[None]]] = []

    def now(self) -> datetime:
        return self.t

    async def sleep(self, seconds: float) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.t + timedelta(seconds=seconds), fut))
        await fut

    async def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)
        due = [f for d, f in self.sleepers if d <= self.t]
        self.sleepers = [(d, f) for d, f in self.sleepers if d > self.t]
        for f in due:
            if not f.done():
                f.set_result(None)
        await asyncio.sleep(0.02)  # laisse le battement écrire (SQLite : thread)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(bouture.agents, "now", c.now)
    monkeypatch.setattr(bouture.turn, "_sleep", c.sleep)
    return c


def long_definition(clock: Clock) -> AgentDefinition[Note, Any]:
    d = AgentDefinition[Note, Any]("long", model="fake", state_type=Note)

    @d.tool(timeout=None)
    async def render(ctx: ToolContext[Note, Any]) -> str:
        """Rendu qui dure dix fois le bail."""
        ctx.state.title = "pendant"
        await clock.sleep(10 * TTL)
        ctx.state.title = "rendu"
        return "fini"

    return d


async def _start(a: Agents[Any], clock: Clock) -> tuple[UUID, Any]:
    a.register(long_definition(clock))
    await a.setup()
    s = await a.create_session("long", OWNER)
    h = await a.send_message(s.id, OWNER, "rends")
    async for ev in h.events():
        if ev.type == "tool_call":
            break
    await asyncio.sleep(0.02)
    return s.id, h


async def _lease(a: Agents[Any], sid: UUID) -> datetime | None:
    async with a.storage.transaction() as tx:
        s = await tx.get_session(sid)
    assert s is not None
    return s.lock_expires_at


def _render_turn() -> FakeModelClient:
    return FakeModelClient([FakeReply("", tool_calls=[("render", {})]), FakeReply("Voilà.")])


async def test_tool_lasting_ten_leases_keeps_the_lease(
    make_agents: AgentsFactory, clock: Clock
) -> None:
    a = make_agents(_render_turn(), lock_ttl=TTL)
    sid, h = await _start(a, clock)
    for _ in range(30):  # dix baux, par pas de battement
        await clock.advance(TTL / 3)
        # Un autre lecteur qui tente la reprise ne trouve jamais le bail expiré.
        await a._recover_if_expired(sid)  # pyright: ignore[reportPrivateUsage]
        lease = await _lease(a, sid)
        if lease is None:  # l'outil a fini, le tour est clos
            break
        assert lease > clock.t
    end = await h.wait()
    assert end.data["reason"] == "completed"
    branch = await a.get_branch(sid, OWNER)
    assert [e.content["content"] for e in branch if e.type == "tool_result"] == ["fini"]
    assert (await a.get_state(sid, OWNER)).state.title == "rendu"


async def test_crash_during_long_tool(make_agents: AgentsFactory, clock: Clock) -> None:
    a1 = make_agents(_render_turn(), lock_ttl=TTL)
    sid, _ = await _start(a1, clock)
    await clock.advance(3 * TTL)  # l'outil tourne depuis trois baux
    # Le processus meurt : plus de battement, rien n'est écrit.
    for ex in list(a1._executors.values()):  # pyright: ignore[reportPrivateUsage]
        ex.cancel("abandon")
    await a1.runner.join(1)

    a2 = make_agents(FakeModelClient([FakeReply("Reprise.")]), lock_ttl=TTL)
    a2.register(long_definition(clock))
    await a2.setup()
    await clock.advance(TTL / 2)
    with pytest.raises(SessionBusy):  # bail encore valide
        await a2.send_message(sid, OWNER, "encore")
    await clock.advance(TTL)
    end = await (await a2.send_message(sid, OWNER, "encore")).wait()
    assert end.data["reason"] == "completed"

    turns = [e for e in await a2.get_branch(sid, OWNER) if e.type == "tool_result"]
    assert turns[0].content["is_error"]
    assert turns[0].content["content"].startswith("Interrompu")
    assert (await a2.get_state(sid, OWNER)).state.title == ""  # changement de l'outil perdu


async def test_stop_during_long_tool(make_agents: AgentsFactory, clock: Clock) -> None:
    a = make_agents(_render_turn(), lock_ttl=TTL)
    sid, h = await _start(a, clock)
    await clock.advance(5 * TTL)
    assert await _lease(a, sid) is not None
    await a.stop(sid, OWNER)
    end = await h.wait()
    assert end.data["reason"] == "interrupted"
    branch = await a.get_branch(sid, OWNER)
    assert branch[-1].type == "tool_result" and branch[-1].content["is_error"]
    assert (await a.get_state(sid, OWNER)).state.title == ""
    assert await _lease(a, sid) is None


# -- délais (D121) --------------------------------------------------------------------
def test_timeout_values() -> None:
    d = AgentDefinition[None, Any]("t", model="fake", tool_timeout=None)
    assert d.tool_timeout is None
    with pytest.raises(ValueError, match="tool_timeout"):
        AgentDefinition[None, Any]("t", model="fake", tool_timeout=0)
    with pytest.raises(ValueError, match="positif"):

        @d.tool(timeout=-1)
        async def bad(ctx: ToolContext[None, Any]) -> str:  # pyright: ignore[reportUnusedFunction]
            """Jamais enregistré."""
            return ""


async def test_none_means_no_limit(make_agents: AgentsFactory) -> None:
    d = AgentDefinition[Note, Any]("timed", model="fake", state_type=Note, tool_timeout=0.05)

    @d.tool(timeout=None)
    async def unlimited(ctx: ToolContext[Note, Any]) -> str:
        """Plus long que le délai de la définition, sans limite."""
        await asyncio.sleep(0.2)
        return "ok"

    @d.tool
    async def inherited(ctx: ToolContext[Note, Any]) -> str:
        """Hérite des 0,05 s de la définition."""
        await asyncio.sleep(0.2)
        return "jamais"

    fake = FakeModelClient(
        [FakeReply("", tool_calls=[("unlimited", {}), ("inherited", {})]), FakeReply("fin")]
    )
    a = make_agents(fake)
    a.register(d)
    await a.setup()
    s = await a.create_session("timed", OWNER)
    await (await a.send_message(s.id, OWNER, "x")).wait()
    results = [e.content for e in await a.get_branch(s.id, OWNER) if e.type == "tool_result"]
    assert results[0]["content"] == "ok" and not results[0]["is_error"]
    assert results[1]["is_error"] and "délai dépassé (0.05 s)" in results[1]["content"]

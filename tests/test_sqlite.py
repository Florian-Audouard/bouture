"""Spécificités du stockage SQLite (D107–D111).

La conformité au protocole ``Storage`` est couverte par toute la suite (fixture ``storage``) ;
ici : migrations, réglages du fichier, chemins Windows, plusieurs connecteurs sur un fichier.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from bouture import FakeModelClient, SchemaOutdated, SessionBusy
from bouture.models.fake import FakeReply
from bouture.storage.sqlite import SCHEMA_VERSION, SQLiteStorage, apply_migrations, schema_version

from .conftest import AgentsFactory, collect

OWNER = "alice"


async def test_check_mode_does_not_create_the_file(tmp_path: Path) -> None:
    path = tmp_path / "absente.db"
    st = SQLiteStorage(path)
    with pytest.raises(SchemaOutdated):
        await st.setup(auto_migrate=False)
    assert not path.exists()
    await st.close()


async def test_check_mode_refuses_a_foreign_database(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    sync = create_engine(URL.create("sqlite", database=str(path)))
    with sync.begin() as c:
        c.exec_driver_sql("CREATE TABLE app_images (id integer PRIMARY KEY)")
    sync.dispose()
    st = SQLiteStorage(path)
    with pytest.raises(SchemaOutdated):
        await st.setup(auto_migrate=False)
    await st.close()


async def test_migrations_from_a_sync_connection_like_alembic(tmp_path: Path) -> None:
    """Comme ``op.get_bind()`` dans une révision Alembic : pilote ``sqlite3`` synchrone."""
    path = tmp_path / "app.db"
    sync = create_engine(URL.create("sqlite", database=str(path)))
    with sync.begin() as c:
        c.exec_driver_sql("CREATE TABLE app_images (id integer PRIMARY KEY)")
        assert schema_version(c) == 0
        assert apply_migrations(c) == SCHEMA_VERSION
    with sync.begin() as c:
        assert apply_migrations(c) == SCHEMA_VERSION  # idempotent
        assert schema_version(c) == SCHEMA_VERSION
        tables = {
            r[0] for r in c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    sync.dispose()
    # Les tables de l'hôte et celles de bouture cohabitent ; celles de bouture sont préfixées.
    assert "app_images" in tables
    assert {t for t in tables if t != "app_images" and t != "sqlite_sequence"} == {
        t for t in tables if t.startswith("bouture_")
    }
    st = SQLiteStorage(path)
    await st.setup(auto_migrate=False)
    await st.close()


async def test_wal_and_foreign_keys(tmp_path: Path) -> None:
    st = SQLiteStorage(tmp_path / "b.db")
    await st.setup()
    async with st.engine.connect() as conn:
        assert (await conn.exec_driver_sql("PRAGMA journal_mode")).scalar() == "wal"
        assert (await conn.exec_driver_sql("PRAGMA foreign_keys")).scalar() == 1
    await st.close()


async def test_awkward_path(make_agents: AgentsFactory, tmp_path: Path) -> None:
    """Espaces, ``#``, ``%``, accents, chemin relatif : jamais interprétés comme une URL."""
    folder = tmp_path / "Mes documents" / "studio #1 100% été"
    st = SQLiteStorage(folder / "base de données.db")
    a = make_agents(FakeModelClient([FakeReply("bonjour")]), store=st)
    await a.setup()
    s = await a.create_session("note", OWNER)
    end = await (await a.send_message(s.id, OWNER, "x")).wait()
    assert end.data["reason"] == "completed"
    assert (folder / "base de données.db").exists()


async def test_many_concurrent_turns_in_one_process(
    make_agents: AgentsFactory, tmp_path: Path
) -> None:
    """Les tâches du processus partagent le fichier sans « database is locked »."""
    st = SQLiteStorage(tmp_path / "b.db")
    fake = FakeModelClient(default=FakeReply("", tool_calls=[("set_title", {"title": "t"})]))
    a = make_agents(fake, store=st)
    a.definition("note").max_steps = 3
    await a.setup()
    sessions = await asyncio.gather(*(a.create_session("note", OWNER) for _ in range(12)))
    turns = await asyncio.gather(*(a.send_message(s.id, OWNER, "x") for s in sessions))
    ends = await asyncio.gather(*(t.wait() for t in turns))
    assert {e.data["reason"] for e in ends} == {"max_steps"}


# -- Deux connecteurs sur un fichier (D25 reste vrai ; réveil par sondage, D110) -------------
async def _two(make_agents: AgentsFactory, path: Path, first: FakeModelClient, **kw: Any) -> Any:
    a1 = make_agents(first, store=SQLiteStorage(path), **kw)
    a2 = make_agents(FakeModelClient([FakeReply("reprise")]), store=SQLiteStorage(path), **kw)
    await a1.setup()
    await a2.setup()
    return a1, a2


async def test_lease_holds_across_connections(make_agents: AgentsFactory, tmp_path: Path) -> None:
    fake = FakeModelClient([FakeReply("long", hang=True)])
    a1, a2 = await _two(make_agents, tmp_path / "b.db", fake)
    s = await a1.create_session("note", OWNER)
    await a1.send_message(s.id, OWNER, "x")
    await fake.started.wait()
    with pytest.raises(SessionBusy):
        await a2.send_message(s.id, OWNER, "y")
    await a1.stop(s.id, OWNER)


async def test_other_connection_follows_by_polling(
    make_agents: AgentsFactory, tmp_path: Path
) -> None:
    a1, a2 = await _two(make_agents, tmp_path / "b.db", FakeModelClient([FakeReply("salut")]))
    s = await a1.create_session("note", OWNER)
    seen: list[str] = []
    listening = asyncio.Event()

    async def follow() -> None:
        sub = a2.subscribe(s.id, OWNER)
        listening.set()
        async for ev in sub:
            seen.append(ev.type)
            if ev.type == "turn_end":
                return

    task = asyncio.create_task(follow())
    await listening.wait()
    await asyncio.sleep(0.1)  # l'abonné attend : seul le sondage peut le réveiller
    end = await (await a1.send_message(s.id, OWNER, "x")).wait()
    await asyncio.wait_for(task, 5)
    assert seen[0] == "turn_start" and seen[-1] == "turn_end"
    assert end.data["reason"] == "completed"


async def test_stop_from_other_connection_is_applied_by_heartbeat(
    make_agents: AgentsFactory, tmp_path: Path
) -> None:
    """Sans NOTIFY, le stop d'un autre connecteur passe par ``cancel_requested`` (bail / 3)."""
    fake = FakeModelClient([FakeReply("abc", hang=True)])
    a1, a2 = await _two(make_agents, tmp_path / "b.db", fake, lock_ttl=0.3)
    s = await a1.create_session("note", OWNER)
    turn = await a1.send_message(s.id, OWNER, "x")
    await fake.started.wait()
    await a2.stop(s.id, OWNER)
    end = await asyncio.wait_for(turn.wait(), 5)
    assert end.data["reason"] == "interrupted"


async def test_crash_recovered_by_other_connection(
    make_agents: AgentsFactory, tmp_path: Path
) -> None:
    fake = FakeModelClient([FakeReply("partiel", hang=True)])
    a1, a2 = await _two(make_agents, tmp_path / "b.db", fake, lock_ttl=0.3)
    s = await a1.create_session("note", OWNER)
    await a1.send_message(s.id, OWNER, "x")
    await fake.started.wait()
    await asyncio.sleep(0.15)
    for ex in list(a1._executors.values()):  # pyright: ignore[reportPrivateUsage]
        ex.cancel("abandon")
    await a1.runner.join(1)
    await asyncio.sleep(0.4)
    evs = await collect((await a2.send_message(s.id, OWNER, "encore")).events())
    assert evs[-1].data["reason"] == "completed"
    branch = await a2.get_branch(s.id, OWNER)
    assert branch[2].interrupted and branch[2].content["content"][0]["text"] == "partiel"

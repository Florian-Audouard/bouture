"""Stockage Postgres via SQLAlchemy Core async (D70).

Dialectes acceptés : ``postgresql+asyncpg`` et ``postgresql+psycopg`` (LISTEN/NOTIFY).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import datetime
from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger as SABigInteger
from sqlalchemy import (
    Boolean,
    Column,
    Connection,
    DateTime,
    Integer,
    MetaData,
    Table,
    Text,
    and_,
    delete,
    func,
    insert,
    or_,
    select,
    text,
    tuple_,
    update,
)
from sqlalchemy import Uuid as SAUuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession

from ..errors import SchemaOutdated
from .base import (
    CancelCallback,
    ChildResultRecord,
    EntryRecord,
    EventRecord,
    EventsCallback,
    InputRecord,
    NewEvent,
    PublicationRecord,
    SessionRecord,
    StateRecord,
    TurnRecord,
)

log = logging.getLogger("bouture")

SCHEMA = "bouture"
EVENTS_CHANNEL = "bouture_events"
CANCEL_CHANNEL = "bouture_cancel"
_TS = DateTime(timezone=True)

md = MetaData(schema=SCHEMA)

sessions_t = Table(
    "sessions",
    md,
    Column("id", SAUuid, primary_key=True),
    Column("kind", Text),
    Column("owner", Text),
    Column("title", Text),
    Column("metadata", JSONB),
    Column("model_override", Text),
    Column("model_settings", JSONB),
    Column("leaf_entry_id", SAUuid),
    Column("object_id", SAUuid),
    Column("cloned_from_session_id", SAUuid),
    Column("cloned_from_entry_id", SAUuid),
    Column("origin_session_id", SAUuid),
    Column("origin_entry_id", SAUuid),
    Column("origin_handler", Text),
    Column("origin_payload", JSONB),
    Column("pending", JSONB),
    Column("running_turn_id", SAUuid),
    Column("lock_expires_at", _TS),
    Column("cancel_requested", Boolean),
    Column("archived_at", _TS),
    Column("created_at", _TS),
    Column("updated_at", _TS),
)
inputs_t = Table(
    "session_inputs",
    md,
    Column("session_id", SAUuid, primary_key=True),
    Column("name", Text, primary_key=True),
    Column("mode", Text),
    Column("kind", Text),
    Column("object_id", SAUuid),
    Column("version", Integer),
    Column("data", JSONB),
    Column("data_version", Integer),
)
states_t = Table(
    "states",
    md,
    Column("id", SAUuid, primary_key=True),
    Column("kind", Text),
    Column("state_version", Integer),
    Column("data", JSONB),
    Column("created_at", _TS),
)
entries_t = Table(
    "entries",
    md,
    Column("id", SAUuid, primary_key=True),
    Column("session_id", SAUuid),
    Column("parent_id", SAUuid),
    Column("type", Text),
    Column("content", JSONB),
    Column("state_id", SAUuid),
    Column("is_checkpoint", Boolean),
    Column("interrupted", Boolean),
    Column("turn_id", SAUuid),
    Column("usage", JSONB),
    Column("created_at", _TS),
    Column("label", Text),
    Column("hidden_at", _TS),
    Column("metadata", JSONB),
    Column("derived_from", SAUuid),
)
turns_t = Table(
    "turns",
    md,
    Column("id", SAUuid, primary_key=True),
    Column("session_id", SAUuid),
    Column("trigger", Text),
    Column("status", Text),
    Column("parent_entry_id", SAUuid),
    Column("usage", JSONB),
    Column("error", JSONB),
    Column("started_at", _TS),
    Column("ended_at", _TS),
)
events_t = Table(
    "turn_events",
    md,
    Column("id", SABigInteger, primary_key=True, autoincrement=True),
    Column("session_id", SAUuid),
    Column("turn_id", SAUuid),
    Column("type", Text),
    Column("data", JSONB),
    Column("created_at", _TS),
)
publications_t = Table(
    "publications",
    md,
    Column("kind", Text),
    Column("object_id", SAUuid, primary_key=True),
    Column("version", Integer, primary_key=True),
    Column("session_id", SAUuid),
    Column("entry_id", SAUuid),
    Column("state_id", SAUuid),
    Column("owner", Text),
    Column("published_at", _TS),
)
child_results_t = Table(
    "pending_child_results",
    md,
    Column("id", SAUuid, primary_key=True),
    Column("parent_session_id", SAUuid),
    Column("child_session_id", SAUuid),
    Column("object_id", SAUuid),
    Column("version", Integer),
    Column("created_at", _TS),
    Column("applied_at", _TS),
    Column("applied_entry_id", SAUuid),
)


def _state_used() -> Any:
    """Le snapshot est référencé par une entrée ou une publication."""
    s = states_t.c
    used_e = select(entries_t.c.id).where(entries_t.c.state_id == s.id).exists()
    used_p = select(publications_t.c.object_id).where(publications_t.c.state_id == s.id).exists()
    return or_(used_e, used_p)


def _row_to[T](cls: type[T], row: Any) -> T:
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in row._mapping.items() if k in names})


def _rec_to_row(rec: Any) -> dict[str, Any]:
    return {f.name: getattr(rec, f.name) for f in fields(rec)}


def _migrations() -> list[tuple[int, str]]:
    pkg = resources.files("bouture.storage.migrations")
    out: list[tuple[int, str]] = []
    for item in pkg.iterdir():
        if item.name.endswith(".sql"):
            out.append((int(item.name.split("_", 1)[0]), item.read_text(encoding="utf-8")))
    return sorted(out)


SCHEMA_VERSION = _migrations()[-1][0]
"""Version de schéma attendue par cette version de bouture."""

_LOCK_ID = 7342981120


def schema_version(connection: Connection) -> int:
    """Version de schéma appliquée (0 si bouture n'a jamais été installé). Lecture seule."""
    reg = connection.execute(text(f"SELECT to_regclass('{SCHEMA}.schema_version')")).scalar()
    if reg is None:
        return 0
    return connection.execute(
        text(f"SELECT coalesce(max(version), 0) FROM {SCHEMA}.schema_version")
    ).scalar_one()


def apply_migrations(connection: Connection, target: int | None = None) -> int:
    """Applique les migrations bouture manquantes jusqu'à ``target`` (défaut : la dernière).

    Connexion **synchrone** dans une transaction ouverte : celle d'Alembic (``op.get_bind()``)
    ou celle que passe ``AsyncConnection.run_sync``. Idempotent, sérialisé par verrou
    consultatif. Renvoie la version atteinte. Pas de retour arrière (D103).
    """
    migrations = _migrations()
    goal = migrations[-1][0] if target is None else target
    connection.execute(text(f"SELECT pg_advisory_xact_lock({_LOCK_ID})"))
    current = schema_version(connection)
    if current >= goal:
        return current
    connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    connection.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_version ("
            "version int PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
    )
    for version, sql in migrations:
        if version <= current or version > goal:
            continue
        for statement in (s.strip() for s in sql.split(";")):
            if statement:
                connection.exec_driver_sql(statement)
        connection.execute(
            text(f"INSERT INTO {SCHEMA}.schema_version (version) VALUES (:v)"), {"v": version}
        )
        log.info("bouture: migration %s appliquée", version)
        current = version
    return current


class PostgresStorage:
    def __init__(self, engine: AsyncEngine) -> None:
        driver = engine.dialect.driver
        if engine.dialect.name != "postgresql" or driver not in ("asyncpg", "psycopg"):
            raise ValueError(
                "bouture exige un moteur postgresql+asyncpg ou postgresql+psycopg "
                f"(reçu {engine.dialect.name}+{driver})"
            )
        self.engine = engine
        self._driver = driver
        self._listen_conn: AsyncConnection | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._listeners: list[tuple[EventsCallback, CancelCallback]] = []

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[_PgTx]:
        async with self.engine.begin() as conn:
            tx = _PgTx(conn)
            try:
                yield tx
                if tx.session_obj is not None:
                    await tx.session_obj.flush()
                for sid in tx.notified_sessions:
                    await conn.execute(select(func.pg_notify(EVENTS_CHANNEL, str(sid))))
                for tid in tx.notified_cancels:
                    await conn.execute(select(func.pg_notify(CANCEL_CHANNEL, str(tid))))
            finally:
                if tx.session_obj is not None:
                    await tx.session_obj.close()

    async def setup(self, *, auto_migrate: bool = True) -> None:
        """``auto_migrate=False`` : vérification en lecture seule, ``SchemaOutdated`` si en retard."""
        async with self.engine.begin() as conn:
            if auto_migrate:
                await conn.run_sync(apply_migrations)
                return
            current = await conn.run_sync(schema_version)
            if current < SCHEMA_VERSION:
                raise SchemaOutdated(
                    f"schéma bouture en version {current}, attendu {SCHEMA_VERSION}"
                )

    async def listen(
        self, on_events: EventsCallback, on_cancel: CancelCallback
    ) -> Callable[[], Awaitable[None]]:
        pair = (on_events, on_cancel)
        self._listeners.append(pair)
        if self._listen_conn is None:
            await self._start_listener()

        async def unlisten() -> None:
            if pair in self._listeners:
                self._listeners.remove(pair)

        return unlisten

    def _dispatch(self, channel: str, payload: str) -> None:
        try:
            ident = UUID(payload)
        except ValueError:
            return
        for on_events, on_cancel in list(self._listeners):
            (on_events if channel == EVENTS_CHANNEL else on_cancel)(ident)

    async def _start_listener(self) -> None:
        conn = await self.engine.connect()
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        raw = await conn.get_raw_connection()
        driver_conn: Any = raw.driver_connection
        self._listen_conn = conn
        if self._driver == "asyncpg":

            def _cb(_c: Any, _pid: int, channel: str, payload: str) -> None:
                self._dispatch(channel, payload)

            await driver_conn.add_listener(EVENTS_CHANNEL, _cb)
            await driver_conn.add_listener(CANCEL_CHANNEL, _cb)
        else:
            await driver_conn.execute(f"LISTEN {EVENTS_CHANNEL}")
            await driver_conn.execute(f"LISTEN {CANCEL_CHANNEL}")

            async def _loop() -> None:
                async for n in driver_conn.notifies():
                    self._dispatch(n.channel, n.payload)

            self._listen_task = asyncio.create_task(_loop())

    async def close(self) -> None:
        if self._listen_task is not None:
            self._listen_task.cancel()
            self._listen_task = None
        if self._listen_conn is not None:
            await self._listen_conn.close()
            self._listen_conn = None


class _PgTx:
    def __init__(self, conn: AsyncConnection) -> None:
        self.conn = conn
        self.session_obj: AsyncSession | None = None
        self.notified_sessions: list[UUID] = []
        self.notified_cancels: list[UUID] = []

    @property
    def db(self) -> AsyncSession:
        if self.session_obj is None:
            self.session_obj = AsyncSession(bind=self.conn, expire_on_commit=False)
        return self.session_obj

    async def _all(self, stmt: Any) -> list[Any]:
        return list((await self.conn.execute(stmt)).fetchall())

    # sessions -------------------------------------------------------------------------
    async def get_session(
        self, session_id: UUID, *, for_update: bool = False
    ) -> SessionRecord | None:
        stmt = select(sessions_t).where(sessions_t.c.id == session_id)
        if for_update:
            stmt = stmt.with_for_update()
        rows = await self._all(stmt)
        return _row_to(SessionRecord, rows[0]) if rows else None

    async def insert_session(self, rec: SessionRecord) -> None:
        await self.conn.execute(insert(sessions_t).values(**_rec_to_row(rec)))

    async def update_session(self, session_id: UUID, **fields_: Any) -> None:
        await self.conn.execute(
            update(sessions_t).where(sessions_t.c.id == session_id).values(**fields_)
        )

    async def list_sessions(
        self,
        owner: str,
        *,
        kind: str | None,
        origin_session_id: UUID | None,
        archived: bool,
        limit: int,
        before: tuple[datetime, UUID] | None,
        object_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SessionRecord]:
        c = sessions_t.c
        conds = [
            c.owner == owner,
            c.archived_at.isnot(None) if archived else c.archived_at.is_(None),
        ]
        if kind is not None:
            conds.append(c.kind == kind)
        if origin_session_id is not None:
            conds.append(c.origin_session_id == origin_session_id)
        if before is not None:
            conds.append(tuple_(c.updated_at, c.id) < tuple_(*before))
        if object_id is not None:
            conds.append(c.object_id == object_id)
        if metadata is not None:
            conds.append(c.metadata.contains(metadata))
        stmt = (
            select(sessions_t)
            .where(and_(*conds))
            .order_by(c.updated_at.desc(), c.id.desc())
            .limit(limit)
        )
        return [_row_to(SessionRecord, r) for r in await self._all(stmt)]

    async def sessions_of_owner(self, owner: str) -> list[UUID]:
        return [
            r[0]
            for r in await self._all(select(sessions_t.c.id).where(sessions_t.c.owner == owner))
        ]

    # inputs ---------------------------------------------------------------------------
    async def insert_inputs(self, recs: Sequence[InputRecord]) -> None:
        if recs:
            await self.conn.execute(insert(inputs_t), [_rec_to_row(r) for r in recs])

    async def get_inputs(self, session_id: UUID) -> list[InputRecord]:
        rows = await self._all(select(inputs_t).where(inputs_t.c.session_id == session_id))
        return [_row_to(InputRecord, r) for r in rows]

    # états ----------------------------------------------------------------------------
    async def insert_states(self, recs: Sequence[StateRecord]) -> None:
        if recs:
            await self.conn.execute(insert(states_t), [_rec_to_row(r) for r in recs])

    async def get_states(self, ids: Sequence[UUID]) -> dict[UUID, StateRecord]:
        if not ids:
            return {}
        rows = await self._all(select(states_t).where(states_t.c.id.in_(list(set(ids)))))
        return {r.id: _row_to(StateRecord, r) for r in rows}

    async def delete_unused_states(self, ids: Sequence[UUID]) -> None:
        if ids:
            await self.conn.execute(
                delete(states_t).where(states_t.c.id.in_(list(set(ids))), ~_state_used())
            )

    # entrées --------------------------------------------------------------------------
    async def insert_entries(self, recs: Sequence[EntryRecord]) -> None:
        for r in recs:  # ordre parent → enfant requis par la FK
            await self.conn.execute(insert(entries_t).values(**_rec_to_row(r)))

    async def get_entries(self, session_id: UUID) -> list[EntryRecord]:
        stmt = (
            select(entries_t)
            .where(entries_t.c.session_id == session_id)
            .order_by(entries_t.c.created_at, entries_t.c.id)
        )
        return [_row_to(EntryRecord, r) for r in await self._all(stmt)]

    async def get_entry(self, entry_id: UUID) -> EntryRecord | None:
        rows = await self._all(select(entries_t).where(entries_t.c.id == entry_id))
        return _row_to(EntryRecord, rows[0]) if rows else None

    async def update_entry(self, entry_id: UUID, **fields_: Any) -> None:
        await self.conn.execute(
            update(entries_t).where(entries_t.c.id == entry_id).values(**fields_)
        )

    # tours ----------------------------------------------------------------------------
    async def insert_turn(self, rec: TurnRecord) -> None:
        await self.conn.execute(insert(turns_t).values(**_rec_to_row(rec)))

    async def update_turn(self, turn_id: UUID, **fields_: Any) -> None:
        await self.conn.execute(update(turns_t).where(turns_t.c.id == turn_id).values(**fields_))

    async def get_turn(self, turn_id: UUID) -> TurnRecord | None:
        rows = await self._all(select(turns_t).where(turns_t.c.id == turn_id))
        return _row_to(TurnRecord, rows[0]) if rows else None

    # publications ---------------------------------------------------------------------
    async def insert_publication(self, rec: PublicationRecord) -> None:
        await self.conn.execute(insert(publications_t).values(**_rec_to_row(rec)))

    async def get_publications(self, object_id: UUID) -> list[PublicationRecord]:
        stmt = (
            select(publications_t)
            .where(publications_t.c.object_id == object_id)
            .order_by(publications_t.c.version)
        )
        return [_row_to(PublicationRecord, r) for r in await self._all(stmt)]

    # résultats d'enfants --------------------------------------------------------------
    async def insert_child_result(self, rec: ChildResultRecord) -> None:
        await self.conn.execute(insert(child_results_t).values(**_rec_to_row(rec)))

    async def pending_child_results(self, parent_session_id: UUID) -> list[ChildResultRecord]:
        c = child_results_t.c
        stmt = (
            select(child_results_t)
            .where(c.parent_session_id == parent_session_id, c.applied_at.is_(None))
            .order_by(c.id)
        )
        return [_row_to(ChildResultRecord, r) for r in await self._all(stmt)]

    async def update_child_result(self, rec_id: UUID, **fields_: Any) -> None:
        await self.conn.execute(
            update(child_results_t).where(child_results_t.c.id == rec_id).values(**fields_)
        )

    # événements -----------------------------------------------------------------------
    async def append_events(self, events: Sequence[NewEvent]) -> list[EventRecord]:
        out: list[EventRecord] = []
        for ev in events:
            row = _rec_to_row(ev)
            new_id = (
                await self.conn.execute(insert(events_t).values(**row).returning(events_t.c.id))
            ).scalar_one()
            out.append(EventRecord(id=new_id, **row))
        return out

    async def read_events(
        self, session_id: UUID, *, after: int | None, turn_id: UUID | None = None, limit: int = 500
    ) -> list[EventRecord]:
        c = events_t.c
        conds = [c.session_id == session_id]
        if after is not None:
            conds.append(c.id > after)
        if turn_id is not None:
            conds.append(c.turn_id == turn_id)
        stmt = select(events_t).where(and_(*conds)).order_by(c.id).limit(limit)
        return [_row_to(EventRecord, r) for r in await self._all(stmt)]

    async def first_event_id(self, turn_id: UUID) -> int | None:
        return (
            await self.conn.execute(
                select(func.min(events_t.c.id)).where(events_t.c.turn_id == turn_id)
            )
        ).scalar_one_or_none()

    async def purge_events(self, ended_before: datetime) -> int:
        ended = select(turns_t.c.id).where(turns_t.c.ended_at < ended_before)
        res = await self.conn.execute(delete(events_t).where(events_t.c.turn_id.in_(ended)))
        return res.rowcount or 0

    # purge ----------------------------------------------------------------------------
    async def delete_sessions(self, ids: Sequence[UUID]) -> None:
        if not ids:
            return
        ids = list(ids)
        await self.conn.execute(delete(events_t).where(events_t.c.session_id.in_(ids)))
        await self.conn.execute(delete(sessions_t).where(sessions_t.c.id.in_(ids)))
        await self.conn.execute(delete(states_t).where(~_state_used()))

    # notifications --------------------------------------------------------------------
    def notify_events(self, session_id: UUID) -> None:
        if session_id not in self.notified_sessions:
            self.notified_sessions.append(session_id)

    def notify_cancel(self, turn_id: UUID) -> None:
        self.notified_cancels.append(turn_id)

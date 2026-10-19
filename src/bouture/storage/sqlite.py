"""Stockage SQLite via SQLAlchemy Core async et ``aiosqlite`` (D107–D111).

Pensé pour une application locale mono-utilisateur : **un seul processus** par fichier de base.
Pas de schéma nommé en SQLite, les tables sont préfixées ``bouture_`` ; l'hôte choisit le
fichier (le sien ou un fichier dédié). Mode WAL, transactions ``BEGIN IMMEDIATE``, pas de
LISTEN/NOTIFY : les abonnés du processus sont réveillés au commit, les autres par sondage (D83).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Connection,
    Dialect,
    Integer,
    MetaData,
    Table,
    Text,
    TypeDecorator,
    and_,
    delete,
    event,
    func,
    insert,
    literal,
    or_,
    select,
    text,
    tuple_,
    update,
)
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, create_async_engine

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
    json_contains,
)

log = logging.getLogger("bouture")

PREFIX = "bouture_"
_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class _UuidText(TypeDecorator[UUID]):
    """uuid stocké en texte canonique (``8-4-4-4-12``), lisible depuis le SQL de l'hôte."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: UUID | None, dialect: Dialect) -> str | None:
        return None if value is None else str(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> UUID | None:
        return None if value is None else UUID(value)


class _UtcText(TypeDecorator[datetime]):
    """Date UTC en texte à largeur fixe : l'ordre du texte est celui des dates."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).strftime(_TS_FORMAT)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=UTC)


_ID = _UuidText()
_TS = _UtcText()

md = MetaData()

sessions_t = Table(
    f"{PREFIX}sessions",
    md,
    Column("id", _ID, primary_key=True),
    Column("kind", Text),
    Column("owner", Text),
    Column("title", Text),
    Column("metadata", JSON),
    Column("model_override", Text),
    Column("model_settings", JSON),
    Column("leaf_entry_id", _ID),
    Column("object_id", _ID),
    Column("cloned_from_session_id", _ID),
    Column("cloned_from_entry_id", _ID),
    Column("origin_session_id", _ID),
    Column("origin_entry_id", _ID),
    Column("origin_handler", Text),
    Column("origin_payload", JSON),
    Column("pending", JSON),
    Column("running_turn_id", _ID),
    Column("lock_expires_at", _TS),
    Column("cancel_requested", Boolean),
    Column("archived_at", _TS),
    Column("created_at", _TS),
    Column("updated_at", _TS),
)
inputs_t = Table(
    f"{PREFIX}session_inputs",
    md,
    Column("session_id", _ID, primary_key=True),
    Column("name", Text, primary_key=True),
    Column("mode", Text),
    Column("kind", Text),
    Column("object_id", _ID),
    Column("version", Integer),
    Column("data", JSON),
    Column("data_version", Integer),
)
states_t = Table(
    f"{PREFIX}states",
    md,
    Column("id", _ID, primary_key=True),
    Column("kind", Text),
    Column("state_version", Integer),
    Column("data", JSON),
    Column("created_at", _TS),
)
entries_t = Table(
    f"{PREFIX}entries",
    md,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID),
    Column("parent_id", _ID),
    Column("type", Text),
    Column("content", JSON),
    Column("state_id", _ID),
    Column("is_checkpoint", Boolean),
    Column("interrupted", Boolean),
    Column("turn_id", _ID),
    Column("usage", JSON),
    Column("created_at", _TS),
    Column("label", Text),
    Column("hidden_at", _TS),
    Column("metadata", JSON),
    Column("derived_from", _ID),
)
turns_t = Table(
    f"{PREFIX}turns",
    md,
    Column("id", _ID, primary_key=True),
    Column("session_id", _ID),
    Column("trigger", Text),
    Column("status", Text),
    Column("parent_entry_id", _ID),
    Column("usage", JSON),
    Column("error", JSON),
    Column("started_at", _TS),
    Column("ended_at", _TS),
)
events_t = Table(
    f"{PREFIX}turn_events",
    md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", _ID),
    Column("turn_id", _ID),
    Column("type", Text),
    Column("data", JSON),
    Column("created_at", _TS),
)
publications_t = Table(
    f"{PREFIX}publications",
    md,
    Column("kind", Text),
    Column("object_id", _ID, primary_key=True),
    Column("version", Integer, primary_key=True),
    Column("session_id", _ID),
    Column("entry_id", _ID),
    Column("state_id", _ID),
    Column("owner", Text),
    Column("published_at", _TS),
)
child_results_t = Table(
    f"{PREFIX}pending_child_results",
    md,
    Column("id", _ID, primary_key=True),
    Column("parent_session_id", _ID),
    Column("child_session_id", _ID),
    Column("object_id", _ID),
    Column("version", Integer),
    Column("created_at", _TS),
    Column("applied_at", _TS),
    Column("applied_entry_id", _ID),
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


def _sql_json_contains(doc: str | None, sub: str | None) -> int:
    """Fonction SQL ``bouture_json_contains`` : même code que la mémoire (D109)."""
    if doc is None or sub is None:
        return 0
    return int(json_contains(json.loads(doc), json.loads(sub)))


# migrations -------------------------------------------------------------------------------
def _migrations() -> list[tuple[int, str]]:
    pkg = resources.files("bouture.storage.sqlite_migrations")
    out: list[tuple[int, str]] = []
    for item in pkg.iterdir():
        if item.name.endswith(".sql"):
            out.append((int(item.name.split("_", 1)[0]), item.read_text(encoding="utf-8")))
    return sorted(out)


def _statements(sql: str) -> list[str]:
    body = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


SCHEMA_VERSION = _migrations()[-1][0]
"""Version de schéma SQLite attendue par cette version de bouture (propre au dialecte)."""

_VERSION_TABLE = f"{PREFIX}schema_version"


def schema_version(connection: Connection) -> int:
    """Version de schéma appliquée (0 si bouture n'a jamais été installé). Lecture seule."""
    found = connection.exec_driver_sql(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_VERSION_TABLE,)
    ).first()
    if found is None:
        return 0
    return connection.exec_driver_sql(
        f"SELECT coalesce(max(version), 0) FROM {_VERSION_TABLE}"
    ).scalar_one()


def apply_migrations(connection: Connection, target: int | None = None) -> int:
    """Applique les migrations SQLite manquantes jusqu'à ``target`` (défaut : la dernière).

    Connexion **synchrone** : celle d'Alembic (``op.get_bind()``) ou celle que passe
    ``AsyncConnection.run_sync``. Idempotent ; renvoie la version atteinte. Pas de retour
    arrière. SQLite n'a pas de verrou consultatif : la sérialisation vient de la transaction
    d'écriture (``BEGIN IMMEDIATE`` dans ``SQLiteStorage.setup``), et chaque instruction est
    rejouable (``IF NOT EXISTS``) pour le cas où le pilote de l'hôte applique le DDL hors
    transaction (comportement par défaut de ``sqlite3``).
    """
    migrations = _migrations()
    goal = migrations[-1][0] if target is None else target
    current = schema_version(connection)
    if current >= goal:
        return current
    connection.exec_driver_sql(
        f"CREATE TABLE IF NOT EXISTS {_VERSION_TABLE} ("
        "version integer PRIMARY KEY, applied_at text NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    for version, sql in migrations:
        if version <= current or version > goal:
            continue
        for statement in _statements(sql):
            connection.exec_driver_sql(statement)
        connection.execute(
            text(f"INSERT INTO {_VERSION_TABLE} (version) VALUES (:v)"), {"v": version}
        )
        log.info("bouture: migration sqlite %s appliquée", version)
        current = version
    return current


# stockage ---------------------------------------------------------------------------------
class SQLiteStorage:
    """Stockage SQLite d'un fichier. Crée et possède son moteur ``sqlite+aiosqlite``.

    ``path`` : chemin du fichier (``str`` ou ``pathlib.Path``, espaces acceptés). Le dossier
    parent est créé au besoin. ``busy_timeout`` : attente maximale du verrou d'écriture quand
    un autre connecteur (l'application hôte, un outil) écrit dans le même fichier.
    """

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout: float = 30.0) -> None:
        self.path = Path(path).expanduser().resolve()
        self.busy_timeout = busy_timeout
        # URL.create : le chemin n'est jamais analysé comme une URL (Windows, espaces, ``#``).
        self.engine: AsyncEngine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(self.path))
        )
        self._configure(self.engine)
        self._lock = asyncio.Lock()
        self._listeners: list[tuple[EventsCallback, CancelCallback]] = []

    def _configure(self, engine: AsyncEngine) -> None:
        timeout_ms = int(self.busy_timeout * 1000)

        @event.listens_for(engine.sync_engine, "connect")
        def _connect(dbapi_conn: Any, _record: Any) -> None:  # pyright: ignore[reportUnusedFunction]
            # Le module sqlite3 n'émet pas de BEGIN de lui-même : c'est ``_begin`` qui le fait.
            dbapi_conn.isolation_level = None
            cur = dbapi_conn.cursor()
            cur.execute(f"PRAGMA busy_timeout = {timeout_ms}")
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.execute("PRAGMA synchronous = NORMAL")
            cur.close()
            dbapi_conn.create_function(
                "bouture_json_contains", 2, _sql_json_contains, deterministic=True
            )

        @event.listens_for(engine.sync_engine, "begin")
        def _begin(conn: Any) -> None:  # pyright: ignore[reportUnusedFunction]
            # Verrou d'écriture pris dès l'ouverture : la lecture ``for_update`` du bail (D25)
            # et l'écriture qui suit ne peuvent pas être entrelacées avec un autre écrivain.
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[_SqliteTx]:
        # Le verrou asyncio évite que les tâches du processus se disputent le verrou SQLite
        # (busy_timeout) ; BEGIN IMMEDIATE protège contre les autres connecteurs du fichier.
        async with self._lock, self.engine.begin() as conn:
            tx = _SqliteTx(conn)
            try:
                yield tx
                if tx.session_obj is not None:
                    await tx.session_obj.flush()
            finally:
                if tx.session_obj is not None:
                    await tx.session_obj.close()
        for session_id in tx.notified_sessions:
            for on_events, _ in list(self._listeners):
                on_events(session_id)
        for turn_id in tx.notified_cancels:
            for _, on_cancel in list(self._listeners):
                on_cancel(turn_id)

    async def setup(self, *, auto_migrate: bool = True) -> None:
        """``auto_migrate=False`` : vérification en lecture seule, ``SchemaOutdated`` si en retard.

        En vérification, un fichier absent n'est pas créé.
        """
        if not auto_migrate:
            if not self.path.exists():
                raise SchemaOutdated(
                    f"base SQLite {self.path} absente, schéma attendu {SCHEMA_VERSION}"
                )
            async with self._lock, self.engine.connect() as conn:
                current = await conn.run_sync(schema_version)
            if current < SCHEMA_VERSION:
                raise SchemaOutdated(
                    f"schéma bouture SQLite en version {current}, attendu {SCHEMA_VERSION}"
                )
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock, self.engine.begin() as conn:
            await conn.run_sync(apply_migrations)

    async def listen(
        self, on_events: EventsCallback, on_cancel: CancelCallback
    ) -> Callable[[], Awaitable[None]]:
        pair = (on_events, on_cancel)
        self._listeners.append(pair)

        async def unlisten() -> None:
            if pair in self._listeners:
                self._listeners.remove(pair)

        return unlisten

    async def close(self) -> None:
        """Ferme les connexions du pool ; le stockage reste utilisable (reconnexion à la demande)."""
        await self.engine.dispose()


class _SqliteTx:
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
        # ``for_update`` : rien à faire, la transaction tient déjà le verrou d'écriture.
        rows = await self._all(select(sessions_t).where(sessions_t.c.id == session_id))
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
            conds.append(
                tuple_(c.updated_at, c.id)
                < tuple_(literal(before[0], _TS), literal(before[1], _ID))
            )
        if object_id is not None:
            conds.append(c.object_id == object_id)
        if metadata is not None:
            conds.append(
                func.bouture_json_contains(c.metadata, literal(json.dumps(metadata), Text)) == 1
            )
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
            res = await self.conn.execute(insert(events_t).values(**row))
            pk = res.inserted_primary_key
            assert pk is not None
            out.append(EventRecord(id=pk[0], **row))
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

    # notifications (processus courant uniquement, au commit) --------------------------
    def notify_events(self, session_id: UUID) -> None:
        if session_id not in self.notified_sessions:
            self.notified_sessions.append(session_id)

    def notify_cancel(self, turn_id: UUID) -> None:
        self.notified_cancels.append(turn_id)

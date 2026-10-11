"""Stockage en mémoire, pour les tests.

Les transactions sont sérialisées par un verrou global ; en cas d'exception, les tables sont
restaurées depuis la copie prise à l'ouverture. Plusieurs instances ``Agents`` peuvent partager
la même instance (simulation de plusieurs processus ou d'un redémarrage).
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from typing import Any
from uuid import UUID

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

_TABLES = (
    "sessions",
    "inputs",
    "states",
    "entries",
    "turns",
    "publications",
    "child_results",
    "events",
)


class MemoryStorage:
    def __init__(self) -> None:
        self.sessions: dict[UUID, SessionRecord] = {}
        self.inputs: dict[tuple[UUID, str], InputRecord] = {}
        self.states: dict[UUID, StateRecord] = {}
        self.entries: dict[UUID, EntryRecord] = {}
        self.turns: dict[UUID, TurnRecord] = {}
        self.publications: dict[tuple[UUID, int], PublicationRecord] = {}
        self.child_results: dict[UUID, ChildResultRecord] = {}
        self.events: list[EventRecord] = []
        self._event_seq = 0
        self._lock = asyncio.Lock()
        self._listeners: list[tuple[EventsCallback, CancelCallback]] = []

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[_MemoryTx]:
        async with self._lock:
            backup = {name: copy.copy(getattr(self, name)) for name in _TABLES}
            backup_seq = self._event_seq
            tx = _MemoryTx(self)
            try:
                yield tx
            except BaseException:
                for name, value in backup.items():
                    setattr(self, name, value)
                self._event_seq = backup_seq
                raise
        for session_id in tx.notified_sessions:
            for on_events, _ in list(self._listeners):
                on_events(session_id)
        for turn_id in tx.notified_cancels:
            for _, on_cancel in list(self._listeners):
                on_cancel(turn_id)

    async def setup(self, *, auto_migrate: bool = True) -> None:
        return None

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
        return None


class _MemoryTx:
    def __init__(self, store: MemoryStorage) -> None:
        self._s = store
        self.notified_sessions: list[UUID] = []
        self.notified_cancels: list[UUID] = []

    @property
    def db(self) -> Any:
        return None

    # sessions -------------------------------------------------------------------------
    async def get_session(
        self, session_id: UUID, *, for_update: bool = False
    ) -> SessionRecord | None:
        rec = self._s.sessions.get(session_id)
        return copy.deepcopy(rec) if rec else None

    async def insert_session(self, rec: SessionRecord) -> None:
        self._s.sessions[rec.id] = copy.deepcopy(rec)

    async def update_session(self, session_id: UUID, **fields: Any) -> None:
        self._s.sessions[session_id] = replace(
            self._s.sessions[session_id], **copy.deepcopy(fields)
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
    ) -> list[SessionRecord]:
        rows = [
            s
            for s in self._s.sessions.values()
            if s.owner == owner
            and (kind is None or s.kind == kind)
            and (origin_session_id is None or s.origin_session_id == origin_session_id)
            and ((s.archived_at is not None) == archived)
            and (before is None or (s.updated_at, s.id) < before)
        ]
        rows.sort(key=lambda s: (s.updated_at, s.id), reverse=True)
        return [copy.deepcopy(s) for s in rows[:limit]]

    async def sessions_of_owner(self, owner: str) -> list[UUID]:
        return [s.id for s in self._s.sessions.values() if s.owner == owner]

    # inputs ---------------------------------------------------------------------------
    async def insert_inputs(self, recs: Sequence[InputRecord]) -> None:
        for r in recs:
            self._s.inputs[(r.session_id, r.name)] = copy.deepcopy(r)

    async def get_inputs(self, session_id: UUID) -> list[InputRecord]:
        return [copy.deepcopy(r) for (sid, _), r in self._s.inputs.items() if sid == session_id]

    # états ----------------------------------------------------------------------------
    async def insert_states(self, recs: Sequence[StateRecord]) -> None:
        for r in recs:
            self._s.states[r.id] = copy.deepcopy(r)

    async def get_states(self, ids: Sequence[UUID]) -> dict[UUID, StateRecord]:
        return {i: copy.deepcopy(self._s.states[i]) for i in ids if i in self._s.states}

    # entrées --------------------------------------------------------------------------
    async def insert_entries(self, recs: Sequence[EntryRecord]) -> None:
        for r in recs:
            self._s.entries[r.id] = copy.deepcopy(r)

    async def get_entries(self, session_id: UUID) -> list[EntryRecord]:
        rows = [copy.deepcopy(e) for e in self._s.entries.values() if e.session_id == session_id]
        rows.sort(key=lambda e: e.id)
        return rows

    async def get_entry(self, entry_id: UUID) -> EntryRecord | None:
        e = self._s.entries.get(entry_id)
        return copy.deepcopy(e) if e else None

    # tours ----------------------------------------------------------------------------
    async def insert_turn(self, rec: TurnRecord) -> None:
        self._s.turns[rec.id] = copy.deepcopy(rec)

    async def update_turn(self, turn_id: UUID, **fields: Any) -> None:
        self._s.turns[turn_id] = replace(self._s.turns[turn_id], **copy.deepcopy(fields))

    async def get_turn(self, turn_id: UUID) -> TurnRecord | None:
        t = self._s.turns.get(turn_id)
        return copy.deepcopy(t) if t else None

    # publications ---------------------------------------------------------------------
    async def insert_publication(self, rec: PublicationRecord) -> None:
        key = (rec.object_id, rec.version)
        if key in self._s.publications:
            raise ValueError("publication déjà existante")
        self._s.publications[key] = copy.deepcopy(rec)

    async def get_publications(self, object_id: UUID) -> list[PublicationRecord]:
        rows = [
            copy.deepcopy(p) for (oid, _), p in self._s.publications.items() if oid == object_id
        ]
        rows.sort(key=lambda p: p.version)
        return rows

    # résultats d'enfants --------------------------------------------------------------
    async def insert_child_result(self, rec: ChildResultRecord) -> None:
        self._s.child_results[rec.id] = copy.deepcopy(rec)

    async def pending_child_results(self, parent_session_id: UUID) -> list[ChildResultRecord]:
        rows = [
            copy.deepcopy(c)
            for c in self._s.child_results.values()
            if c.parent_session_id == parent_session_id and c.applied_at is None
        ]
        rows.sort(key=lambda c: c.id)
        return rows

    async def update_child_result(self, rec_id: UUID, **fields: Any) -> None:
        self._s.child_results[rec_id] = replace(self._s.child_results[rec_id], **fields)

    # événements -----------------------------------------------------------------------
    async def append_events(self, events: Sequence[NewEvent]) -> list[EventRecord]:
        out: list[EventRecord] = []
        self._s.events = list(self._s.events)
        for ev in events:
            self._s._event_seq += 1  # pyright: ignore[reportPrivateUsage]
            rec = EventRecord(
                id=self._s._event_seq,  # pyright: ignore[reportPrivateUsage]
                session_id=ev.session_id,
                turn_id=ev.turn_id,
                type=ev.type,
                data=copy.deepcopy(ev.data),
                created_at=ev.created_at,
            )
            self._s.events.append(rec)
            out.append(rec)
        return out

    async def read_events(
        self, session_id: UUID, *, after: int | None, turn_id: UUID | None = None, limit: int = 500
    ) -> list[EventRecord]:
        rows = [
            e
            for e in self._s.events
            if e.session_id == session_id
            and (after is None or e.id > after)
            and (turn_id is None or e.turn_id == turn_id)
        ]
        return [copy.deepcopy(e) for e in rows[:limit]]

    async def first_event_id(self, turn_id: UUID) -> int | None:
        for e in self._s.events:
            if e.turn_id == turn_id:
                return e.id
        return None

    async def purge_events(self, ended_before: datetime) -> int:
        ended = {
            t.id
            for t in self._s.turns.values()
            if t.ended_at is not None and t.ended_at < ended_before
        }
        before = len(self._s.events)
        self._s.events = [e for e in self._s.events if e.turn_id not in ended]
        return before - len(self._s.events)

    # purge ----------------------------------------------------------------------------
    async def delete_sessions(self, ids: Sequence[UUID]) -> None:
        doomed = set(ids)
        s = self._s
        s.sessions = {k: v for k, v in s.sessions.items() if k not in doomed}
        for k, v in list(s.sessions.items()):
            changes: dict[str, Any] = {}
            if v.cloned_from_session_id in doomed:
                changes["cloned_from_session_id"] = None
            if v.origin_session_id in doomed:
                changes["origin_session_id"] = None
            if changes:
                s.sessions[k] = replace(v, **changes)
        s.inputs = {k: v for k, v in s.inputs.items() if k[0] not in doomed}
        s.entries = {k: v for k, v in s.entries.items() if v.session_id not in doomed}
        doomed_turns = {k for k, v in s.turns.items() if v.session_id in doomed}
        s.turns = {k: v for k, v in s.turns.items() if k not in doomed_turns}
        s.events = [e for e in s.events if e.session_id not in doomed]
        s.publications = {k: v for k, v in s.publications.items() if v.session_id not in doomed}
        s.child_results = {
            k: v
            for k, v in s.child_results.items()
            if v.parent_session_id not in doomed and v.child_session_id not in doomed
        }
        used = {e.state_id for e in s.entries.values()} | {
            p.state_id for p in s.publications.values()
        }
        s.states = {k: v for k, v in s.states.items() if k in used}

    # notifications --------------------------------------------------------------------
    def notify_events(self, session_id: UUID) -> None:
        if session_id not in self.notified_sessions:
            self.notified_sessions.append(session_id)

    def notify_cancel(self, turn_id: UUID) -> None:
        self.notified_cancels.append(turn_id)

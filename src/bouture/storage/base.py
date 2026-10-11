"""Interface de stockage.

Tout passe par une transaction (``Storage.transaction()``) ; les notifications émises dans
une transaction ne partent qu'au commit. Deux implémentations : mémoire (tests) et Postgres.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID


@dataclass(slots=True)
class SessionRecord:
    id: UUID
    kind: str
    owner: str
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    model_override: str | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)
    leaf_entry_id: UUID | None = None
    object_id: UUID | None = None
    cloned_from_session_id: UUID | None = None
    cloned_from_entry_id: UUID | None = None
    origin_session_id: UUID | None = None
    origin_entry_id: UUID | None = None
    origin_handler: str | None = None
    origin_payload: Any = None
    pending: list[dict[str, Any]] = field(default_factory=list)
    running_turn_id: UUID | None = None
    lock_expires_at: datetime | None = None
    cancel_requested: bool = False
    archived_at: datetime | None = None


@dataclass(slots=True)
class InputRecord:
    session_id: UUID
    name: str
    mode: str  # "ref" | "data"
    kind: str
    object_id: UUID | None = None
    version: int | None = None
    data: Any = None
    data_version: int | None = None


@dataclass(slots=True)
class StateRecord:
    id: UUID
    kind: str
    state_version: int
    data: Any
    created_at: datetime


@dataclass(slots=True)
class EntryRecord:
    id: UUID
    session_id: UUID
    parent_id: UUID | None
    type: str
    content: dict[str, Any]
    state_id: UUID | None
    is_checkpoint: bool
    created_at: datetime
    interrupted: bool = False
    turn_id: UUID | None = None
    usage: dict[str, Any] | None = None
    label: str | None = None
    hidden_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    derived_from: UUID | None = None


@dataclass(slots=True)
class TurnRecord:
    id: UUID
    session_id: UUID
    trigger: str
    status: str
    started_at: datetime
    parent_entry_id: UUID | None = None
    usage: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    ended_at: datetime | None = None


@dataclass(slots=True)
class PublicationRecord:
    kind: str
    object_id: UUID
    version: int
    session_id: UUID
    entry_id: UUID
    state_id: UUID
    owner: str
    published_at: datetime


@dataclass(slots=True)
class ChildResultRecord:
    id: UUID
    parent_session_id: UUID
    child_session_id: UUID
    object_id: UUID
    version: int
    created_at: datetime
    applied_at: datetime | None = None
    applied_entry_id: UUID | None = None


@dataclass(slots=True)
class NewEvent:
    session_id: UUID
    turn_id: UUID
    type: str
    data: dict[str, Any]
    created_at: datetime


@dataclass(slots=True)
class EventRecord:
    id: int
    session_id: UUID
    turn_id: UUID
    type: str
    data: dict[str, Any]
    created_at: datetime


class StorageTx(Protocol):
    """Opérations disponibles dans une transaction."""

    @property
    def db(self) -> Any:
        """``AsyncSession`` SQLAlchemy liée à la transaction (Postgres) ou ``None`` (mémoire)."""
        ...

    # sessions
    async def get_session(
        self, session_id: UUID, *, for_update: bool = False
    ) -> SessionRecord | None: ...
    async def insert_session(self, rec: SessionRecord) -> None: ...
    async def update_session(self, session_id: UUID, **fields: Any) -> None: ...
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
        """``metadata`` filtre par inclusion JSON (``@>``) : chaque clé donnée doit correspondre."""
        ...

    async def sessions_of_owner(self, owner: str) -> list[UUID]: ...

    # inputs
    async def insert_inputs(self, recs: Sequence[InputRecord]) -> None: ...
    async def get_inputs(self, session_id: UUID) -> list[InputRecord]: ...

    # états
    async def insert_states(self, recs: Sequence[StateRecord]) -> None: ...
    async def get_states(self, ids: Sequence[UUID]) -> dict[UUID, StateRecord]: ...
    async def delete_unused_states(self, ids: Sequence[UUID]) -> None:
        """Supprime ceux de ces snapshots qu'aucune entrée ni publication ne référence."""
        ...

    # entrées
    async def insert_entries(self, recs: Sequence[EntryRecord]) -> None: ...
    async def get_entries(self, session_id: UUID) -> list[EntryRecord]: ...
    async def get_entry(self, entry_id: UUID) -> EntryRecord | None: ...
    async def update_entry(self, entry_id: UUID, **fields: Any) -> None: ...

    # tours
    async def insert_turn(self, rec: TurnRecord) -> None: ...
    async def update_turn(self, turn_id: UUID, **fields: Any) -> None: ...
    async def get_turn(self, turn_id: UUID) -> TurnRecord | None: ...

    # publications
    async def insert_publication(self, rec: PublicationRecord) -> None: ...
    async def get_publications(self, object_id: UUID) -> list[PublicationRecord]: ...

    # résultats d'enfants en attente
    async def insert_child_result(self, rec: ChildResultRecord) -> None: ...
    async def pending_child_results(self, parent_session_id: UUID) -> list[ChildResultRecord]: ...
    async def update_child_result(self, rec_id: UUID, **fields: Any) -> None: ...

    # événements
    async def append_events(self, events: Sequence[NewEvent]) -> list[EventRecord]: ...
    async def read_events(
        self, session_id: UUID, *, after: int | None, turn_id: UUID | None = None, limit: int = 500
    ) -> list[EventRecord]: ...
    async def first_event_id(self, turn_id: UUID) -> int | None: ...
    async def purge_events(self, ended_before: datetime) -> int: ...

    # purge physique
    async def delete_sessions(self, ids: Sequence[UUID]) -> None: ...

    # notifications (envoyées au commit)
    def notify_events(self, session_id: UUID) -> None: ...
    def notify_cancel(self, turn_id: UUID) -> None: ...


EventsCallback = Callable[[UUID], None]
CancelCallback = Callable[[UUID], None]


class Storage(Protocol):
    def transaction(self) -> AbstractAsyncContextManager[StorageTx]: ...
    async def setup(self, *, auto_migrate: bool = True) -> None: ...
    async def listen(
        self, on_events: EventsCallback, on_cancel: CancelCallback
    ) -> Callable[[], Awaitable[None]]:
        """Abonne le processus aux notifications ; renvoie une fonction de désabonnement."""
        ...

    async def close(self) -> None: ...

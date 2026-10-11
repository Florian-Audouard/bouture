"""Objets renvoyés par l'API publique (sérialisables tels quels par FastAPI)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from .messages import Message, Usage
from .storage.base import EntryRecord

SessionStatus = Literal["idle", "running", "awaiting_input"]
TurnReason = Literal[
    "completed", "awaiting_input", "interrupted", "error", "max_steps", "crashed", "server_shutdown"
]


class PendingQuestion(BaseModel):
    tool_call_id: str
    tool_name: str
    question: str
    choices: list[str] | None = None


class SessionInfo(BaseModel):
    id: UUID
    kind: str
    owner: str
    title: str | None
    metadata: dict[str, Any]
    model: str
    model_settings: dict[str, Any] = {}
    status: SessionStatus
    leaf_entry_id: UUID | None
    pending: list[PendingQuestion]
    object_id: UUID | None
    cloned_from_session_id: UUID | None
    cloned_from_entry_id: UUID | None
    origin_session_id: UUID | None
    origin_entry_id: UUID | None
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


class EntryView(BaseModel):
    id: UUID
    parent_id: UUID | None
    type: str
    content: dict[str, Any]
    state_id: UUID | None
    is_checkpoint: bool
    interrupted: bool
    turn_id: UUID | None
    usage: dict[str, Any] | None
    created_at: datetime
    label: str | None = None
    metadata: dict[str, Any] = {}
    hidden_at: datetime | None = None
    derived_from: UUID | None = None


def entry_view(rec: EntryRecord) -> EntryView:
    return EntryView(
        id=rec.id,
        parent_id=rec.parent_id,
        type=rec.type,
        content=rec.content,
        state_id=rec.state_id,
        is_checkpoint=rec.is_checkpoint,
        interrupted=rec.interrupted,
        turn_id=rec.turn_id,
        usage=rec.usage,
        created_at=rec.created_at,
        label=rec.label,
        metadata=rec.metadata,
        hidden_at=rec.hidden_at,
        derived_from=rec.derived_from,
    )


class InputView(BaseModel):
    """Input effectif d'une session : ``Ref`` (``id``, ``version``) ou ``Data`` (``value``)."""

    mode: Literal["ref", "data"]
    kind: str
    id: UUID | None = None
    version: int | None = None
    value: Any = None


class RequestPreview(BaseModel):
    """Ce que recevrait le modèle au prochain appel (D93).

    ``system``, ``messages`` et ``tools`` valent ``None`` quand le prompt n'est pas exposé.
    ``input_tokens`` / ``cost`` valent ``None`` si le client ne sait pas estimer.
    """

    model: str
    settings: dict[str, Any]
    input_tokens: int | None = None
    cost: float | None = None
    system: str | None = None
    messages: list[Message] | None = None
    tools: list[dict[str, Any]] | None = None


class StateView(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: Any
    state_version: int
    entry_id: UUID
    state_id: UUID | None


class Siblings(BaseModel):
    entries: list[EntryView]
    index: int


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None


class PublishedRefView(BaseModel):
    kind: str
    object_id: UUID
    version: int


class PublicationInfo(BaseModel):
    kind: str
    object_id: UUID
    version: int
    session_id: UUID
    entry_id: UUID
    state_id: UUID
    owner: str
    published_at: datetime


class Event(BaseModel):
    id: int
    session_id: UUID
    turn_id: UUID
    type: str
    data: dict[str, Any]
    ts: datetime


class TurnResult(BaseModel):
    turn_id: UUID
    session_id: UUID
    reason: TurnReason
    leaf_entry_id: UUID | None
    usage: Usage
    error: dict[str, Any] | None = None


class TurnHandle:
    """Renvoyé par les opérations qui lancent un tour."""

    def __init__(
        self,
        session_id: UUID,
        turn_id: UUID,
        events_factory: Callable[[], AsyncIterator[Event]],
    ) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self._events_factory = events_factory

    def events(self) -> AsyncIterator[Event]:
        """Événements de ce tour, depuis son début, jusqu'à ``turn_end`` inclus."""
        return self._events_factory()

    async def wait(self) -> Event:
        """Consomme le flux et renvoie l'événement ``turn_end``."""
        last: Event | None = None
        async for ev in self.events():
            last = ev
        assert last is not None and last.type == "turn_end"
        return last

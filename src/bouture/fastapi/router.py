"""Fabrique de routeur FastAPI (D65–D67). Aucune logique : chaque route appelle le cœur."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..agents import Agents
from ..errors import (
    BoutureError,
    InvalidSettings,
    InvalidTarget,
    NotAwaitingInput,
    NotFound,
    ResolveError,
    SessionBusy,
    StateValidationError,
    UnknownKind,
)
from ..refs import Data, Ref
from ..views import (
    EntryView,
    Event,
    InputView,
    Page,
    PublicationInfo,
    PublishedRefView,
    RequestPreview,
    SessionInfo,
    Siblings,
    TurnHandle,
)

CurrentUser = Callable[..., str] | Callable[..., Awaitable[str]]

STATUS: dict[type[BoutureError], int] = {
    NotFound: 404,
    UnknownKind: 404,
    SessionBusy: 409,
    InvalidTarget: 409,
    NotAwaitingInput: 409,
    StateValidationError: 422,
    ResolveError: 422,
    InvalidSettings: 422,
}


def error_response(e: BoutureError) -> JSONResponse:
    status = next((v for k, v in STATUS.items() if isinstance(e, k)), 500)
    return JSONResponse({"error": type(e).__name__, "message": str(e)}, status_code=status)


async def bouture_exception_handler(request: Request, exc: Exception) -> Response:
    """À enregistrer sur l'application : ``app.add_exception_handler(BoutureError, …)``."""
    assert isinstance(exc, BoutureError)
    return error_response(exc)


# -- corps de requêtes -------------------------------------------------------------------
class InputBody(BaseModel):
    kind: str
    id: UUID | None = None
    version: int | None = None
    data: Any = None

    def to_input(self) -> Ref | Data:
        return (
            Ref(self.kind, self.id, self.version)
            if self.id is not None
            else Data(self.kind, self.data)
        )


class CreateSessionBody(BaseModel):
    kind: str
    inputs: dict[str, InputBody] = {}
    title: str | None = None
    metadata: dict[str, Any] = {}
    model: str | None = None
    model_settings: dict[str, Any] = {}


class PatchSessionBody(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] | None = None
    model: str | None = None
    model_settings: dict[str, Any] | None = None


class CloneBody(BaseModel):
    title: str | None = None
    entry_id: UUID | None = None


class PreviewBody(BaseModel):
    content: str | list[dict[str, Any]] | None = None


class AppendBody(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]
    parent_entry_id: UUID | None = None


class EditEntryBody(BaseModel):
    content: str | list[dict[str, Any]]


class LabelBody(BaseModel):
    label: str | None


class EntryMetadataBody(BaseModel):
    metadata: dict[str, Any]


class InputsBody(BaseModel):
    inputs: dict[str, InputBody]


class HiddenResponse(BaseModel):
    entry_ids: list[UUID]


class MessageBody(BaseModel):
    content: str | list[dict[str, Any]]


class AnswerBody(BaseModel):
    answer: Any
    tool_call_id: str | None = None


class EntryBody(BaseModel):
    entry_id: UUID


class StateBody(BaseModel):
    state: dict[str, Any]
    amend: bool = False


class StateResponse(BaseModel):
    state: Any
    state_version: int
    entry_id: UUID
    state_id: UUID | None


# -- SSE ---------------------------------------------------------------------------------
def sse_line(ev: Event) -> str:
    payload = ev.model_dump(mode="json")
    return f"id: {ev.id}\nevent: {ev.type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _sse(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    # Fermer la connexion ne fait que détacher l'abonné : le tour continue (D26).
    async for ev in events:
        yield sse_line(ev)


def sse_response(
    events: AsyncIterator[Event], headers: dict[str, str] | None = None
) -> StreamingResponse:
    return StreamingResponse(
        _sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **(headers or {})},
    )


def turn_response(h: TurnHandle) -> StreamingResponse:
    return sse_response(h.events(), {"X-Turn-Id": str(h.turn_id)})


# -- routeur -----------------------------------------------------------------------------
def make_router(
    agents: Agents[Any], current_user: CurrentUser, *, expose_prompt: bool = True
) -> APIRouter:
    """Routeur prêt à monter. ``current_user`` est une dépendance FastAPI renvoyant l'owner.

    ``expose_prompt=False`` : ``/preview`` ne renvoie que modèle, réglages et estimation (D93).
    """
    r = APIRouter(responses={404: {"description": "Introuvable"}, 409: {"description": "Conflit"}})
    User = Depends(current_user)

    def handled[**P, T](fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T | JSONResponse]]:
        import functools

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | JSONResponse:
            try:
                return await fn(*args, **kwargs)
            except BoutureError as e:
                return error_response(e)

        return wrapper

    # sessions
    @r.post("/sessions", status_code=201, response_model=SessionInfo)
    @handled
    async def create_session(body: CreateSessionBody, owner: str = User) -> SessionInfo:
        return await agents.create_session(
            body.kind,
            owner,
            inputs={k: v.to_input() for k, v in body.inputs.items()},
            title=body.title,
            metadata=body.metadata,
            model=body.model,
            model_settings=body.model_settings,
        )

    @r.get("/sessions", response_model=Page[SessionInfo])
    @handled
    async def list_sessions(
        kind: str | None = None,
        origin_session_id: UUID | None = None,
        archived: bool = False,
        limit: int = 50,
        cursor: str | None = None,
        object_id: UUID | None = None,
        metadata: str | None = None,
        owner: str = User,
    ) -> Page[SessionInfo] | JSONResponse:
        """``metadata`` : objet JSON, sessions dont les métadonnées le contiennent (D102)."""
        filt: dict[str, Any] | None = None
        if metadata is not None:
            try:
                filt = json.loads(metadata)
            except ValueError:
                filt = None
            if not isinstance(filt, dict):
                return JSONResponse(
                    {"error": "InvalidQuery", "message": "metadata : objet JSON attendu"},
                    status_code=422,
                )
        return await agents.list_sessions(
            owner,
            kind=kind,
            origin_session_id=origin_session_id,
            archived=archived,
            limit=min(max(limit, 1), 200),
            cursor=cursor,
            object_id=object_id,
            metadata=filt,
        )

    @r.get("/sessions/{session_id}", response_model=SessionInfo)
    @handled
    async def get_session(session_id: UUID, owner: str = User) -> SessionInfo:
        return await agents.get_session(session_id, owner)

    @r.patch("/sessions/{session_id}", response_model=SessionInfo)
    @handled
    async def patch_session(
        session_id: UUID, body: PatchSessionBody, owner: str = User
    ) -> SessionInfo:
        info = await agents.get_session(session_id, owner)
        fields = body.model_fields_set
        if "title" in fields:
            info = await agents.rename(session_id, owner, body.title)
        if "metadata" in fields:
            info = await agents.update_metadata(session_id, owner, body.metadata or {})
        if "model" in fields:
            info = await agents.set_model(session_id, owner, body.model)
        if "model_settings" in fields:
            info = await agents.set_model_settings(session_id, owner, body.model_settings or {})
        return info

    @r.post("/sessions/{session_id}/preview", response_model=RequestPreview)
    @handled
    async def preview(
        session_id: UUID, body: PreviewBody | None = None, owner: str = User
    ) -> RequestPreview:
        return await agents.preview(
            session_id, owner, body.content if body else None, expose_prompt=expose_prompt
        )

    @r.post("/sessions/{session_id}/archive", response_model=SessionInfo)
    @handled
    async def archive(session_id: UUID, owner: str = User) -> SessionInfo:
        return await agents.archive(session_id, owner)

    @r.post("/sessions/{session_id}/unarchive", response_model=SessionInfo)
    @handled
    async def unarchive(session_id: UUID, owner: str = User) -> SessionInfo:
        return await agents.unarchive(session_id, owner)

    @r.post("/sessions/{session_id}/clone", status_code=201, response_model=SessionInfo)
    @handled
    async def clone(
        session_id: UUID, body: CloneBody | None = None, owner: str = User
    ) -> SessionInfo:
        return await agents.clone(
            session_id,
            owner,
            title=body.title if body else None,
            entry_id=body.entry_id if body else None,
        )

    # tours (réponses SSE)
    @r.post("/sessions/{session_id}/messages")
    @handled
    async def send_message(session_id: UUID, body: MessageBody, owner: str = User) -> Response:
        return turn_response(await agents.send_message(session_id, owner, body.content))

    @r.post("/sessions/{session_id}/answer")
    @handled
    async def answer(session_id: UUID, body: AnswerBody, owner: str = User) -> Response:
        return turn_response(
            await agents.answer(session_id, owner, body.answer, tool_call_id=body.tool_call_id)
        )

    @r.post("/sessions/{session_id}/entries/{entry_id}/fork")
    @handled
    async def fork(
        session_id: UUID, entry_id: UUID, body: MessageBody, owner: str = User
    ) -> Response:
        return turn_response(await agents.fork(session_id, owner, entry_id, body.content))

    @r.post("/sessions/{session_id}/entries/{entry_id}/regenerate")
    @handled
    async def regenerate(session_id: UUID, entry_id: UUID, owner: str = User) -> Response:
        return turn_response(await agents.regenerate(session_id, owner, entry_id))

    @r.post("/sessions/{session_id}/stop", status_code=204)
    @handled
    async def stop(session_id: UUID, owner: str = User) -> Response:
        await agents.stop(session_id, owner)
        return Response(status_code=204)

    @r.get("/sessions/{session_id}/events")
    @handled
    async def events(
        session_id: UUID,
        after: int | None = None,
        last_event_id: str | None = Header(default=None),
        owner: str = User,
    ) -> Response:
        cursor = after
        if last_event_id is not None and last_event_id.isdigit():
            cursor = int(last_event_id)
        await agents.get_session(session_id, owner)  # 404 avant d'ouvrir le flux
        return sse_response(agents.subscribe(session_id, owner, after=cursor))

    # arbre
    @r.get("/sessions/{session_id}/tree", response_model=list[EntryView])
    @handled
    async def tree(
        session_id: UUID, include_hidden: bool = False, labeled: bool = False, owner: str = User
    ) -> list[EntryView]:
        return await agents.get_tree(
            session_id, owner, include_hidden=include_hidden, labeled=labeled
        )

    @r.get("/sessions/{session_id}/branch", response_model=list[EntryView])
    @handled
    async def branch(
        session_id: UUID, entry_id: UUID | None = None, owner: str = User
    ) -> list[EntryView]:
        return await agents.get_branch(session_id, owner, entry_id=entry_id)

    @r.get("/sessions/{session_id}/entries/{entry_id}/siblings", response_model=Siblings)
    @handled
    async def siblings(session_id: UUID, entry_id: UUID, owner: str = User) -> Siblings:
        return await agents.siblings(session_id, owner, entry_id)

    @r.post("/sessions/{session_id}/switch", response_model=SessionInfo)
    @handled
    async def switch(session_id: UUID, body: EntryBody, owner: str = User) -> SessionInfo:
        return await agents.switch_branch(session_id, owner, body.entry_id)

    @r.post("/sessions/{session_id}/rollback", response_model=SessionInfo)
    @handled
    async def rollback(session_id: UUID, body: EntryBody, owner: str = User) -> SessionInfo:
        return await agents.rollback(session_id, owner, body.entry_id)

    @r.post("/sessions/{session_id}/entries/{entry_id}/hide", response_model=HiddenResponse)
    @handled
    async def hide(session_id: UUID, entry_id: UUID, owner: str = User) -> HiddenResponse:
        return HiddenResponse(entry_ids=await agents.hide_branch(session_id, owner, entry_id))

    @r.post("/sessions/{session_id}/entries/{entry_id}/restore", response_model=HiddenResponse)
    @handled
    async def restore(session_id: UUID, entry_id: UUID, owner: str = User) -> HiddenResponse:
        return HiddenResponse(entry_ids=await agents.restore_branch(session_id, owner, entry_id))

    # entrées hors tour
    @r.post("/sessions/{session_id}/entries", status_code=201, response_model=EntryView)
    @handled
    async def append(session_id: UUID, body: AppendBody, owner: str = User) -> EntryView:
        return await agents.append_message(
            session_id, owner, body.role, body.content, parent_entry_id=body.parent_entry_id
        )

    @r.patch("/sessions/{session_id}/entries/{entry_id}", response_model=EntryView)
    @handled
    async def edit_entry(
        session_id: UUID, entry_id: UUID, body: EditEntryBody, owner: str = User
    ) -> EntryView:
        return await agents.edit_entry(session_id, owner, entry_id, body.content)

    @r.put("/sessions/{session_id}/entries/{entry_id}/label", response_model=EntryView)
    @handled
    async def label(
        session_id: UUID, entry_id: UUID, body: LabelBody, owner: str = User
    ) -> EntryView:
        return await agents.label_entry(session_id, owner, entry_id, body.label)

    @r.patch("/sessions/{session_id}/entries/{entry_id}/metadata", response_model=EntryView)
    @handled
    async def entry_metadata(
        session_id: UUID, entry_id: UUID, body: EntryMetadataBody, owner: str = User
    ) -> EntryView:
        return await agents.set_entry_metadata(session_id, owner, entry_id, body.metadata)

    # inputs
    @r.get("/sessions/{session_id}/inputs", response_model=dict[str, InputView])
    @handled
    async def get_inputs(
        session_id: UUID, entry_id: UUID | None = None, owner: str = User
    ) -> dict[str, InputView]:
        return await agents.get_inputs(session_id, owner, entry_id=entry_id)

    @r.patch("/sessions/{session_id}/inputs", response_model=EntryView)
    @handled
    async def update_inputs(session_id: UUID, body: InputsBody, owner: str = User) -> EntryView:
        return await agents.update_inputs(
            session_id, owner, {k: v.to_input() for k, v in body.inputs.items()}
        )

    # état
    @r.get("/sessions/{session_id}/state", response_model=StateResponse)
    @handled
    async def get_state(
        session_id: UUID, entry_id: UUID | None = None, owner: str = User
    ) -> StateResponse:
        v = await agents.get_state(session_id, owner, entry_id=entry_id)
        state = v.state.model_dump(mode="json") if isinstance(v.state, BaseModel) else v.state
        return StateResponse(
            state=state, state_version=v.state_version, entry_id=v.entry_id, state_id=v.state_id
        )

    @r.put("/sessions/{session_id}/state", response_model=EntryView)
    @handled
    async def edit_state(session_id: UUID, body: StateBody, owner: str = User) -> EntryView:
        return await agents.edit_state(session_id, owner, body.state, amend=body.amend)

    # publication
    @r.post("/sessions/{session_id}/publish", response_model=PublishedRefView)
    @handled
    async def publish(session_id: UUID, owner: str = User) -> PublishedRefView:
        ref = await agents.publish(session_id, owner)
        return PublishedRefView(kind=ref.kind, object_id=ref.object_id, version=ref.version)

    @r.get("/publications/{kind}/{object_id}", response_model=list[PublicationInfo])
    @handled
    async def publications(kind: str, object_id: UUID, owner: str = User) -> list[PublicationInfo]:
        return await agents.list_publications(owner, kind, object_id)

    return r

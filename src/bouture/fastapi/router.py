"""Fabrique de routeur FastAPI (D65–D67). Aucune logique : chaque route appelle le cœur."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..agents import Agents
from ..errors import (
    BoutureError,
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
    Page,
    PublicationInfo,
    PublishedRefView,
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


class PatchSessionBody(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] | None = None
    model: str | None = None


class CloneBody(BaseModel):
    title: str | None = None


class MessageBody(BaseModel):
    content: str | list[dict[str, Any]]


class AnswerBody(BaseModel):
    answer: Any
    tool_call_id: str | None = None


class EntryBody(BaseModel):
    entry_id: UUID


class StateBody(BaseModel):
    state: dict[str, Any]


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
def make_router(agents: Agents[Any], current_user: CurrentUser) -> APIRouter:
    """Routeur prêt à monter. ``current_user`` est une dépendance FastAPI renvoyant l'owner."""
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
        )

    @r.get("/sessions", response_model=Page[SessionInfo])
    @handled
    async def list_sessions(
        kind: str | None = None,
        origin_session_id: UUID | None = None,
        archived: bool = False,
        limit: int = 50,
        cursor: str | None = None,
        owner: str = User,
    ) -> Page[SessionInfo]:
        return await agents.list_sessions(
            owner,
            kind=kind,
            origin_session_id=origin_session_id,
            archived=archived,
            limit=min(max(limit, 1), 200),
            cursor=cursor,
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
        return info

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
        return await agents.clone(session_id, owner, title=body.title if body else None)

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
    async def tree(session_id: UUID, owner: str = User) -> list[EntryView]:
        return await agents.get_tree(session_id, owner)

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
        return await agents.edit_state(session_id, owner, body.state)

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

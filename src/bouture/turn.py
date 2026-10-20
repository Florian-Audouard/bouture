"""Exécution d'un tour, application des résultats d'enfants et reconstruction après crash."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from pydantic import BaseModel, ValidationError

from . import jsonpatch
from .context import BuildContext, ChildResultContext, EntriesContext, ImageContext, ToolContext
from .definition import AgentDefinition, AskUser
from .errors import BoutureError, ModelError, StateValidationError, ToolFatal
from .events import EventSink, now
from .ids import uuid7
from .messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    content_text,
    images_of,
    message_images,
    normalize_content,
)
from .models.base import (
    CallContext,
    LoadedImage,
    ModelEvent,
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    UsageReport,
)
from .refs import Data, PublishedRef, Ref
from .state import StateCodec
from .storage.base import EntryRecord, NewEvent, SessionRecord, StateRecord, StorageTx
from .tree import Tree
from .views import EntryView, TurnResult, entry_view

if TYPE_CHECKING:
    from .agents import Agents

log = logging.getLogger("bouture")

_IMAGE_ERROR = "Erreur : image(s) renvoyée(s) par l'outil impossible(s) à charger."
CANCELLED_QUESTION = "Question annulée : l'utilisateur a poursuivi sans y répondre."
INTERRUPTED_CALL = "Interrompu : l'outil n'a pas été exécuté jusqu'au bout."


class Builder:
    """Accumule les nouvelles entrées d'un tour ; un snapshot n'est créé que si l'état change."""

    def __init__(
        self,
        session_id: UUID,
        turn_id: UUID | None,
        kind: str,
        state_version: int,
        parent_id: UUID | None,
        state_id: UUID | None,
        state_dump: dict[str, Any] | None,
    ) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.kind = kind
        self.state_version = state_version
        self.leaf_id = parent_id
        self.state_id = state_id
        self.state_dump = state_dump
        self._snap_dump = state_dump
        self.entries: list[EntryRecord] = []
        self.states: list[StateRecord] = []

    def set_state(self, dump: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Enregistre le nouvel état ; renvoie le JSON Patch depuis l'état précédent."""
        patch = jsonpatch.diff(self.state_dump, dump)
        self.state_dump = dump
        return patch

    def _snapshot(self) -> None:
        if self.state_dump != self._snap_dump:
            sid = uuid7()
            self.states.append(
                StateRecord(sid, self.kind, self.state_version, self.state_dump, now())
            )
            self.state_id = sid
            self._snap_dump = self.state_dump

    def add(
        self,
        type: str,
        content: dict[str, Any],
        *,
        entry_id: UUID | None = None,
        interrupted: bool = False,
        checkpoint: bool = False,
        usage: dict[str, Any] | None = None,
    ) -> EntryRecord:
        self._snapshot()
        e = EntryRecord(
            id=entry_id or uuid7(),
            session_id=self.session_id,
            parent_id=self.leaf_id,
            type=type,
            content=content,
            state_id=self.state_id,
            is_checkpoint=checkpoint or interrupted,
            created_at=now(),
            interrupted=interrupted,
            turn_id=self.turn_id,
            usage=usage,
        )
        self.entries.append(e)
        self.leaf_id = e.id
        return e

    def finish(self, checkpoint: bool) -> None:
        if self.entries and self.state_dump != self._snap_dump:
            # Changement d'état sans entrée qui le suive (outil qui pose une question) :
            # rattaché à la dernière entrée du tour.
            self._snapshot()
            self.entries[-1].state_id = self.state_id
        if checkpoint and self.entries:
            self.entries[-1].is_checkpoint = True

    async def write(self, tx: StorageTx) -> None:
        if self.states:
            await tx.insert_states(self.states)
        if self.entries:
            await tx.insert_entries(self.entries)


def to_text(result: Any) -> str:
    if result is None:
        return "ok"
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result, ensure_ascii=False, default=str)


ToolContent = str | list[TextBlock | ImageBlock]


def to_result(result: Any) -> ToolContent:
    """Valeur renvoyée par un outil → contenu du ``tool_result`` (D112).

    Un bloc ou une liste de blocs ``TextBlock`` / ``ImageBlock`` est gardé tel quel ; tout le
    reste passe par ``to_text`` (une liste de dicts reste du JSON).
    """
    if isinstance(result, TextBlock | ImageBlock):
        return [result]
    if isinstance(result, list | tuple) and result:
        items = cast("Sequence[Any]", result)
        if all(isinstance(b, TextBlock | ImageBlock) for b in items):
            return list(cast("Sequence[TextBlock | ImageBlock]", items))
    return to_text(result)


def tool_result_content(
    call_id: str, name: str, content: ToolContent, is_error: bool
) -> dict[str, Any]:
    return ToolResultMessage(
        tool_call_id=call_id, name=name, content=content, is_error=is_error
    ).model_dump(mode="json")


def _dump_blocks(blocks: Sequence[TextBlock | ImageBlock]) -> list[dict[str, Any]]:
    return [b.model_dump(mode="json") for b in blocks]


def _without_images(m: Message, missing: Mapping[str, str]) -> Message:
    """Remplace, dans la copie envoyée au modèle, les images qui n'ont pas pu être chargées."""

    def swap(blocks: list[TextBlock | ImageBlock]) -> list[TextBlock | ImageBlock]:
        return [
            TextBlock(text=f"[image indisponible : {b.ref}]")
            if isinstance(b, ImageBlock) and b.ref in missing
            else b
            for b in blocks
        ]

    if isinstance(m, UserMessage):
        return m.model_copy(update={"content": swap(m.content)})
    if isinstance(m, ToolResultMessage) and not isinstance(m.content, str):
        return m.model_copy(update={"content": swap(m.content)})
    return m


def revalidate(codec: StateCodec, state: Any) -> BaseModel:
    """Revalide la copie de travail (Pydantic ne valide pas les affectations par défaut)."""
    try:
        raw = state.model_dump(mode="json") if isinstance(state, BaseModel) else state
    except Exception as e:
        raise StateValidationError(str(e)) from e
    return codec.validate(raw)


async def build_request(
    agents: Agents[Any],
    definition: AgentDefinition[Any, Any],
    *,
    state: Any,
    owner: str,
    session_id: UUID,
    turn_id: UUID | None,
    model: str,
    model_settings: Mapping[str, Any],
    inputs: Mapping[str, Ref | Data],
    resolved: Mapping[str, str],
    branch: Sequence[EntryView],
) -> ModelRequest:
    """Requête LLM d'une étape : ``build_context`` du projet ou comportement par défaut."""
    bctx = BuildContext[Any, Any](
        definition=definition,
        state=state,
        deps=agents.deps,
        owner=owner,
        session_id=session_id,
        model=model,
        inputs=inputs,
        resolved_inputs=resolved,
        branch=branch,
        tools=[t.spec() for t in definition.tools.values()],
        model_settings=model_settings,
    )
    if definition.context_builder is not None:
        request = await definition.context_builder(bctx)
    else:
        request = bctx.default_request()
    if request.context is None:
        request.context = CallContext(owner, session_id, definition.kind, turn_id)
    return request


async def _aclose(it: AsyncIterator[ModelEvent]) -> None:
    aclose = getattr(it, "aclose", None)
    if aclose is not None:
        await aclose()


class TurnExecutor:
    def __init__(
        self,
        agents: Agents[Any],
        session_id: UUID,
        turn_id: UUID,
        trigger: str,
        owner: str,
        *,
        content: Any = None,
        answer: Any = None,
        tool_call_id: str | None = None,
    ) -> None:
        self.agents = agents
        self.session_id = session_id
        self.turn_id = turn_id
        self.trigger = trigger
        self.owner = owner
        self.content = content
        self.answer = answer
        self.tool_call_id = tool_call_id
        self.kind = ""
        self.attach_id: UUID | None = None
        self.builder: Builder | None = None
        self.pending: list[dict[str, Any]] = []
        self.state: Any = None
        self.usage = Usage()
        self._task: asyncio.Task[Any] | None = None
        self._cancel_reason: str | None = None
        self._committing = False
        self._open_msg: dict[str, Any] | None = None
        self._open_calls: list[ToolCallBlock] = []
        self._tool_before: dict[str, Any] | None = None
        self._images: dict[str, LoadedImage] = {}
        self._image_errors: dict[str, str] = {}

    # -- annulation ------------------------------------------------------------------
    def cancel(self, reason: str) -> None:
        """``interrupted`` (stop), ``server_shutdown`` ou ``abandon`` (bail perdu : rien n'est écrit)."""
        if self._committing or self._cancel_reason is not None:
            return
        self._cancel_reason = reason
        if self._task is not None:
            self._task.cancel()

    # -- point d'entrée --------------------------------------------------------------
    async def run(self) -> None:
        a = self.agents
        self._task = asyncio.current_task()
        a._executors[self.turn_id] = self  # pyright: ignore[reportPrivateUsage]
        self.sink = EventSink(
            a.storage,
            a.hub,
            self.session_id,
            self.turn_id,
            flush_interval=a.flush_interval,
            flush_chars=a.flush_chars,
        )
        reason, error = "completed", None
        hb: asyncio.Task[None] | None = None
        try:
            try:
                if self._cancel_reason is not None:
                    raise asyncio.CancelledError
                await self._load()
                self.sink.start()
                hb = asyncio.create_task(self._heartbeat())
                await self.sink.emit(
                    "turn_start",
                    {"trigger": self.trigger, "parent_entry_id": str(self.attach_id)},
                )
                reason = await self._execute()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
                if self._cancel_reason is None:
                    self._cancel_reason = "server_shutdown"
                if self._cancel_reason == "abandon":
                    return
                reason = self._cancel_reason
                await self._finalize_abnormal()
            except Exception as e:
                reason, error = "error", self._error_info(e)
                if not isinstance(e, BoutureError | ModelError):
                    log.exception("tour %s : erreur", self.turn_id)
                with contextlib.suppress(Exception):
                    await self._finalize_abnormal()
            finally:
                if hb is not None:
                    hb.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await hb
            self._committing = True
            with contextlib.suppress(Exception):
                await self.sink.close()
            result = await self._commit(reason, error)
        finally:
            a._executors.pop(self.turn_id, None)  # pyright: ignore[reportPrivateUsage]
            if self._cancel_reason == "abandon":
                with contextlib.suppress(Exception):
                    await self.sink.close()
        if result is not None:
            await a._after_turn(result, self.owner, self.kind)  # pyright: ignore[reportPrivateUsage]

    @staticmethod
    def _error_info(e: Exception) -> dict[str, Any]:
        if isinstance(e, ModelError):
            return {"code": e.code, "message": str(e)}
        if isinstance(e, ToolFatal):
            return {"code": "tool_fatal", "message": str(e)}
        if isinstance(e, BoutureError):
            return {"code": type(e).__name__, "message": str(e)}
        return {"code": "internal_error", "message": "Erreur interne."}

    # -- chargement ------------------------------------------------------------------
    async def _load(self) -> None:
        a = self.agents
        async with a.storage.transaction() as tx:
            s = await tx.get_session(self.session_id)
            turn = await tx.get_turn(self.turn_id)
            assert s is not None and turn is not None
            entries = await tx.get_entries(self.session_id)
            input_recs = await tx.get_inputs(self.session_id)
            tree = Tree(entries)
            assert turn.parent_entry_id is not None
            self.base_path = tree.path(turn.parent_entry_id)
            attach = self.base_path[-1]
            states = await tx.get_states([attach.state_id]) if attach.state_id else {}
        self.session: SessionRecord = s
        self.kind = s.kind
        self.attach_id = attach.id
        self.definition: AgentDefinition[Any, Any] = a.definition(s.kind)
        self.codec = a.codec(s.kind)
        self.model = s.model_override or self.definition.model
        self.model_settings = dict(s.model_settings)
        if self.trigger in ("message", "answer"):
            self.pending = list(s.pending)
        rec = states.get(attach.state_id) if attach.state_id else None
        self.state = self.codec.load(rec.data, rec.state_version) if rec else None
        if self.state is None and self.codec.state_type is not None:
            self.state = self.definition.make_initial_state({})
        self.builder = Builder(
            self.session_id,
            self.turn_id,
            s.kind,
            self.codec.version,
            attach.id,
            attach.state_id,
            StateCodec.dump(self.state),
        )
        self.inputs = a._load_inputs(a._effective_input_records(input_recs, self.base_path))  # pyright: ignore[reportPrivateUsage]
        self.resolved = await a._resolve_inputs(self.inputs, self.owner, self.session_id)  # pyright: ignore[reportPrivateUsage]

    # -- déroulé ---------------------------------------------------------------------
    async def _execute(self) -> str:
        b = self.builder
        assert b is not None
        if self.trigger in ("message", "fork"):
            if self.trigger == "message":
                await self._cancel_pending()
            msg = UserMessage(content=normalize_content(self.content))
            dump = msg.model_dump(mode="json")
            e = b.add("user", dump)
            await self.sink.emit(
                "user_message", {"entry_id": str(e.id), "content": dump["content"]}
            )
        elif self.trigger == "answer":
            await self._handle_answer()
            if self.pending:
                await self._ask_next()
                return "awaiting_input"
        elif self.trigger == "child_result":
            return "completed"
        return await self._loop()

    async def _cancel_pending(self) -> None:
        while self.pending:
            q = self.pending[0]
            await self._record_result(q["tool_call_id"], q["tool_name"], CANCELLED_QUESTION, False)

    async def _ask_next(self) -> None:
        q = self.pending[0]
        await self.sink.emit(
            "ask_user",
            {
                "tool_call_id": q["tool_call_id"],
                "tool_name": q["tool_name"],
                "question": q["question"],
                "choices": q.get("choices"),
            },
        )

    async def _handle_answer(self) -> None:
        q = next(
            (p for p in self.pending if p["tool_call_id"] == self.tool_call_id), self.pending[0]
        )
        tool = self.definition.tools.get(q["tool_name"])
        before = StateCodec.dump(self.state)
        self._tool_before = before
        assert self.attach_id is not None
        ctx = self._tool_ctx(self.attach_id)
        is_error = False
        try:
            if tool is not None and tool.answer_fn is not None:
                result = await tool.answer_fn(ctx, self.answer, q.get("payload"))
                text, is_error = await self._tool_content(result)
                if is_error:
                    self._restore(before)
                else:
                    self.state = ctx.state
            else:
                text = f"Réponse de l'utilisateur : {to_text(self.answer)}"
        except (asyncio.CancelledError, ToolFatal):
            raise
        except Exception as e:
            self._restore(before)
            text, is_error = f"Erreur : {type(e).__name__}: {e}", True
        await self._record_result(q["tool_call_id"], q["tool_name"], text, is_error)

    async def _loop(self) -> str:
        for _ in range(self.definition.max_steps):
            request = await self._build_request()
            await self._attach_images(request)
            msg, entry_id = await self._call_model(request)
            if not msg.tool_calls:
                return "completed"
            self._open_calls = list(msg.tool_calls)
            while self._open_calls:
                await self._run_tool(self._open_calls[0], entry_id)
            if self.pending:
                await self._ask_next()
                return "awaiting_input"
        return "max_steps"

    async def _build_request(self) -> ModelRequest:
        assert self.builder is not None
        d = self.definition
        branch = [entry_view(e) for e in self.base_path] + [
            entry_view(e) for e in self.builder.entries
        ]
        return await build_request(
            self.agents,
            d,
            state=self.state,
            owner=self.owner,
            session_id=self.session_id,
            turn_id=self.turn_id,
            model=self.model,
            model_settings=self.model_settings,
            inputs=self.inputs,
            resolved=self.resolved,
            branch=branch,
        )

    # -- images (D113) ---------------------------------------------------------------
    async def _load_images(self, blocks: Sequence[ImageBlock]) -> dict[str, str]:
        """Charge (une fois par tour) les images via l'``image_loader`` ; renvoie les échecs."""
        loader = self.agents._image_loader  # pyright: ignore[reportPrivateUsage]
        failed: dict[str, str] = {}
        if loader is None:
            return failed
        ctx = ImageContext[Any](self.agents.deps, self.owner, self.session_id, self.kind)
        for b in blocks:
            ref = b.ref
            if ref in self._images or ref in failed:
                continue
            if ref in self._image_errors:
                failed[ref] = self._image_errors[ref]
                continue
            try:
                data, media_type = await loader(ref, ctx)
                # Le loader vient de l'hôte : son type déclaré n'est pas une garantie.
                if not isinstance(cast(object, data), bytes | bytearray) or not media_type:
                    raise TypeError("l'image_loader doit renvoyer (bytes, media_type)")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failed[ref] = f"{type(e).__name__}: {e}"
                log.warning("image %r indisponible : %s", ref, failed[ref])
                continue
            self._images[ref] = LoadedImage(bytes(data), media_type)
        self._image_errors.update(failed)
        return failed

    async def _attach_images(self, request: ModelRequest) -> None:
        """Joint à la requête les octets des images de l'historique ; les absentes → texte."""
        blocks = [b for b in message_images(request.messages) if b.ref not in request.images]
        if not blocks:
            return
        failed = await self._load_images(blocks)
        for b in blocks:
            if b.ref in self._images:
                request.images[b.ref] = self._images[b.ref]
        if failed:
            request.messages = [_without_images(m, failed) for m in request.messages]

    async def _tool_content(self, result: Any) -> tuple[ToolContent, bool]:
        """Contenu du résultat et ``is_error`` : une image impossible à charger est une erreur."""
        content = to_result(result)
        if isinstance(content, str):
            return content, False
        failed = await self._load_images(images_of(content))
        if failed:
            lines = [f"- {ref} : {err}" for ref, err in failed.items()]
            return f"{_IMAGE_ERROR}\n" + "\n".join(lines), True
        return content, False

    # -- appel au modèle -------------------------------------------------------------
    def _append(self, kind: str, text: str, signature: str | None = None) -> None:
        assert self._open_msg is not None
        blocks: list[TextBlock | ReasoningBlock] = self._open_msg["blocks"]
        last = blocks[-1] if blocks else None
        if kind == "text":
            if isinstance(last, TextBlock):
                last.text += text
            else:
                blocks.append(TextBlock(text=text))
        else:
            if isinstance(last, ReasoningBlock):
                last.text += text
                last.signature = signature or last.signature
            else:
                blocks.append(ReasoningBlock(text=text, signature=signature))

    async def _call_model(self, request: ModelRequest) -> tuple[AssistantMessage, UUID]:
        a = self.agents
        entry_id = uuid7()
        self._open_msg = {"id": entry_id, "blocks": [], "calls": []}
        await self.sink.emit("message_start", {"entry_id": str(entry_id), "role": "assistant"})
        usage = Usage(model=request.model)
        stream_args = self.definition.stream_tool_args
        attempt = 0
        while True:
            got_token = False
            partial: dict[int, list[str | None]] = {}
            stream = a.model_client.stream(request)
            try:
                async for ev in stream:
                    if isinstance(ev, TextDelta):
                        got_token = True
                        self._append("text", ev.text)
                        await self.sink.delta("text_delta", entry_id, ev.text)
                    elif isinstance(ev, ReasoningDelta):
                        got_token = True
                        self._append("reasoning", ev.text, ev.signature)
                        await self.sink.delta("reasoning_delta", entry_id, ev.text)
                    elif isinstance(ev, ToolCallDelta):
                        got_token = True
                        if stream_args:
                            known = partial.setdefault(ev.index, [None, None])
                            known[0] = known[0] or ev.id
                            known[1] = known[1] or ev.name
                            await self.sink.tool_delta(
                                entry_id, ev.index, known[0], known[1], ev.args_delta
                            )
                    elif isinstance(ev, ToolCallComplete):
                        got_token = True
                        call = ToolCallBlock(id=ev.id, name=ev.name, args=ev.args)
                        self._open_msg["calls"].append(call)
                        await self.sink.emit(
                            "tool_call",
                            {
                                "entry_id": str(entry_id),
                                "tool_call_id": ev.id,
                                "name": ev.name,
                                "args": ev.args,
                            },
                        )
                    elif isinstance(ev, UsageReport):
                        usage = usage + ev.usage
                break
            except ModelError as e:
                if e.transient and not got_token and attempt < a.retries:
                    attempt += 1
                    await asyncio.sleep(a.retry_base_delay * 2 ** (attempt - 1))
                    continue
                raise
            finally:
                await _aclose(stream)
        assert self.builder is not None
        msg = AssistantMessage(
            content=[*self._open_msg["blocks"], *self._open_msg["calls"]], usage=usage
        )
        udump = usage.model_dump(mode="json")
        self.builder.add("assistant", msg.model_dump(mode="json"), entry_id=entry_id, usage=udump)
        self._open_msg = None
        self.usage = self.usage + usage
        await self.sink.emit(
            "message_end", {"entry_id": str(entry_id), "interrupted": False, "usage": udump}
        )
        return msg, entry_id

    # -- outils ----------------------------------------------------------------------
    def _tool_ctx(self, entry_id: UUID) -> ToolContext[Any, Any]:
        async def spawn(kind: str, **kw: Any) -> Any:
            return await self.agents._spawn_child(  # pyright: ignore[reportPrivateUsage]
                self.session, self.owner, entry_id, self.sink, kind, **kw
            )

        async def emit(name: str, data: dict[str, Any]) -> None:
            await self.sink.emit("custom", {"name": name, "data": data})

        return ToolContext[Any, Any](
            state=self.state,
            deps=self.agents.deps,
            owner=self.owner,
            session_id=self.session_id,
            turn_id=self.turn_id,
            entry_id=entry_id,
            inputs=self.inputs,
            _spawn=spawn,
            _emit=emit,
        )

    def _restore(self, before: dict[str, Any] | None) -> None:
        if before is not None:
            self.state = self.codec.validate(before)

    async def _run_tool(self, call: ToolCallBlock, entry_id: UUID) -> None:
        d = self.definition
        tool = d.tools.get(call.name)
        if tool is None:
            await self._record_result(
                call.id, call.name, f"Erreur : outil inconnu « {call.name} ».", True
            )
            return
        try:
            params = tool.params_model.model_validate(call.args)
        except ValidationError as e:
            await self._record_result(
                call.id, call.name, f"Erreur : arguments invalides.\n{e}", True
            )
            return
        kwargs = {k: getattr(params, k) for k in type(params).model_fields}
        before = StateCodec.dump(self.state)
        self._tool_before = before
        ctx = self._tool_ctx(entry_id)
        timeout = tool.timeout or d.tool_timeout
        try:
            result = await asyncio.wait_for(tool.fn(ctx, **kwargs), timeout)
        except (asyncio.CancelledError, ToolFatal):
            raise
        except TimeoutError:
            self._restore(before)
            await self._record_result(
                call.id, call.name, f"Erreur : délai dépassé ({timeout} s).", True
            )
            return
        except Exception as e:
            self._restore(before)
            await self._record_result(call.id, call.name, f"Erreur : {type(e).__name__}: {e}", True)
            return
        if isinstance(result, AskUser):
            self.state = ctx.state
            patch = self._check_state(before)
            if isinstance(patch, str):
                # État invalide : la question n'est pas posée, l'erreur revient au LLM.
                await self._record_result(call.id, call.name, patch, True)
                return
            self.pending.append(
                {
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "question": result.question,
                    "choices": result.choices,
                    "payload": result.payload,
                }
            )
            self._open_calls = [c for c in self._open_calls if c.id != call.id]
            self._tool_before = None
            if patch:
                await self._emit_state(patch)
            return
        content, is_error = await self._tool_content(result)
        if is_error:
            self._restore(before)
            await self._record_result(call.id, call.name, content, True)
            return
        self.state = ctx.state
        await self._record_result(call.id, call.name, content, False)

    def _check_state(self, before: dict[str, Any] | None) -> list[dict[str, Any]] | str:
        """Revalide l'état ; renvoie le patch, ou un message d'erreur si l'état a été annulé."""
        assert self.builder is not None
        if self.codec.state_type is None:
            return []
        try:
            self.state = revalidate(self.codec, self.state)
        except StateValidationError as e:
            self._restore(before)
            return f"Erreur : l'état produit est invalide, modifications annulées.\n{e}"
        return self.builder.set_state(StateCodec.dump(self.state))

    async def _emit_state(self, patch: list[dict[str, Any]]) -> None:
        await self.sink.emit("state_changed", {"patch": patch, "state_version": self.codec.version})

    async def _record_result(
        self, call_id: str, name: str, content: ToolContent, is_error: bool
    ) -> None:
        """Valide l'état, ajoute l'entrée ``tool_result`` (sans ``await`` entre les deux)."""
        assert self.builder is not None
        checked = self._check_state(self._tool_before)
        if isinstance(checked, str):
            content, is_error, checked = checked, True, []
        e = self.builder.add("tool_result", tool_result_content(call_id, name, content, is_error))
        self._open_calls = [c for c in self._open_calls if c.id != call_id]
        self.pending = [p for p in self.pending if p["tool_call_id"] != call_id]
        self._tool_before = None
        if checked:
            await self._emit_state(checked)
        await self.sink.emit(
            "tool_result",
            {
                "entry_id": str(e.id),
                "tool_call_id": call_id,
                "name": name,
                "is_error": is_error,
                "content": content if isinstance(content, str) else content_text(content),
                **({} if isinstance(content, str) else {"blocks": _dump_blocks(content)}),
            },
        )

    # -- fin anormale ----------------------------------------------------------------
    async def _finalize_abnormal(self) -> None:
        b = self.builder
        if b is None:
            return
        if self._open_msg is not None:
            m, self._open_msg = self._open_msg, None
            if m["blocks"]:
                msg = AssistantMessage(content=m["blocks"], interrupted=True)
                b.add("assistant", msg.model_dump(mode="json"), entry_id=m["id"], interrupted=True)
            await self.sink.emit(
                "message_end", {"entry_id": str(m["id"]), "interrupted": True, "usage": None}
            )
        if self._tool_before is not None:
            self._restore(self._tool_before)
        for call in list(self._open_calls):
            await self._record_result(call.id, call.name, INTERRUPTED_CALL, True)
        for q in list(self.pending):
            await self._record_result(q["tool_call_id"], q["tool_name"], INTERRUPTED_CALL, True)

    async def _heartbeat(self) -> None:
        a = self.agents
        while True:
            await asyncio.sleep(a.lock_ttl / 3)
            action = None
            try:
                async with a.storage.transaction() as tx:
                    s = await tx.get_session(self.session_id, for_update=True)
                    if s is None or s.running_turn_id != self.turn_id:
                        action = "abandon"
                    elif s.cancel_requested:
                        action = "interrupted"
                    else:
                        await tx.update_session(
                            self.session_id,
                            lock_expires_at=a._lease_expiry(),  # pyright: ignore[reportPrivateUsage]
                        )
            except Exception:
                log.exception("tour %s : renouvellement du bail impossible", self.turn_id)
                continue
            if action is not None:
                self.cancel(action)
                return

    # -- commit ----------------------------------------------------------------------
    async def _commit(self, reason: str, error: dict[str, Any] | None) -> TurnResult | None:
        a = self.agents
        b = self.builder
        async with a.storage.transaction() as tx:
            s = await tx.get_session(self.session_id, for_update=True)
            if s is None or s.running_turn_id != self.turn_id:
                return None  # bail repris par un autre processus : le tour a été clos « crashed »
            events: list[NewEvent] = []
            leaf = s.leaf_entry_id
            if b is not None:
                b.finish(checkpoint=reason != "awaiting_input")
                if reason != "awaiting_input":
                    self.state, evs = await apply_child_results(
                        a, tx, s, self.owner, b, self.state, self.sink
                    )
                    events += evs
                if meta := await self._entry_metadata(tx, b):
                    events.append(self.sink.new("entry_metadata", {"entries": meta}))
                await b.write(tx)
                leaf = b.leaf_id
            await tx.update_session(
                self.session_id,
                leaf_entry_id=leaf,
                pending=self.pending if reason == "awaiting_input" else [],
                running_turn_id=None,
                lock_expires_at=None,
                cancel_requested=False,
                updated_at=now(),
            )
            usage = self.usage.model_dump(mode="json")
            await tx.update_turn(
                self.turn_id, status=reason, usage=usage, error=error, ended_at=now()
            )
            events.append(
                self.sink.new(
                    "turn_end",
                    {
                        "reason": reason,
                        "leaf_entry_id": str(leaf) if leaf else None,
                        "usage": usage,
                        "error": error,
                    },
                )
            )
            await tx.append_events(events)
            tx.notify_events(self.session_id)
        a.hub.wake(self.session_id)
        return TurnResult(
            turn_id=self.turn_id,
            session_id=self.session_id,
            reason=reason,  # type: ignore[arg-type]
            leaf_entry_id=leaf,
            usage=self.usage,
            error=error,
        )

    async def _entry_metadata(self, tx: StorageTx, b: Builder) -> dict[str, dict[str, Any]]:
        """Appelle ``on_entries`` sur les entrées du tour ; renvoie les métadonnées posées (D101)."""
        hook = self.definition.entries_hook
        if hook is None or not b.entries:
            return {}
        ctx = EntriesContext[Any](
            db=tx.db,
            deps=self.agents.deps,
            owner=self.owner,
            session_id=self.session_id,
            turn_id=self.turn_id,
        )
        try:
            result = await hook(ctx, [entry_view(e) for e in b.entries])
        except Exception:
            log.exception("tour %s : on_entries a échoué", self.turn_id)
            return {}
        by_id = {e.id: e for e in b.entries}
        out: dict[str, dict[str, Any]] = {}
        for entry_id, data in result.items():
            e = by_id.get(entry_id)
            if e is not None and data:
                e.metadata = {**e.metadata, **data}
                out[str(e.id)] = e.metadata
        return out


async def apply_child_results(
    agents: Agents[Any],
    tx: StorageTx,
    session: SessionRecord,
    owner: str,
    b: Builder,
    state: Any,
    sink: EventSink,
) -> tuple[Any, list[NewEvent]]:
    """Applique les résultats d'enfants en file sur la feuille du parent (D58, D59)."""
    events: list[NewEvent] = []
    queued = await tx.pending_child_results(session.id)
    if not queued:
        return state, events
    d = agents.definition(session.kind)
    codec = agents.codec(session.kind)
    for r in queued:
        child = await tx.get_session(r.child_session_id)
        pub = next(
            (p for p in await tx.get_publications(r.object_id) if p.version == r.version), None
        )
        error: str | None = None
        before = StateCodec.dump(state)
        if child is None or pub is None:
            error = "publication introuvable"
        else:
            handler = d.child_handlers.get(child.origin_handler) if child.origin_handler else None
            if handler is not None:
                try:
                    child_codec = agents.codec(child.kind)
                    srec = (await tx.get_states([pub.state_id]))[pub.state_id]
                    child_state = child_codec.load(srec.data, srec.state_version)
                    ctx = ChildResultContext[Any, Any](
                        state=state,
                        deps=agents.deps,
                        owner=owner,
                        session_id=session.id,
                        child_session_id=child.id,
                    )
                    await handler(
                        ctx,
                        PublishedRef(child.kind, r.object_id, r.version),
                        child_state,
                        child.origin_payload,
                    )
                    state = ctx.state
                    if codec.state_type is not None:
                        state = revalidate(codec, state)
                except Exception as e:
                    log.exception("résultat d'enfant %s : le handler a échoué", r.id)
                    state = codec.validate(before) if before is not None else state
                    error = f"{type(e).__name__}: {e}"
        patch = b.set_state(StateCodec.dump(state))
        e = b.add(
            "child_result",
            {
                "child_session_id": str(r.child_session_id),
                "kind": child.kind if child else None,
                "object_id": str(r.object_id),
                "version": r.version,
                "handler": child.origin_handler if child else None,
                "error": error,
            },
            checkpoint=True,
        )
        if patch:
            events.append(
                sink.new("state_changed", {"patch": patch, "state_version": codec.version})
            )
        events.append(
            sink.new(
                "child_result_applied",
                {
                    "entry_id": str(e.id),
                    "child_session_id": str(r.child_session_id),
                    "object_id": str(r.object_id),
                    "version": r.version,
                    "error": error,
                },
            )
        )
        await tx.update_child_result(r.id, applied_at=now(), applied_entry_id=e.id)
    return state, events


async def recover_crashed(agents: Agents[Any], tx: StorageTx, s: SessionRecord) -> None:
    """Clôt ``crashed`` un tour dont le bail a expiré, en le reconstruisant depuis ses événements (D28)."""
    turn_id = s.running_turn_id
    assert turn_id is not None
    turn = await tx.get_turn(turn_id)
    leaf = s.leaf_entry_id
    usage = Usage()
    if turn is not None and turn.parent_entry_id is not None:
        attach = await tx.get_entry(turn.parent_entry_id)
        assert attach is not None
        dump: dict[str, Any] | None = None
        version = 1
        try:
            codec = agents.codec(s.kind)
            version = codec.version
            if attach.state_id is not None:
                srec = (await tx.get_states([attach.state_id]))[attach.state_id]
                dump = StateCodec.dump(codec.load(srec.data, srec.state_version))
        except BoutureError:
            log.warning("tour %s : état non relu pendant la reconstruction", turn_id)
        b = Builder(s.id, turn_id, s.kind, version, attach.id, attach.state_id, dump)
        open_msg: dict[str, Any] | None = None
        open_calls: list[ToolCallBlock] = []
        events = []
        cursor: int | None = None
        while batch := await tx.read_events(s.id, after=cursor, turn_id=turn_id):
            events += batch
            cursor = batch[-1].id
        for ev in events:
            d = ev.data
            if ev.type == "user_message":
                c = UserMessage(content=normalize_content(d["content"])).model_dump(mode="json")
                b.add("user", c, entry_id=UUID(d["entry_id"]))
            elif ev.type == "message_start":
                open_msg = {"id": UUID(d["entry_id"]), "blocks": [], "calls": []}
            elif ev.type in ("text_delta", "reasoning_delta") and open_msg is not None:
                cls = TextBlock if ev.type == "text_delta" else ReasoningBlock
                blocks = open_msg["blocks"]
                if blocks and isinstance(blocks[-1], cls):
                    blocks[-1].text += d["text"]
                else:
                    blocks.append(cls(text=d["text"]))
            elif ev.type == "tool_call" and open_msg is not None:
                open_msg["calls"].append(
                    ToolCallBlock(id=d["tool_call_id"], name=d["name"], args=d["args"])
                )
            elif ev.type == "message_end" and open_msg is not None:
                u = Usage.model_validate(d["usage"]) if d.get("usage") else None
                if u is not None:
                    usage = usage + u
                msg = AssistantMessage(content=[*open_msg["blocks"], *open_msg["calls"]], usage=u)
                b.add(
                    "assistant",
                    msg.model_dump(mode="json"),
                    entry_id=open_msg["id"],
                    usage=d.get("usage"),
                )
                open_calls += open_msg["calls"]
                open_msg = None
            elif ev.type == "state_changed" and dump is not None:
                with contextlib.suppress(Exception):
                    b.set_state(jsonpatch.apply(b.state_dump, d["patch"]))
            elif ev.type == "tool_result":
                b.add(
                    "tool_result",
                    tool_result_content(
                        d["tool_call_id"],
                        d["name"],
                        normalize_content(d["blocks"]) if "blocks" in d else d["content"],
                        d["is_error"],
                    ),
                    entry_id=UUID(d["entry_id"]),
                )
                open_calls = [c for c in open_calls if c.id != d["tool_call_id"]]
        if open_msg is not None and open_msg["blocks"]:
            msg = AssistantMessage(content=open_msg["blocks"], interrupted=True)
            b.add(
                "assistant", msg.model_dump(mode="json"), entry_id=open_msg["id"], interrupted=True
            )
        if turn.trigger in ("message", "answer"):
            resolved = {e.content.get("tool_call_id") for e in b.entries if e.type == "tool_result"}
            for q in s.pending:
                if q["tool_call_id"] not in resolved:
                    open_calls.append(
                        ToolCallBlock(id=q["tool_call_id"], name=q["tool_name"], args={})
                    )
        for c in open_calls:
            b.add("tool_result", tool_result_content(c.id, c.name, INTERRUPTED_CALL, True))
        b.finish(checkpoint=True)
        await b.write(tx)
        leaf = b.leaf_id
        await tx.update_turn(
            turn_id,
            status="crashed",
            usage=usage.model_dump(mode="json"),
            ended_at=now(),
            error={
                "code": "crashed",
                "message": "Le processus qui exécutait le tour s'est arrêté.",
            },
        )
    await tx.update_session(
        s.id,
        leaf_entry_id=leaf,
        pending=[],
        running_turn_id=None,
        lock_expires_at=None,
        cancel_requested=False,
        updated_at=now(),
    )
    await tx.append_events(
        [
            NewEvent(
                s.id,
                turn_id,
                "turn_end",
                {
                    "reason": "crashed",
                    "leaf_entry_id": str(leaf) if leaf else None,
                    "usage": usage.model_dump(mode="json"),
                    "error": {
                        "code": "crashed",
                        "message": "Le processus qui exécutait le tour s'est arrêté.",
                    },
                },
                now(),
            )
        ]
    )
    tx.notify_events(s.id)
    s.leaf_entry_id = leaf
    s.pending = []
    s.running_turn_id = None
    s.lock_expires_at = None
    s.cancel_requested = False

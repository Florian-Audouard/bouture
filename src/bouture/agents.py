"""Façade publique : ``Agents``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from .context import PurgeContext, ResolveContext, TurnEndContext
from .definition import AgentDefinition
from .errors import (
    InvalidTarget,
    NotAwaitingInput,
    NotFound,
    ResolveError,
    SessionBusy,
    UnknownKind,
)
from .events import Hub, now, to_event
from .ids import uuid7
from .models.base import ModelClient
from .refs import Data, PublishedRef, Ref
from .runner import InProcessRunner, TurnRunner
from .state import Migration, StateCodec, migrate
from .storage.base import (
    ChildResultRecord,
    EntryRecord,
    InputRecord,
    PublicationRecord,
    SessionRecord,
    StateRecord,
    Storage,
    StorageTx,
    TurnRecord,
)
from .tree import Tree
from .turn import CANCELLED_QUESTION, Builder, TurnExecutor, recover_crashed, tool_result_content
from .views import (
    EntryView,
    Event,
    Page,
    PendingQuestion,
    PublicationInfo,
    SessionInfo,
    Siblings,
    StateView,
    TurnHandle,
    TurnResult,
    entry_view,
)

log = logging.getLogger("bouture")

Resolver = Callable[[Ref | Data, ResolveContext[Any]], Awaitable[str]]
PurgeHook = Callable[[PurgeContext[Any], str, list[UUID]], Awaitable[None]]


class _DataType:
    def __init__(self, model: type[BaseModel], version: int, migrations: Mapping[int, Migration]):
        self.model = model
        self.version = version
        self.migrations = dict(migrations)


class Agents[D]:
    """Point d'entrée de la lib. Toutes les opérations prennent ``owner`` (D61)."""

    def __init__(
        self,
        *,
        engine: Any = None,
        storage: Storage | None = None,
        deps: D = None,
        model_client: ModelClient,
        runner: TurnRunner | None = None,
        lock_ttl: float = 30.0,
        retries: int = 3,
        retry_base_delay: float = 1.0,
        event_retention: float = 3600.0,
        poll_interval: float = 1.0,
        flush_interval: float = 0.1,
        flush_chars: int = 200,
        resolve_cache_size: int = 256,
    ) -> None:
        if (engine is None) == (storage is None):
            raise ValueError("fournir exactement un de engine= ou storage=")
        if storage is None:
            from .storage.postgres import PostgresStorage

            storage = PostgresStorage(engine)
        self.storage: Storage = storage
        self.deps: D = deps  # type: ignore[assignment]
        self.model_client = model_client
        self.runner: TurnRunner = runner or InProcessRunner()
        self.lock_ttl = lock_ttl
        self.retries = retries
        self.retry_base_delay = retry_base_delay
        self.event_retention = event_retention
        self.poll_interval = poll_interval
        self.flush_interval = flush_interval
        self.flush_chars = flush_chars
        self.hub = Hub()
        self._defs: dict[str, AgentDefinition[Any, D]] = {}
        self._codecs: dict[str, StateCodec] = {}
        self._resolvers: dict[str, Resolver] = {}
        self._data_types: dict[str, _DataType] = {}
        self._purge_hooks: list[PurgeHook] = []
        self._resolve_cache: OrderedDict[tuple[str, UUID, int], str] = OrderedDict()
        self._resolve_cache_size = resolve_cache_size
        self._executors: dict[UUID, TurnExecutor] = {}
        self._unlisten: Callable[[], Awaitable[None]] | None = None
        self._cleanup: asyncio.Task[None] | None = None

    # ================================================================================
    # Enregistrement
    # ================================================================================
    def register(self, *definitions: AgentDefinition[Any, D]) -> None:
        for d in definitions:
            if d.kind in self._defs:
                raise ValueError(f"kind {d.kind} déjà enregistré")
            self._defs[d.kind] = d
            self._codecs[d.kind] = StateCodec(
                d.kind,
                d.state_type,
                d.state_version,
                d.state_migrations,  # type: ignore[arg-type]
            )

    def definition(self, kind: str) -> AgentDefinition[Any, D]:
        try:
            return self._defs[kind]
        except KeyError:
            raise UnknownKind(kind) from None

    def codec(self, kind: str) -> StateCodec:
        self.definition(kind)
        return self._codecs[kind]

    def resolver(self, kind: str) -> Callable[[Resolver], Resolver]:
        """Résolveur d'input pour ``kind`` : ``async (Ref | Data, ResolveContext) -> str`` (D55)."""

        def register(fn: Resolver) -> Resolver:
            self._resolvers[kind] = fn
            return fn

        return register

    def data_type(
        self,
        kind: str,
        model: type[BaseModel],
        *,
        version: int = 1,
        migrations: Mapping[int, Migration] | None = None,
    ) -> None:
        """Déclare le type des inputs ``Data(kind, …)`` (validés, versionnés, migrés)."""
        self._data_types[kind] = _DataType(model, version, migrations or {})

    def on_purge(self, fn: PurgeHook) -> PurgeHook:
        self._purge_hooks.append(fn)
        return fn

    # ================================================================================
    # Cycle de vie
    # ================================================================================
    async def setup(self, *, auto_migrate: bool = True) -> None:
        await self.storage.setup(auto_migrate=auto_migrate)
        if self._unlisten is None:
            self._unlisten = await self.storage.listen(self.hub.wake, self._on_cancel)
        if self._cleanup is None:
            self._cleanup = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Clôt les tours locaux (``server_shutdown``) puis arrête l'écoute (D27)."""
        for ex in list(self._executors.values()):
            ex.cancel("server_shutdown")
        await self.runner.join(timeout)
        if self._cleanup is not None:
            self._cleanup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup
            self._cleanup = None
        if self._unlisten is not None:
            await self._unlisten()
            self._unlisten = None
        await self.storage.close()

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                async with self.storage.transaction() as tx:
                    await tx.purge_events(now() - timedelta(seconds=self.event_retention))
            except Exception:
                log.exception("purge du journal d'événements impossible")
            await asyncio.sleep(max(60.0, self.event_retention / 6))

    def _on_cancel(self, turn_id: UUID) -> None:
        ex = self._executors.get(turn_id)
        if ex is not None:
            ex.cancel("interrupted")

    def _lease_expiry(self) -> datetime:
        return now() + timedelta(seconds=self.lock_ttl)

    # ================================================================================
    # Outils internes
    # ================================================================================
    async def _session(
        self, tx: StorageTx, session_id: UUID, owner: str | None, *, for_update: bool = False
    ) -> SessionRecord:
        s = await tx.get_session(session_id, for_update=for_update)
        if s is None or (owner is not None and s.owner != owner):
            raise NotFound(f"session {session_id} introuvable")
        return s

    async def _idle_session(
        self, tx: StorageTx, session_id: UUID, owner: str | None
    ) -> SessionRecord:
        """Session verrouillée et sans tour actif (un bail expiré est d'abord clos ``crashed``)."""
        s = await self._session(tx, session_id, owner, for_update=True)
        if s.running_turn_id is not None:
            if s.lock_expires_at is not None and s.lock_expires_at > now():
                raise SessionBusy(f"un tour est en cours sur la session {session_id}")
            await recover_crashed(self, tx, s)
        return s

    def _info(self, s: SessionRecord) -> SessionInfo:
        if s.running_turn_id is not None:
            status = "running"
        elif s.pending:
            status = "awaiting_input"
        else:
            status = "idle"
        return SessionInfo(
            id=s.id,
            kind=s.kind,
            owner=s.owner,
            title=s.title,
            metadata=s.metadata,
            model=s.model_override or (self._defs[s.kind].model if s.kind in self._defs else ""),
            status=status,
            leaf_entry_id=s.leaf_entry_id,
            pending=[PendingQuestion.model_validate(p) for p in s.pending],
            object_id=s.object_id,
            cloned_from_session_id=s.cloned_from_session_id,
            cloned_from_entry_id=s.cloned_from_entry_id,
            origin_session_id=s.origin_session_id,
            origin_entry_id=s.origin_entry_id,
            archived_at=s.archived_at,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )

    async def _state_at(self, tx: StorageTx, kind: str, entry: EntryRecord) -> Any:
        codec = self.codec(kind)
        if entry.state_id is None:
            return None
        rec = (await tx.get_states([entry.state_id]))[entry.state_id]
        return codec.load(rec.data, rec.state_version)

    def _load_inputs(self, recs: list[InputRecord]) -> dict[str, Ref | Data]:
        out: dict[str, Ref | Data] = {}
        for r in sorted(recs, key=lambda r: r.name):
            if r.mode == "ref":
                assert r.object_id is not None
                out[r.name] = Ref(r.kind, r.object_id, r.version)
            else:
                dt = self._data_types.get(r.kind)
                value = r.data
                if dt is not None:
                    value = dt.model.model_validate(
                        migrate(
                            value, r.data_version or 1, dt.version, dt.migrations, f"input {r.kind}"
                        )
                    )
                out[r.name] = Data(r.kind, value)
        return out

    async def _resolve_inputs(
        self, inputs: Mapping[str, Ref | Data], owner: str, session_id: UUID | None
    ) -> dict[str, str]:
        ctx = ResolveContext[Any](deps=self.deps, owner=owner, session_id=session_id)
        out: dict[str, str] = {}
        for name, inp in inputs.items():
            out[name] = await self._resolve_one(inp, ctx)
        return out

    async def _resolve_one(self, inp: Ref | Data, ctx: ResolveContext[Any]) -> str:
        fn = self._resolvers.get(inp.kind)
        if isinstance(inp, Data):
            if fn is not None:
                return await fn(inp, ctx)
            v = inp.value.model_dump(mode="json") if isinstance(inp.value, BaseModel) else inp.value
            return json.dumps(v, ensure_ascii=False, indent=2, default=str)
        assert inp.version is not None
        key = (inp.kind, inp.id, inp.version)
        if key in self._resolve_cache:
            self._resolve_cache.move_to_end(key)
            return self._resolve_cache[key]
        if fn is not None:
            text = await fn(inp, ctx)
        else:
            state = await self.get_published_state(inp)
            text = (
                state.model_dump_json(indent=2)
                if isinstance(state, BaseModel)
                else json.dumps(state)
            )
        self._resolve_cache[key] = text
        if len(self._resolve_cache) > self._resolve_cache_size:
            self._resolve_cache.popitem(last=False)
        return text

    # ================================================================================
    # Sessions
    # ================================================================================
    async def create_session(
        self,
        kind: str,
        owner: str,
        *,
        inputs: Mapping[str, Ref | Data] | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        _origin: dict[str, Any] | None = None,
    ) -> SessionInfo:
        d = self.definition(kind)
        inputs = dict(inputs or {})
        missing = set(d.inputs) - set(inputs)
        extra = set(inputs) - set(d.inputs)
        if missing or extra:
            raise ResolveError(
                f"inputs attendus pour {kind} : {sorted(d.inputs)} ; reçus : {sorted(inputs)}"
            )
        sid = uuid7()
        ts = now()
        async with self.storage.transaction() as tx:
            records: list[InputRecord] = []
            pinned: dict[str, Ref | Data] = {}
            for name, inp in sorted(inputs.items()):
                if inp.kind != d.inputs[name]:
                    raise ResolveError(
                        f"input {name} : kind {d.inputs[name]} attendu, {inp.kind} reçu"
                    )
                if isinstance(inp, Ref):
                    pubs = await tx.get_publications(inp.id)
                    if not pubs or pubs[0].kind != inp.kind:
                        raise ResolveError(f"input {name} : objet publié {inp.id} introuvable")
                    version = inp.version if inp.version is not None else pubs[-1].version
                    if not any(p.version == version for p in pubs):
                        raise ResolveError(f"input {name} : version {version} introuvable")
                    records.append(InputRecord(sid, name, "ref", inp.kind, inp.id, version))
                    pinned[name] = Ref(inp.kind, inp.id, version)
                else:
                    dt = self._data_types.get(inp.kind)
                    if dt is not None:
                        try:
                            value = dt.model.model_validate(
                                inp.value.model_dump()
                                if isinstance(inp.value, BaseModel)
                                else inp.value
                            )
                        except ValidationError as e:
                            raise ResolveError(f"input {name} : {e}") from e
                        records.append(
                            InputRecord(
                                sid,
                                name,
                                "data",
                                inp.kind,
                                data=value.model_dump(mode="json"),
                                data_version=dt.version,
                            )
                        )
                        pinned[name] = Data(inp.kind, value)
                    else:
                        records.append(InputRecord(sid, name, "data", inp.kind, data=inp.value))
                        pinned[name] = inp
            state = d.make_initial_state(pinned)
            codec = self._codecs[kind]
            state_id: UUID | None = None
            if state is not None:
                state = codec.validate(state)
                state_id = uuid7()
                await tx.insert_states(
                    [StateRecord(state_id, kind, codec.version, StateCodec.dump(state), ts)]
                )
            root = EntryRecord(uuid7(), sid, None, "root", {}, state_id, True, ts)
            o = _origin or {}
            rec = SessionRecord(
                id=sid,
                kind=kind,
                owner=owner,
                created_at=ts,
                updated_at=ts,
                title=title,
                metadata=dict(metadata or {}),
                model_override=model,
                leaf_entry_id=root.id,
                origin_session_id=o.get("session_id"),
                origin_entry_id=o.get("entry_id"),
                origin_handler=o.get("handler"),
                origin_payload=o.get("payload"),
            )
            await tx.insert_session(rec)
            await tx.insert_inputs(records)
            await tx.insert_entries([root])
        return self._info(rec)

    async def get_session(self, session_id: UUID, owner: str) -> SessionInfo:
        async with self.storage.transaction() as tx:
            return self._info(await self._session(tx, session_id, owner))

    async def list_sessions(
        self,
        owner: str,
        *,
        kind: str | None = None,
        origin_session_id: UUID | None = None,
        archived: bool = False,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[SessionInfo]:
        before: tuple[datetime, UUID] | None = None
        if cursor:
            try:
                ts, _, sid = cursor.partition("|")
                before = (datetime.fromisoformat(ts), UUID(sid))
            except ValueError:
                raise InvalidTarget("curseur invalide") from None
        async with self.storage.transaction() as tx:
            rows = await tx.list_sessions(
                owner,
                kind=kind,
                origin_session_id=origin_session_id,
                archived=archived,
                limit=limit + 1,
                before=before,
            )
        nxt = None
        if len(rows) > limit:
            rows = rows[:limit]
            nxt = f"{rows[-1].updated_at.isoformat()}|{rows[-1].id}"
        return Page[SessionInfo](items=[self._info(r) for r in rows], next_cursor=nxt)

    async def _patch_session(self, session_id: UUID, owner: str, **fields: Any) -> SessionInfo:
        async with self.storage.transaction() as tx:
            await self._session(tx, session_id, owner, for_update=True)
            await tx.update_session(session_id, updated_at=now(), **fields)
            s = await self._session(tx, session_id, owner)
        return self._info(s)

    async def rename(self, session_id: UUID, owner: str, title: str | None) -> SessionInfo:
        return await self._patch_session(session_id, owner, title=title)

    async def update_metadata(
        self, session_id: UUID, owner: str, metadata: dict[str, Any]
    ) -> SessionInfo:
        return await self._patch_session(session_id, owner, metadata=dict(metadata))

    async def set_model(self, session_id: UUID, owner: str, model: str | None) -> SessionInfo:
        return await self._patch_session(session_id, owner, model_override=model)

    async def archive(self, session_id: UUID, owner: str) -> SessionInfo:
        return await self._patch_session(session_id, owner, archived_at=now())

    async def unarchive(self, session_id: UUID, owner: str) -> SessionInfo:
        return await self._patch_session(session_id, owner, archived_at=None)

    async def clone(self, session_id: UUID, owner: str, *, title: str | None = None) -> SessionInfo:
        """Nouvelle session indépendante ; ne copie que la branche active (D17 ⚠️)."""
        async with self.storage.transaction() as tx:
            s = await self._idle_session(tx, session_id, owner)
            tree = Tree(await tx.get_entries(session_id))
            assert s.leaf_entry_id is not None
            target = tree.nearest_checkpoint(s.leaf_entry_id)
            path = tree.path(target.id)
            inputs = await tx.get_inputs(session_id)
            new_id = uuid7()
            ts = now()
            mapping: dict[UUID, UUID] = {}
            copies: list[EntryRecord] = []
            for e in path:
                mapping[e.id] = uuid7()
                copies.append(
                    EntryRecord(
                        id=mapping[e.id],
                        session_id=new_id,
                        parent_id=mapping[e.parent_id] if e.parent_id else None,
                        type=e.type,
                        content=e.content,
                        state_id=e.state_id,
                        is_checkpoint=e.is_checkpoint,
                        created_at=e.created_at,
                        interrupted=e.interrupted,
                        turn_id=None,
                        usage=e.usage,
                    )
                )
            rec = SessionRecord(
                id=new_id,
                kind=s.kind,
                owner=owner,
                created_at=ts,
                updated_at=ts,
                title=title if title is not None else s.title,
                metadata=dict(s.metadata),
                model_override=s.model_override,
                leaf_entry_id=mapping[target.id],
                cloned_from_session_id=s.id,
                cloned_from_entry_id=target.id,
            )
            await tx.insert_session(rec)
            for r in inputs:
                r.session_id = new_id
            await tx.insert_inputs(inputs)
            await tx.insert_entries(copies)
        return self._info(rec)

    # ================================================================================
    # Tours
    # ================================================================================
    async def _start_turn(
        self,
        session_id: UUID,
        owner: str | None,
        trigger: str,
        *,
        entry_id: UUID | None = None,
        content: Any = None,
        answer: Any = None,
        tool_call_id: str | None = None,
    ) -> TurnHandle | None:
        turn_id = uuid7()
        async with self.storage.transaction() as tx:
            s = await self._idle_session(tx, session_id, owner)
            assert s.leaf_entry_id is not None
            attach = s.leaf_entry_id
            if trigger == "answer":
                if not s.pending:
                    raise NotAwaitingInput("aucune question en attente")
                if tool_call_id is not None and not any(
                    p["tool_call_id"] == tool_call_id for p in s.pending
                ):
                    raise NotAwaitingInput(f"aucune question en attente pour {tool_call_id}")
            elif trigger in ("fork", "regenerate"):
                assert entry_id is not None
                tree = Tree(await tx.get_entries(session_id))
                target = tree.get(entry_id)
                if trigger == "fork":
                    if target.type != "user" or target.parent_id is None:
                        raise InvalidTarget(
                            "fork : l'entrée visée doit être un message utilisateur"
                        )
                    attach = target.parent_id
                else:
                    attach = tree.turn_user(entry_id).id
            elif trigger == "child_result":
                if s.pending or not await tx.pending_child_results(session_id):
                    return None
            await tx.insert_turn(
                TurnRecord(turn_id, session_id, trigger, "running", now(), parent_entry_id=attach)
            )
            await tx.update_session(
                session_id,
                running_turn_id=turn_id,
                lock_expires_at=self._lease_expiry(),
                cancel_requested=False,
                updated_at=now(),
            )
        ex = TurnExecutor(
            self,
            session_id,
            turn_id,
            trigger,
            s.owner,
            content=content,
            answer=answer,
            tool_call_id=tool_call_id,
        )
        self._executors[turn_id] = ex
        self.runner.start(turn_id, ex.run)
        return TurnHandle(session_id, turn_id, lambda: self._turn_events(session_id, turn_id))

    async def send_message(self, session_id: UUID, owner: str, content: Any) -> TurnHandle:
        """``content`` : ``str`` ou liste de blocs. Une question en attente est annulée (D32)."""
        h = await self._start_turn(session_id, owner, "message", content=content)
        assert h is not None
        return h

    async def answer(
        self, session_id: UUID, owner: str, answer: Any, *, tool_call_id: str | None = None
    ) -> TurnHandle:
        h = await self._start_turn(
            session_id, owner, "answer", answer=answer, tool_call_id=tool_call_id
        )
        assert h is not None
        return h

    async def fork(self, session_id: UUID, owner: str, entry_id: UUID, content: Any) -> TurnHandle:
        """Édite un message utilisateur déjà envoyé : nouvelle branche sœur (D14)."""
        h = await self._start_turn(session_id, owner, "fork", entry_id=entry_id, content=content)
        assert h is not None
        return h

    async def regenerate(self, session_id: UUID, owner: str, entry_id: UUID) -> TurnHandle:
        """Nouvelle réponse au message utilisateur du tour contenant ``entry_id`` (D15)."""
        h = await self._start_turn(session_id, owner, "regenerate", entry_id=entry_id)
        assert h is not None
        return h

    async def stop(self, session_id: UUID, owner: str) -> None:
        """Demande explicite d'arrêt (D26). Idempotent."""
        async with self.storage.transaction() as tx:
            s = await self._session(tx, session_id, owner, for_update=True)
            turn_id = s.running_turn_id
            if turn_id is None:
                return
            await tx.update_session(session_id, cancel_requested=True)
            tx.notify_cancel(turn_id)
        self._on_cancel(turn_id)

    async def _after_turn(self, result: TurnResult, owner: str, kind: str) -> None:
        d = self._defs.get(kind)
        if d is not None and d.turn_end_hook is not None:
            try:
                await d.turn_end_hook(
                    TurnEndContext[Any](
                        deps=self.deps, owner=owner, session_id=result.session_id, kind=kind
                    ),
                    result,
                )
            except Exception:
                log.exception("on_turn_end a échoué (session %s)", result.session_id)
        await self._kick_children(result.session_id)

    async def _kick_children(self, session_id: UUID) -> None:
        """Lance un mini-tour si des résultats d'enfants attendent et que la session est libre."""
        with contextlib.suppress(SessionBusy, NotFound):
            await self._start_turn(session_id, None, "child_result")

    async def _spawn_child(
        self,
        parent: SessionRecord,
        owner: str,
        origin_entry_id: UUID,
        sink: Any,
        kind: str,
        *,
        inputs: Mapping[str, Ref | Data] | None = None,
        on_result: str | None = None,
        payload: Any = None,
        initial_message: str | None = None,
        title: str | None = None,
    ) -> SessionInfo:
        pd = self.definition(parent.kind)
        if on_result is not None and on_result not in pd.child_handlers:
            raise InvalidTarget(f"child_handler « {on_result} » absent de {parent.kind}")
        info = await self.create_session(
            kind,
            owner,
            inputs=inputs,
            title=title,
            _origin={
                "session_id": parent.id,
                "entry_id": origin_entry_id,
                "handler": on_result,
                "payload": payload,
            },
        )
        await sink.emit("child_spawned", {"session_id": str(info.id), "kind": kind})
        if initial_message is not None:
            await self.send_message(info.id, owner, initial_message)
        return info

    # ================================================================================
    # Événements
    # ================================================================================
    async def _follow(
        self, session_id: UUID, after: int | None, turn_id: UUID | None
    ) -> AsyncIterator[Event]:
        cursor = after
        idle = 0.0
        while True:
            waiter = self.hub.waiter(session_id)
            async with self.storage.transaction() as tx:
                rows = await tx.read_events(session_id, after=cursor, turn_id=turn_id)
            for r in rows:
                cursor = r.id
                ev = to_event(r)
                yield ev
                if turn_id is not None and ev.type == "turn_end":
                    return
            if rows:
                idle = 0.0
                continue
            try:
                await asyncio.wait_for(waiter.wait(), self.poll_interval)
            except TimeoutError:
                idle += self.poll_interval
                if idle >= 2 * self.poll_interval:
                    idle = 0.0
                    await self._recover_if_expired(session_id)

    async def _recover_if_expired(self, session_id: UUID) -> None:
        async with self.storage.transaction() as tx:
            s = await tx.get_session(session_id, for_update=True)
            if (
                s is not None
                and s.running_turn_id is not None
                and s.lock_expires_at is not None
                and s.lock_expires_at <= now()
            ):
                await recover_crashed(self, tx, s)
                recovered = True
            else:
                recovered = False
        if recovered:
            self.hub.wake(session_id)
            await self._kick_children(session_id)

    def _turn_events(self, session_id: UUID, turn_id: UUID) -> AsyncIterator[Event]:
        return self._follow(session_id, None, turn_id)

    async def subscribe(
        self, session_id: UUID, owner: str, *, after: int | None = None
    ) -> AsyncIterator[Event]:
        """Rejoue les événements après ``after`` (``Last-Event-ID``) puis suit en direct (D24).

        Sans ``after`` : seulement les événements à venir. Ne se termine jamais d'elle-même.
        """
        async with self.storage.transaction() as tx:
            await self._session(tx, session_id, owner)
            if after is None:
                last = await tx.read_events(session_id, after=None, limit=1_000_000)
                after = last[-1].id if last else 0
        async for ev in self._follow(session_id, after, None):
            yield ev

    # ================================================================================
    # Arbre
    # ================================================================================
    async def get_tree(self, session_id: UUID, owner: str) -> list[EntryView]:
        async with self.storage.transaction() as tx:
            await self._session(tx, session_id, owner)
            return [entry_view(e) for e in await tx.get_entries(session_id)]

    async def get_branch(
        self, session_id: UUID, owner: str, *, entry_id: UUID | None = None
    ) -> list[EntryView]:
        async with self.storage.transaction() as tx:
            s = await self._session(tx, session_id, owner)
            tree = Tree(await tx.get_entries(session_id))
        target = entry_id or s.leaf_entry_id
        assert target is not None
        return [entry_view(e) for e in tree.path(target)]

    async def siblings(self, session_id: UUID, owner: str, entry_id: UUID) -> Siblings:
        async with self.storage.transaction() as tx:
            await self._session(tx, session_id, owner)
            tree = Tree(await tx.get_entries(session_id))
        sibs, idx = tree.siblings(entry_id)
        return Siblings(entries=[entry_view(e) for e in sibs], index=idx)

    async def switch_branch(self, session_id: UUID, owner: str, entry_id: UUID) -> SessionInfo:
        """Feuille = descendant le plus récent de ``entry_id``, ramené au checkpoint le plus proche (D16)."""
        async with self.storage.transaction() as tx:
            await self._idle_session(tx, session_id, owner)
            tree = Tree(await tx.get_entries(session_id))
            target = tree.nearest_checkpoint(tree.latest_descendant(entry_id).id)
            await tx.update_session(
                session_id, leaf_entry_id=target.id, pending=[], updated_at=now()
            )
            s = await self._session(tx, session_id, owner)
        return self._info(s)

    async def rollback(self, session_id: UUID, owner: str, entry_id: UUID) -> SessionInfo:
        """Déplace la feuille sur un checkpoint. Rien n'est supprimé (D13)."""
        async with self.storage.transaction() as tx:
            await self._idle_session(tx, session_id, owner)
            target = Tree(await tx.get_entries(session_id)).get(entry_id)
            if not target.is_checkpoint:
                raise InvalidTarget("rollback : l'entrée visée n'est pas un checkpoint")
            await tx.update_session(
                session_id, leaf_entry_id=target.id, pending=[], updated_at=now()
            )
            s = await self._session(tx, session_id, owner)
        return self._info(s)

    # ================================================================================
    # État
    # ================================================================================
    async def get_state(
        self, session_id: UUID, owner: str, *, entry_id: UUID | None = None
    ) -> StateView:
        async with self.storage.transaction() as tx:
            s = await self._session(tx, session_id, owner)
            target_id = entry_id or s.leaf_entry_id
            assert target_id is not None
            entry = await tx.get_entry(target_id)
            if entry is None or entry.session_id != session_id:
                raise NotFound(f"entrée {target_id} introuvable")
            state = await self._state_at(tx, s.kind, entry)
        return StateView(
            state=state,
            state_version=self.codec(s.kind).version,
            entry_id=entry.id,
            state_id=entry.state_id,
        )

    async def edit_state(self, session_id: UUID, owner: str, state: Any) -> EntryView:
        """Édition manuelle : entrée ``state_edit`` (checkpoint) ; annule les questions en attente (D43)."""
        async with self.storage.transaction() as tx:
            s = await self._idle_session(tx, session_id, owner)
            codec = self.codec(s.kind)
            if codec.state_type is None:
                raise InvalidTarget(f"le kind {s.kind} n'a pas d'état")
            new_state = codec.validate(state)
            assert s.leaf_entry_id is not None
            leaf = await tx.get_entry(s.leaf_entry_id)
            assert leaf is not None
            old = StateCodec.dump(await self._state_at(tx, s.kind, leaf))
            new = StateCodec.dump(new_state)
            b = Builder(session_id, None, s.kind, codec.version, leaf.id, leaf.state_id, old)
            for q in s.pending:
                b.add(
                    "tool_result",
                    tool_result_content(
                        q["tool_call_id"], q["tool_name"], CANCELLED_QUESTION, False
                    ),
                )
            patch = b.set_state(new)
            if not patch and not s.pending:
                return entry_view(leaf)
            e = b.add("state_edit", {"patch": patch}, checkpoint=True)
            await b.write(tx)
            await tx.update_session(session_id, leaf_entry_id=e.id, pending=[], updated_at=now())
        return entry_view(e)

    # ================================================================================
    # Publication
    # ================================================================================
    async def publish(self, session_id: UUID, owner: str) -> PublishedRef:
        """Publie l'état de la feuille (checkpoint) comme nouvelle version de l'objet (D50–D53)."""
        async with self.storage.transaction() as tx:
            s = await self._idle_session(tx, session_id, owner)
            d = self.definition(s.kind)
            if d.state_type is None:
                raise InvalidTarget(f"le kind {s.kind} n'a pas d'état à publier")
            if s.pending:
                raise InvalidTarget("publication impossible pendant une question en attente")
            assert s.leaf_entry_id is not None
            leaf = await tx.get_entry(s.leaf_entry_id)
            assert leaf is not None and leaf.state_id is not None
            if not leaf.is_checkpoint:
                raise InvalidTarget("publication : la feuille n'est pas un checkpoint")
            object_id = s.object_id or uuid7()
            pubs = await tx.get_publications(object_id)
            if pubs and pubs[-1].state_id == leaf.state_id:
                return PublishedRef(s.kind, object_id, pubs[-1].version)
            version = pubs[-1].version + 1 if pubs else 1
            ref = PublishedRef(s.kind, object_id, version)
            state = await self._state_at(tx, s.kind, leaf)
            await tx.insert_publication(
                PublicationRecord(
                    s.kind, object_id, version, session_id, leaf.id, leaf.state_id, owner, now()
                )
            )
            await tx.update_session(session_id, object_id=object_id, updated_at=now())
            if d.publish_hook is not None:
                from .context import PublishContext

                await d.publish_hook(
                    PublishContext[Any](
                        db=tx.db, deps=self.deps, owner=owner, session_id=session_id
                    ),
                    ref,
                    state,
                )
            parent_id = s.origin_session_id
            if parent_id is not None:
                await tx.insert_child_result(
                    ChildResultRecord(uuid7(), parent_id, session_id, object_id, version, now())
                )
        if parent_id is not None:
            await self._kick_children(parent_id)
        return ref

    async def list_publications(
        self, owner: str, kind: str, object_id: UUID
    ) -> list[PublicationInfo]:
        """Versions publiées par ``owner`` d'un objet."""
        async with self.storage.transaction() as tx:
            pubs = await tx.get_publications(object_id)
        return [
            PublicationInfo(
                kind=p.kind,
                object_id=p.object_id,
                version=p.version,
                session_id=p.session_id,
                entry_id=p.entry_id,
                state_id=p.state_id,
                owner=p.owner,
                published_at=p.published_at,
            )
            for p in pubs
            if p.owner == owner and p.kind == kind
        ]

    async def get_published_state(self, ref: Ref | PublishedRef, owner: str | None = None) -> Any:
        """État publié (migré). ``owner`` restreint aux publications de ce propriétaire.

        ``version=None`` → dernière version. L'autorisation fine reste au projet (D55).
        """
        object_id = ref.id if isinstance(ref, Ref) else ref.object_id
        version = ref.version
        async with self.storage.transaction() as tx:
            pubs = await tx.get_publications(object_id)
            if version is not None:
                pubs = [p for p in pubs if p.version == version]
            if (
                not pubs
                or pubs[-1].kind != ref.kind
                or (owner is not None and pubs[-1].owner != owner)
            ):
                raise NotFound(f"publication {ref.kind}/{object_id}@{version} introuvable")
            pub = pubs[-1]
            srec = (await tx.get_states([pub.state_id]))[pub.state_id]
        return self.codec(pub.kind).load(srec.data, srec.state_version)

    # ================================================================================
    # Purge (admin, D68)
    # ================================================================================
    async def purge_session(self, session_id: UUID) -> None:
        async with self.storage.transaction() as tx:
            s = await tx.get_session(session_id)
            if s is None:
                return
            owner = s.owner
        await self._purge(owner, [session_id])

    async def purge_owner(self, owner: str) -> None:
        async with self.storage.transaction() as tx:
            ids = await tx.sessions_of_owner(owner)
        await self._purge(owner, ids)

    async def _purge(self, owner: str, ids: list[UUID]) -> None:
        if not ids:
            return
        for ex in list(self._executors.values()):
            if ex.session_id in ids:
                ex.cancel("abandon")
        ctx = PurgeContext[Any](deps=self.deps)
        for hook in self._purge_hooks:
            await hook(ctx, owner, ids)
        async with self.storage.transaction() as tx:
            await tx.delete_sessions(ids)

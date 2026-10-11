"""Définition d'un agent (kind) : type d'état, outils, prompt, modèle et crochets."""

from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload
from uuid import UUID

from pydantic import BaseModel, ConfigDict, create_model

from .models.base import ToolSpec

if TYPE_CHECKING:
    from .context import (
        BuildContext,
        ChildResultContext,
        EntriesContext,
        InitContext,
        PublishContext,
        TurnEndContext,
    )
    from .models.base import ModelRequest
    from .refs import PublishedRef
    from .views import EntryView, TurnResult


@dataclass
class AskUser:
    """Valeur de retour d'un outil qui met le tour en pause (D30)."""

    question: str
    choices: list[str] | None = None
    payload: Any = None


ToolFn = Callable[..., Awaitable[Any]]
AnswerFn = Callable[..., Awaitable[Any]]
StateMigration = Callable[[dict[str, Any]], dict[str, Any]]

DEFAULT_OVERRIDABLE_SETTINGS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "seed",
        "stop",
    }
)
"""Réglages qu'une session peut surcharger par défaut (D92) : échantillonnage uniquement."""


class Tool:
    def __init__(
        self, fn: ToolFn, *, name: str | None = None, timeout: float | None = None
    ) -> None:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"l'outil {fn.__name__} doit être une fonction async")
        self.fn = fn
        self.name = name or fn.__name__
        self.description = inspect.getdoc(fn) or ""
        self.timeout = timeout
        self.answer_fn: AnswerFn | None = None
        self.params_model = _params_model(fn, self.name)

    def on_answer(self, fn: AnswerFn) -> AnswerFn:
        """Traite la réponse de l'utilisateur à un ``AskUser`` de cet outil (D31)."""
        self.answer_fn = fn
        return fn

    def spec(self) -> ToolSpec:
        schema = self.params_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(self.name, self.description, schema)


def _params_model(fn: ToolFn, name: str) -> type[BaseModel]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[1:]  # le premier paramètre est le contexte
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    fields: dict[str, Any] = {}
    for p in params:
        hint = hints.get(
            p.name, p.annotation if p.annotation is not inspect.Parameter.empty else Any
        )
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (hint, default)
    return create_model(  # type: ignore[call-overload]
        f"{name}_params", __config__=ConfigDict(extra="forbid"), **fields
    )


class AgentDefinition[S, D]:
    """Un kind d'agent. ``S`` = type d'état (modèle Pydantic ou ``None``), ``D`` = deps."""

    def __init__(
        self,
        kind: str,
        *,
        model: str,
        state_type: type[S] | None = None,
        initial_state: S | Callable[[Mapping[str, Any]], S] | None = None,
        state_version: int = 1,
        state_migrations: Mapping[int, StateMigration] | None = None,
        system_prompt: str | Callable[[BuildContext[S, D]], str] = "",
        model_settings: Mapping[str, Any] | None = None,
        inputs: Mapping[str, str] | None = None,
        max_steps: int = 25,
        tool_timeout: float = 120.0,
        overridable_settings: Collection[str] | None = None,
        stream_tool_args: bool = False,
    ) -> None:
        if state_type is not None and not (
            isinstance(state_type, type)  # pyright: ignore[reportUnnecessaryIsInstance]
            and issubclass(state_type, BaseModel)
        ):
            raise TypeError("state_type doit être un modèle Pydantic")
        self.kind = kind
        self.model = model
        self.state_type = state_type
        self._initial_state = initial_state
        self.state_version = state_version
        self.state_migrations = dict(state_migrations or {})
        self.system_prompt = system_prompt
        self.model_settings = dict(model_settings or {})
        self.inputs = dict(inputs or {})
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.overridable_settings = frozenset(
            DEFAULT_OVERRIDABLE_SETTINGS if overridable_settings is None else overridable_settings
        )
        self.stream_tool_args = stream_tool_args
        self.tools: dict[str, Tool] = {}
        self.publish_hook: (
            Callable[[PublishContext[D], PublishedRef, S], Awaitable[None]] | None
        ) = None
        self.child_handlers: dict[str, Callable[..., Awaitable[None]]] = {}
        self.state_renderer: Callable[[S], str] | None = None
        self.context_builder: Callable[[BuildContext[S, D]], Awaitable[ModelRequest]] | None = None
        self.turn_end_hook: Callable[[TurnEndContext[D], TurnResult], Awaitable[None]] | None = None
        self.initial_messages_hook: (
            Callable[[InitContext[S, D]], Awaitable[Sequence[tuple[str, Any]]]] | None
        ) = None
        self.entries_hook: (
            Callable[[EntriesContext[D], list[EntryView]], Awaitable[Mapping[UUID, dict[str, Any]]]]
            | None
        ) = None

    # -- état initial ----------------------------------------------------------------
    def make_initial_state(self, inputs: Mapping[str, Any]) -> S | None:
        if self.state_type is None:
            return None
        init = self._initial_state
        if init is None:
            return self.state_type()  # type: ignore[call-arg]
        if callable(init) and not isinstance(init, BaseModel):
            return init(inputs)  # type: ignore[call-arg]
        return init.model_copy(deep=True)  # type: ignore[union-attr]

    # -- décorateurs -----------------------------------------------------------------
    @overload
    def tool(self, fn: Callable[..., Awaitable[Any]], /) -> Tool: ...
    @overload
    def tool(
        self, /, *, name: str | None = None, timeout: float | None = None
    ) -> Callable[[Callable[..., Awaitable[Any]]], Tool]: ...
    def tool(
        self,
        fn: Callable[..., Awaitable[Any]] | None = None,
        /,
        *,
        name: str | None = None,
        timeout: float | None = None,
    ) -> Tool | Callable[[Callable[..., Awaitable[Any]]], Tool]:
        """Enregistre un outil. Le premier paramètre reçoit un ``ToolContext[S, D]``."""

        def register(f: Callable[..., Awaitable[Any]]) -> Tool:
            t = Tool(f, name=name, timeout=timeout)
            if t.name in self.tools:
                raise ValueError(f"outil {t.name} déjà enregistré pour {self.kind}")
            self.tools[t.name] = t
            return t

        return register(fn) if fn is not None else register

    def on_publish(
        self, fn: Callable[[PublishContext[D], PublishedRef, S], Awaitable[None]]
    ) -> Callable[[PublishContext[D], PublishedRef, S], Awaitable[None]]:
        self.publish_hook = fn
        return fn

    def child_handler(
        self, name: str
    ) -> Callable[
        [Callable[[ChildResultContext[S, D], PublishedRef, Any, Any], Awaitable[None]]], Any
    ]:
        def register(
            fn: Callable[[ChildResultContext[S, D], PublishedRef, Any, Any], Awaitable[None]],
        ) -> Any:
            self.child_handlers[name] = fn
            return fn

        return register

    def render_state(self, fn: Callable[[S], str]) -> Callable[[S], str]:
        self.state_renderer = fn
        return fn

    def build_context(
        self, fn: Callable[[BuildContext[S, D]], Awaitable[ModelRequest]]
    ) -> Callable[[BuildContext[S, D]], Awaitable[ModelRequest]]:
        self.context_builder = fn
        return fn

    def on_turn_end(
        self, fn: Callable[[TurnEndContext[D], TurnResult], Awaitable[None]]
    ) -> Callable[[TurnEndContext[D], TurnResult], Awaitable[None]]:
        self.turn_end_hook = fn
        return fn

    def initial_messages(
        self, fn: Callable[[InitContext[S, D]], Awaitable[Sequence[tuple[str, Any]]]]
    ) -> Callable[[InitContext[S, D]], Awaitable[Sequence[tuple[str, Any]]]]:
        """Messages posés à la création, sans appel LLM : ``[("assistant", "Bonjour…")]`` (D96)."""
        self.initial_messages_hook = fn
        return fn

    def on_entries(
        self,
        fn: Callable[
            [EntriesContext[D], list[EntryView]], Awaitable[Mapping[UUID, dict[str, Any]]]
        ],
    ) -> Callable[[EntriesContext[D], list[EntryView]], Awaitable[Mapping[UUID, dict[str, Any]]]]:
        """Métadonnées des entrées d'un tour, écrites dans la transaction du commit (D101)."""
        self.entries_hook = fn
        return fn

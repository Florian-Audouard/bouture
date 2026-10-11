"""Contextes passés aux outils et aux crochets du projet, et construction du contexte LLM."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .messages import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolResultMessage,
    UserMessage,
    normalize_content,
)
from .models.base import ModelRequest
from .refs import Data, Ref

if TYPE_CHECKING:
    from .definition import AgentDefinition
    from .views import EntryView, SessionInfo


@dataclass
class ToolContext[S, D]:
    """Reçu en premier argument par chaque outil (et par ``on_answer``)."""

    state: S
    deps: D
    owner: str
    session_id: UUID
    turn_id: UUID
    entry_id: UUID
    inputs: Mapping[str, Ref | Data]
    _spawn: Callable[..., Awaitable[SessionInfo]] = field(repr=False)
    _emit: Callable[[str, dict[str, Any]], Awaitable[None]] = field(repr=False)

    async def spawn_child(
        self,
        kind: str,
        *,
        inputs: Mapping[str, Ref | Data] | None = None,
        on_result: str | None = None,
        payload: Any = None,
        initial_message: str | None = None,
        title: str | None = None,
    ) -> SessionInfo:
        """Crée une session enfant (D57). Son résultat publié sera appliqué via ``on_result``."""
        return await self._spawn(
            kind,
            inputs=inputs,
            on_result=on_result,
            payload=payload,
            initial_message=initial_message,
            title=title,
        )

    async def emit(self, name: str, data: dict[str, Any] | None = None) -> None:
        """Émet un événement ``custom`` dans le flux du tour."""
        await self._emit(name, data or {})


@dataclass
class ChildResultContext[S, D]:
    state: S
    deps: D
    owner: str
    session_id: UUID
    child_session_id: UUID


@dataclass
class PublishContext[D]:
    """``db`` est une ``AsyncSession`` liée à la transaction de publication (``None`` en mémoire)."""

    db: Any
    deps: D
    owner: str
    session_id: UUID


@dataclass
class ResolveContext[D]:
    deps: D
    owner: str
    session_id: UUID | None


@dataclass
class TurnEndContext[D]:
    deps: D
    owner: str
    session_id: UUID
    kind: str


@dataclass
class PurgeContext[D]:
    deps: D


@dataclass
class BuildContext[S, D]:
    """Reçu par ``build_context`` ; ``default_request()`` donne le comportement par défaut."""

    definition: AgentDefinition[S, D]
    state: S | None
    deps: D
    owner: str
    session_id: UUID
    model: str
    inputs: Mapping[str, Ref | Data]
    resolved_inputs: Mapping[str, str]
    branch: Sequence[EntryView]
    tools: list[Any]

    def system_prompt(self) -> str:
        sp = self.definition.system_prompt
        return sp(self) if callable(sp) else sp

    def state_text(self) -> str | None:
        if self.state is None:
            return None
        renderer = self.definition.state_renderer
        if renderer is not None:
            return renderer(self.state)
        return json.dumps(self.state.model_dump(mode="json"), ensure_ascii=False, indent=2)  # type: ignore[attr-defined]

    def default_request(self) -> ModelRequest:
        system = self.system_prompt()
        if self.resolved_inputs:
            parts = [f"## {name}\n{text}" for name, text in self.resolved_inputs.items()]
            system = f"{system}\n\n# Références\n\n" + "\n\n".join(parts)
        messages = entries_to_messages(self.branch)
        st = self.state_text()
        if st is not None:
            messages.append(
                UserMessage(content=[TextBlock(text=f"<etat_courant>\n{st}\n</etat_courant>")])
            )
        return ModelRequest(
            model=self.model,
            system=system,
            messages=messages,
            tools=self.tools,
            settings=dict(self.definition.model_settings),
        )


def _note(text: str) -> UserMessage:
    return UserMessage(content=[TextBlock(text=f"[Note système] {text}")])


def entries_to_messages(branch: Sequence[EntryView]) -> list[Message]:
    """Conversion par défaut de la branche active en messages neutres."""
    out: list[Message] = []
    for e in branch:
        c = e.content
        if e.type == "user":
            out.append(UserMessage(content=normalize_content(c["content"])))
        elif e.type == "assistant":
            msg = AssistantMessage.model_validate(c)
            if msg.content:
                out.append(msg)
        elif e.type == "tool_result":
            out.append(ToolResultMessage.model_validate(c))
        elif e.type == "state_edit":
            out.append(
                _note(
                    "L'utilisateur a modifié l'état manuellement (JSON Patch) : "
                    + json.dumps(c.get("patch", []), ensure_ascii=False)
                )
            )
        elif e.type == "child_result":
            out.append(
                _note(
                    f"Résultat de la session enfant « {c.get('kind')} » (version {c.get('version')}) "
                    "appliqué à l'état."
                )
            )
        elif e.type == "summary":
            out.append(_note(f"Résumé de la conversation précédente : {c.get('text', '')}"))
    return out

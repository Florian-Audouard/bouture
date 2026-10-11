"""Protocole d'adaptateur LLM (D3) : le cœur ne connaît que ce contrat."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..messages import Message, Usage


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class ModelRequest:
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ReasoningDelta:
    text: str
    signature: str | None = None


@dataclass(slots=True)
class ToolCallComplete:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class UsageReport:
    usage: Usage


@dataclass(slots=True)
class Done:
    stop_reason: str | None = None


ModelEvent = TextDelta | ReasoningDelta | ToolCallComplete | UsageReport | Done


class ModelClient(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Génère les événements d'un appel. Lève ``bouture.errors.ModelError`` en cas d'échec.

        Annuler l'itération (``CancelledError``) doit fermer la connexion au fournisseur.
        """
        ...

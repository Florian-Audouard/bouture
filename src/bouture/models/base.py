"""Protocole d'adaptateur LLM (D3) : le cœur ne connaît que ce contrat."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from ..messages import ImageBlock, Message, Usage


@dataclass(slots=True)
class CallContext:
    """Pour qui l'appel est fait (D92) : le client peut en déduire des identifiants par appel."""

    owner: str
    session_id: UUID | None
    kind: str
    turn_id: UUID | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """Octets d'une image, chargés par l'``image_loader`` de l'hôte pour un seul appel (D113)."""

    data: bytes
    media_type: str

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.base64()}"

    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclass(slots=True)
class ModelRequest:
    """``images`` : octets des ``ImageBlock`` de ``messages``, par ``ref`` (vide sans loader).

    Une image absente de ``images`` est passée au fournisseur par sa référence (URL).
    """

    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    context: CallContext | None = None
    images: dict[str, LoadedImage] = field(default_factory=dict[str, LoadedImage])


def image_url(block: ImageBlock, images: Mapping[str, LoadedImage]) -> str:
    """URL à transmettre : ``data:`` si l'image a été chargée, sinon la référence telle quelle."""
    loaded = images.get(block.ref)
    return loaded.data_url() if loaded is not None else block.ref


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
class ToolCallDelta:
    """Fragment brut des arguments JSON d'un appel d'outil en cours (D105)."""

    index: int
    id: str | None
    name: str | None
    args_delta: str


@dataclass(slots=True)
class UsageReport:
    usage: Usage


@dataclass(slots=True)
class Done:
    stop_reason: str | None = None


@dataclass(slots=True)
class Estimate:
    """Estimation d'un appel avant envoi (D93). ``cost`` en dollars, entrée seule, si connu."""

    input_tokens: int
    cost: float | None = None


ModelEvent = TextDelta | ReasoningDelta | ToolCallDelta | ToolCallComplete | UsageReport | Done


class ModelClient(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Génère les événements d'un appel. Lève ``bouture.errors.ModelError`` en cas d'échec.

        Annuler l'itération (``CancelledError``) doit fermer la connexion au fournisseur.
        ``ToolCallDelta`` est facultatif : un client peut n'émettre que ``ToolCallComplete``.
        """
        ...


# Méthode facultative d'un ``ModelClient`` (D93), détectée par ``getattr`` :
#     async def estimate(self, request: ModelRequest) -> Estimate

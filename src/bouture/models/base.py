"""Protocole d'adaptateur LLM (D3) : le cœur ne connaît que ce contrat."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from uuid import UUID

from ..messages import ImageBlock, Message, ProviderToolBlock, Usage


@dataclass(slots=True)
class CallContext:
    """Pour qui l'appel est fait (D92) : le client peut en déduire des identifiants par appel."""

    owner: str
    session_id: UUID | None
    kind: str
    turn_id: UUID | None = None


PROVIDER_TOOLS = frozenset({"web_search", "web_fetch"})
"""Outils exécutés côté fournisseur que bouture sait demander (D116)."""


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
    ``provider_tools`` : outils exécutés par le fournisseur (D116), noms de ``PROVIDER_TOOLS``.
    """

    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    context: CallContext | None = None
    images: dict[str, LoadedImage] = field(default_factory=dict[str, LoadedImage])
    provider_tools: list[str] = field(default_factory=list[str])


def image_url(block: ImageBlock, images: Mapping[str, LoadedImage]) -> str:
    """URL à transmettre : ``data:`` si l'image a été chargée, sinon la référence telle quelle."""
    loaded = images.get(block.ref)
    return loaded.data_url() if loaded is not None else block.ref


PROVIDER_RESULT_MAX = 20_000
"""Taille maximale (caractères) du résumé conservé d'un résultat d'outil fournisseur."""


def clip_result(text: str) -> str:
    if len(text) <= PROVIDER_RESULT_MAX:
        return text
    return text[:PROVIDER_RESULT_MAX] + f"\n[… tronqué, {len(text)} caractères]"


def provider_tool_text(block: ProviderToolBlock) -> str:
    """Rejeu d'un outil fournisseur passé : en texte, le fournisseur n'en garde pas trace (D118)."""
    args = json.dumps(block.args, ensure_ascii=False)
    head = f'<provider_tool id="{block.id}" name="{block.name}">{args}</provider_tool>'
    if block.result is None:
        return head
    error = ' error="true"' if block.is_error else ""
    return f"{head}\n<provider_tool_result{error}>\n{block.result}\n</provider_tool_result>"


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
class ProviderToolCall:
    """Le fournisseur a appelé un de ses outils (D116) ; bouture ne l'exécute pas."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ProviderToolResult:
    """Résultat (résumé en texte) d'un ``ProviderToolCall`` du même appel."""

    id: str
    content: str
    is_error: bool = False


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


ModelEvent = (
    TextDelta
    | ReasoningDelta
    | ToolCallDelta
    | ToolCallComplete
    | ProviderToolCall
    | ProviderToolResult
    | UsageReport
    | Done
)


class ModelClient(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Génère les événements d'un appel. Lève ``bouture.errors.ModelError`` en cas d'échec.

        Annuler l'itération (``CancelledError``) doit fermer la connexion au fournisseur.
        ``ToolCallDelta`` est facultatif : un client peut n'émettre que ``ToolCallComplete``.
        """
        ...


# Méthodes facultatives d'un ``ModelClient``, détectées par ``getattr`` :
#     async def estimate(self, request: ModelRequest) -> Estimate                  (D93)
#     def provider_tools(self, model: str) -> Collection[str]                      (D117)
# Sans ``provider_tools``, le client n'en supporte aucun.


def supported_provider_tools(client: object, model: str) -> frozenset[str]:
    """Outils fournisseur que ``client`` sait exécuter pour ``model`` (D117)."""
    fn = getattr(client, "provider_tools", None)
    if not callable(fn):
        return frozenset()
    return frozenset(cast("Collection[str]", fn(model)))

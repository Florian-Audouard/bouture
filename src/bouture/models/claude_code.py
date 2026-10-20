"""Adaptateur Claude Code (extra ``bouture[claude-code]``) : appels via un abonnement Claude.

Chaque appel lance Claude Code (Claude Agent SDK) authentifié par ``claude login`` ou par
``CLAUDE_CODE_OAUTH_TOKEN`` (``claude setup-token``). Réservé au développement et à l'usage
personnel : les conditions d'Anthropic interdisent d'offrir un abonnement claude.ai aux
utilisateurs d'un produit tiers ; en production, utiliser une clé d'API (``LiteLLMClient``).

Limites par rapport à ``LiteLLMClient`` :

- l'historique est sérialisé en transcript dans un seul message utilisateur (Claude Code
  n'accepte pas d'historique fabriqué) ; les images chargées par l'``image_loader`` y sont
  insérées en blocs base64 à leur place (entrée streaming, D115), les autres par leur
  référence ;
- les outils sont exposés en MCP ; un hook ``PreToolUse`` les « diffère » pour que Claude Code
  s'arrête au premier appel sans l'exécuter : c'est bouture qui exécute les outils ;
- outils fournisseur (D119) : ``WebSearch`` / ``WebFetch`` du CLI, autorisés par le hook et
  exécutés par Claude Code ; chacun consomme un tour, d'où ``max_turns = 1 +
  provider_tool_turns`` ; ils sont rendus comme ``ProviderToolCall`` / ``ProviderToolResult`` ;
- signatures de raisonnement non conservées, pas d'``estimate`` ; chaque appel démarre un
  processus (une à deux secondes de latence).

Mode ``persistent=True`` (D128) : dans un même tour, le processus de Claude Code est gardé entre
les étapes ; après un appel d'outil, seuls les résultats lui sont envoyés (``tool_result``
natifs). Historique natif, raisonnement signé conservé, préfixe de prompt stable pour le cache.
Toute requête qui n'est pas la suite exacte de la précédente repart en transcript.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from ..errors import ModelError
from ..messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    ProviderToolBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    message_images,
)
from .base import (
    Done,
    LoadedImage,
    ModelEvent,
    ModelRequest,
    ProviderToolCall,
    ProviderToolResult,
    ReasoningDelta,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolSpec,
    UsageReport,
    clip_result,
    provider_tool_text,
)

try:
    import claude_agent_sdk as sdk  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
    raise ImportError("installer l'extra : pip install 'bouture[claude-code]'") from exc

SERVER = "bouture"
PREFIX = f"mcp__{SERVER}__"
MODEL_PREFIX = "claude-code/"

# Outils fournisseur (D116) → outils du CLI qui les exécutent (D119).
CLI_TOOLS = {"web_search": "WebSearch", "web_fetch": "WebFetch"}
_NEUTRAL = {v: k for k, v in CLI_TOOLS.items()}

# Erreurs d'``AssistantMessage.error`` qu'un nouvel essai peut résoudre.
_TRANSIENT = {"rate_limit", "server_error"}

PROTOCOL = """\
La conversation jusqu'ici est donnée dans le message de l'utilisateur, sous forme de transcript \
entre balises <transcript>. Les appels d'outils passés et leurs résultats y figurent pour \
mémoire, de même que les recherches déjà faites (<provider_tool>). Réponds en tant \
qu'assistant au dernier tour uniquement, sans reprendre ces balises. Pour appeler un outil, \
utilise le mécanisme d'appel d'outils, jamais du texte."""

Query = Callable[..., AsyncIterator[Any]]
ClientFactory = Callable[[Any], Any]


@dataclass
class _Live:
    """Processus Claude Code gardé entre deux étapes d'un tour (D128)."""

    client: Any
    key: str
    sent: list[Message] = field(default_factory=list[Message])
    calls: list[str] = field(default_factory=list[str])
    at: float = 0.0


class ClaudeCodeClient:
    """``model`` : alias ou identifiant Claude Code (``"sonnet"``, ``"claude-opus-5-5"``),
    avec ou sans préfixe ``"claude-code/"`` ; ``"default"`` garde le modèle par défaut du CLI.

    ``use_subscription`` (défaut) masque ``ANTHROPIC_API_KEY`` au CLI pour qu'il utilise
    l'abonnement même si une clé traîne dans l'environnement. ``options`` est passé tel quel à
    ``ClaudeAgentOptions`` (``cli_path``, ``effort``, ``thinking``, ``env``…).

    ``provider_tool_turns`` (D119) : nombre d'appels ``WebSearch`` / ``WebFetch`` permis dans un
    même appel au modèle quand le kind en demande ; au-delà, Claude Code s'arrête
    (``Done("max_turns")``) avec le texte déjà produit.

    ``persistent`` (D128, désactivé par défaut) : garde le processus entre les étapes d'un même
    tour (``ClaudeSDKClient``) au lieu de relancer Claude Code avec tout le transcript. Au plus
    ``max_live`` processus gardés ; un processus inutilisé depuis ``live_ttl`` secondes est fermé
    au prochain appel. ``aclose()`` les ferme tous (appelé par ``Agents.shutdown``).
    """

    def __init__(
        self,
        *,
        use_subscription: bool = True,
        query: Query | None = None,
        provider_tool_turns: int = 8,
        persistent: bool = False,
        max_live: int = 4,
        live_ttl: float = 600.0,
        client_factory: ClientFactory | None = None,
        **options: Any,
    ) -> None:
        self.use_subscription = use_subscription
        self.provider_tool_turns = provider_tool_turns
        self.persistent = persistent
        self.max_live = max_live
        self.live_ttl = live_ttl
        self.options = options
        self._query: Query = query or sdk.query
        self._client_factory: ClientFactory = client_factory or sdk.ClaudeSDKClient
        self._live: dict[UUID, _Live] = {}
        # Répertoire vide : Claude Code n'y trouve ni CLAUDE.md ni fichiers du projet.
        self._cwd = tempfile.mkdtemp(prefix="bouture-claude-code-")

    def provider_tools(self, model: str) -> frozenset[str]:
        return frozenset(CLI_TOOLS)

    def _options(self, request: ModelRequest) -> Any:
        env: dict[str, str] = dict(self.options.get("env", {}))
        if self.use_subscription:
            env.setdefault("ANTHROPIC_API_KEY", "")
        env.setdefault("CLAUDE_AGENT_SDK_CLIENT_APP", "bouture")
        servers: dict[str, Any] = {}
        if request.tools:
            servers[SERVER] = sdk.create_sdk_mcp_server(
                SERVER, tools=[_mcp_tool(t) for t in request.tools]
            )
        system = f"{request.system}\n\n{PROTOCOL}" if request.system else PROTOCOL
        settings = {k: v for k, v in self.options.items() if k != "env"}
        settings.update(request.settings)
        fields: dict[str, Any] = {
            "model": _model(request.model),
            "system_prompt": system,
            "tools": [CLI_TOOLS[n] for n in request.provider_tools if n in CLI_TOOLS],
            "mcp_servers": servers,
            "strict_mcp_config": True,
            "setting_sources": [],
            "hooks": {"PreToolUse": [sdk.HookMatcher(hooks=[_defer])]},
            "include_partial_messages": True,
            "verbatim_prompts": True,
            "max_turns": 1 + (self.provider_tool_turns if request.provider_tools else 0),
            "cwd": self._cwd,
            "env": env,
            **settings,
        }
        return sdk.ClaudeAgentOptions(**fields)

    async def aclose(self) -> None:
        """Ferme les processus gardés (mode ``persistent``)."""
        live, self._live = list(self._live.values()), {}
        for lv in live:
            await _disconnect(lv.client)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        options = self._options(request)
        turn = request.context.turn_id if self.persistent and request.context else None
        if turn is None:
            async with aclosing(
                cast("AsyncGenerator[Any]", self._query(prompt=_prompt(request), options=options))
            ) as messages:
                async for ev in self._events(messages, request):
                    yield ev
            return
        await self._sweep()
        live, prompt = await self._resume(turn, request)
        if live is None:
            live = _Live(self._client_factory(options), _key(request))
            await live.client.connect()
            prompt = _prompt(request)
        kept = False
        try:
            await live.client.query(prompt)
            calls: list[str] = []
            async with aclosing(
                cast("AsyncGenerator[Any]", live.client.receive_response())
            ) as messages:
                async for ev in self._events(messages, request):
                    if isinstance(ev, ToolCallComplete):
                        calls.append(ev.id)
                    elif isinstance(ev, Done) and calls:
                        # Gardé avant le dernier événement : le lecteur peut s'arrêter à Done.
                        live.sent, live.calls = list(request.messages), calls
                        live.at = time.monotonic()
                        self._live[turn] = live
                        kept = True
                        await self._evict()
                    yield ev
        finally:
            if not kept:
                await _disconnect(live.client)

    async def _resume(
        self, turn: UUID, request: ModelRequest
    ) -> tuple[_Live | None, str | AsyncIterator[dict[str, Any]]]:
        live = self._live.pop(turn, None)
        if live is None:
            return None, ""
        content = _continuation(live, request) if live.key == _key(request) else None
        if content is None:
            await _disconnect(live.client)
            return None, ""
        return live, _single(content)

    async def _sweep(self) -> None:
        limit = time.monotonic() - self.live_ttl
        for turn, lv in list(self._live.items()):
            if lv.at < limit:
                del self._live[turn]
                await _disconnect(lv.client)

    async def _evict(self) -> None:
        while len(self._live) > self.max_live:
            oldest = next(iter(self._live))
            await _disconnect(self._live.pop(oldest).client)

    async def _events(
        self, messages: AsyncIterator[Any], request: ModelRequest
    ) -> AsyncIterator[ModelEvent]:
        usage = Usage(model=request.model)
        blocks: dict[int, dict[str, Any]] = {}
        calls: dict[str, ToolCallComplete] = {}
        provided: set[str] = set()  # ids des outils fournisseur déjà émis
        meter = {"base": 0}
        stop: str | None = None
        try:
            async for msg in messages:
                if getattr(msg, "parent_tool_use_id", None):
                    continue
                if isinstance(msg, sdk.StreamEvent):
                    for ev in _stream_event(msg.event, blocks, usage, meter):
                        yield ev
                elif isinstance(msg, sdk.AssistantMessage):
                    if msg.error:
                        raise ModelError(
                            f"Claude Code : {msg.error}",
                            transient=msg.error in _TRANSIENT,
                            code=str(msg.error),
                        )
                    for b in msg.content:
                        if isinstance(b, sdk.ToolUseBlock) and b.name in _NEUTRAL:
                            if b.id not in provided:
                                provided.add(b.id)
                                yield ProviderToolCall(b.id, _NEUTRAL[b.name], b.input)
                        elif isinstance(b, sdk.ServerToolUseBlock):
                            if b.name in CLI_TOOLS and b.id not in provided:
                                provided.add(b.id)
                                yield ProviderToolCall(b.id, b.name, b.input)
                        elif isinstance(b, sdk.ServerToolResultBlock):
                            if b.tool_use_id in provided:
                                text = json.dumps(b.content, ensure_ascii=False)
                                yield ProviderToolResult(b.tool_use_id, clip_result(text))
                        elif isinstance(b, sdk.ToolUseBlock):
                            calls[b.id] = ToolCallComplete(b.id, _strip(b.name), b.input)
                elif isinstance(msg, sdk.UserMessage) and not isinstance(msg.content, str):
                    for b in msg.content:
                        if isinstance(b, sdk.ToolResultBlock) and b.tool_use_id in provided:
                            yield ProviderToolResult(
                                b.tool_use_id, _result_text(b.content), bool(b.is_error)
                            )
                elif isinstance(msg, sdk.ResultMessage):
                    stop = msg.stop_reason
                    if msg.deferred_tool_use is not None:
                        d = msg.deferred_tool_use
                        calls.setdefault(d.id, ToolCallComplete(d.id, _strip(d.name), d.input))
                    elif msg.subtype == "error_max_turns" and request.provider_tools:
                        stop = "max_turns"  # budget d'outils fournisseur épuisé (D119)
                    elif msg.is_error and not calls:
                        raise ModelError(
                            f"Claude Code : {msg.subtype}",
                            transient=msg.api_error_status in (429, 500, 502, 503, 529),
                            code=msg.subtype,
                        )
                    break
        except ModelError:
            raise
        except sdk.ClaudeSDKError as exc:
            raise ModelError(type(exc).__name__, code=type(exc).__name__) from exc
        yield UsageReport(usage)
        for call in calls.values():
            yield call
        yield Done("tool_calls" if calls else (stop or "stop"))


def _model(model: str) -> str | None:
    model = model.removeprefix(MODEL_PREFIX)
    return None if model in ("", "default") else model


def _strip(name: str) -> str:
    return name.removeprefix(PREFIX)


def _mcp_tool(spec: ToolSpec) -> Any:
    async def handler(_args: dict[str, Any]) -> dict[str, Any]:  # jamais appelé : différé
        return {"content": [{"type": "text", "text": "deferred"}], "is_error": True}

    return sdk.SdkMcpTool(
        name=spec.name, description=spec.description, input_schema=spec.parameters, handler=handler
    )


async def _defer(input: Any, tool_use_id: str | None, context: Any) -> Any:
    """Arrête Claude Code avant l'exécution : l'appel revient à bouture.

    Seuls ``WebSearch`` / ``WebFetch`` (outils fournisseur, D119) sont exécutés par Claude Code.
    """
    name = cast("dict[str, Any]", input).get("tool_name") if isinstance(input, dict) else None
    decision = "allow" if name in _NEUTRAL else "defer"
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision}}


def _result_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return clip_result(content)
    parts = [
        c.get("text", "") if c.get("type") == "text" else f"[{c.get('type')}]" for c in content
    ]
    return clip_result("\n".join(p for p in parts if p))


def _stream_event(
    event: dict[str, Any],
    blocks: dict[int, dict[str, Any]],
    usage: Usage,
    meter: dict[str, int] | None = None,
) -> list[ModelEvent]:
    """Événement brut de l'API Messages → événements bouture (et cumul de l'usage).

    Un appel peut contenir plusieurs messages de l'API (outils fournisseur, D119) : l'usage
    s'additionne d'un message à l'autre, ``meter["base"]`` retient la sortie des précédents.
    Les blocs ``tool_use`` d'un outil fournisseur ne produisent pas de ``ToolCallDelta``.
    """
    meter = meter if meter is not None else {"base": 0}
    kind = event.get("type")
    if kind == "message_start":
        meter["base"] = usage.output_tokens
        u = event.get("message", {}).get("usage") or {}
        usage.input_tokens += u.get("input_tokens") or 0
        usage.cache_read += u.get("cache_read_input_tokens") or 0
        usage.cache_write += u.get("cache_creation_input_tokens") or 0
        usage.output_tokens += u.get("output_tokens") or 0
    elif kind == "message_delta":
        u = event.get("usage") or {}
        out = meter["base"] + (u.get("output_tokens") or 0)
        usage.output_tokens = max(usage.output_tokens, out)
    elif kind == "content_block_start":
        block = event.get("content_block") or {}
        blocks[event.get("index", 0)] = block
        if block.get("type") == "tool_use" and block.get("name") not in _NEUTRAL:
            return [
                ToolCallDelta(event.get("index", 0), block.get("id"), _strip(block["name"]), "")
            ]
    elif kind == "content_block_delta":
        index = event.get("index", 0)
        delta = event.get("delta") or {}
        dt = delta.get("type")
        if dt == "text_delta" and delta.get("text"):
            return [TextDelta(delta["text"])]
        if dt == "thinking_delta" and delta.get("thinking"):
            return [ReasoningDelta(delta["thinking"])]
        provider = blocks.get(index, {}).get("name") in _NEUTRAL
        if dt == "input_json_delta" and delta.get("partial_json") and not provider:
            return [ToolCallDelta(index, None, None, delta["partial_json"])]
    return []


async def _single(content: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Entrée streaming d'un seul message : seule forme qui accepte des blocs image."""
    yield {
        "type": "user",
        "session_id": "",
        "parent_tool_use_id": None,
        "message": {"role": "user", "content": content},
    }


Segment = str | ImageBlock


def _body(blocks: Sequence[TextBlock | ImageBlock]) -> list[Segment]:
    """Texte d'un contenu ; chaque image laisse sa référence lisible, suivie du bloc lui-même."""
    out: list[Segment] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            out.append(b.text)
        else:
            out += [f"[image : {b.ref}]", b]
    return out


def _segments(messages: list[Message]) -> list[Segment]:
    parts: list[Segment] = ["<transcript>\n"]
    for m in messages:
        if isinstance(m, UserMessage):
            parts += ['<turn role="user">\n', *_body(m.content), "\n</turn>\n"]
        elif isinstance(m, AssistantMessage):
            lines: list[str] = []
            for b in m.content:
                if isinstance(b, TextBlock) and b.text:
                    lines.append(b.text)
                elif isinstance(b, ToolCallBlock):
                    args = json.dumps(b.args, ensure_ascii=False)
                    lines.append(f'<tool_call id="{b.id}" name="{b.name}">{args}</tool_call>')
                elif isinstance(b, ProviderToolBlock):
                    lines.append(provider_tool_text(b))
                elif isinstance(b, ReasoningBlock):
                    continue
            if lines:
                parts.append('<turn role="assistant">\n' + "\n".join(lines) + "\n</turn>\n")
        else:
            error = ' error="true"' if m.is_error else ""
            body = [m.content] if isinstance(m.content, str) else _body(m.content)
            parts += [
                f'<tool_result id="{m.tool_call_id}" name="{m.name}"{error}>\n',
                *body,
                "\n</tool_result>\n",
            ]
    parts.append("</transcript>")
    return parts


def to_transcript(messages: list[Message]) -> str:
    """Historique neutre → transcript texte (un seul message utilisateur pour Claude Code)."""
    return "".join(p for p in _segments(messages) if isinstance(p, str))


def to_transcript_blocks(
    messages: list[Message], images: Mapping[str, LoadedImage]
) -> list[dict[str, Any]]:
    """Transcript en blocs de contenu : texte, et image base64 là où elle a été chargée."""
    out: list[dict[str, Any]] = []
    text = ""
    for p in _segments(messages):
        if isinstance(p, str):
            text += p
            continue
        loaded = images.get(p.ref)
        if loaded is None:
            continue
        if text:
            out.append({"type": "text", "text": text})
            text = ""
        source = {"type": "base64", "media_type": loaded.media_type, "data": loaded.base64()}
        out.append({"type": "image", "source": source})
    if text:
        out.append({"type": "text", "text": text})
    return out


def _prompt(request: ModelRequest) -> str | AsyncIterator[dict[str, Any]]:
    """Requête entière en transcript : texte, ou blocs quand une image est chargée."""
    if any(b.ref in request.images for b in message_images(request.messages)):
        return _single(to_transcript_blocks(request.messages, request.images))
    return to_transcript(request.messages)


def _key(request: ModelRequest) -> str:
    """Ce qui doit rester identique pour continuer un processus gardé (D128)."""
    return json.dumps(
        [
            request.model,
            request.system,
            [(t.name, t.description, t.parameters) for t in request.tools],
            request.settings,
            sorted(request.provider_tools),
        ],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _continuation(live: _Live, request: ModelRequest) -> list[dict[str, Any]] | None:
    """Contenu à envoyer au processus gardé, ou ``None`` si la requête n'en est pas la suite.

    La suite attendue : les messages déjà envoyés (les messages utilisateur de fin, comme
    ``<etat_courant>``, peuvent être remplacés), la réponse de l'assistant avec exactement les
    appels différés, leurs résultats, puis d'éventuels messages utilisateur.
    """
    sent, cur = live.sent, request.messages
    p = 0
    while p < min(len(sent), len(cur)) and sent[p] == cur[p]:
        p += 1
    if not all(isinstance(m, UserMessage) for m in sent[p:]) or p >= len(cur):
        return None
    reply = cur[p]
    if not isinstance(reply, AssistantMessage):
        return None
    if [b.id for b in reply.content if isinstance(b, ToolCallBlock)] != live.calls:
        return None
    results = cur[p + 1 : p + 1 + len(live.calls)]
    if len(results) != len(live.calls) or not all(
        isinstance(m, ToolResultMessage) for m in results
    ):
        return None
    done = cast("list[ToolResultMessage]", results)
    if {m.tool_call_id for m in done} != set(live.calls):
        return None
    rest = cur[p + 1 + len(live.calls) :]
    if not all(isinstance(m, UserMessage) for m in rest):
        return None
    out: list[dict[str, Any]] = []
    for m in done:
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": m.tool_call_id}
        block["content"] = (
            m.content if isinstance(m.content, str) else _native(m.content, request.images)
        )
        if m.is_error:
            block["is_error"] = True
        out.append(block)
    for m in cast("list[UserMessage]", rest):
        out += _native(m.content, request.images)
    return out


def _native(
    blocks: Sequence[TextBlock | ImageBlock], images: Mapping[str, LoadedImage]
) -> list[dict[str, Any]]:
    """Contenu bouture → blocs de l'API Messages (image chargée en base64, sinon sa référence)."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            if b.text:
                out.append({"type": "text", "text": b.text})
            continue
        loaded = images.get(b.ref)
        if loaded is None:
            out.append({"type": "text", "text": f"[image : {b.ref}]"})
        else:
            source = {"type": "base64", "media_type": loaded.media_type, "data": loaded.base64()}
            out.append({"type": "image", "source": source})
    return out


async def _disconnect(client: Any) -> None:
    with contextlib.suppress(Exception):
        await client.disconnect()

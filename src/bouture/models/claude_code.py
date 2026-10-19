"""Adaptateur Claude Code (extra ``bouture[claude-code]``) : appels via un abonnement Claude.

Chaque appel lance Claude Code (Claude Agent SDK) authentifié par ``claude login`` ou par
``CLAUDE_CODE_OAUTH_TOKEN`` (``claude setup-token``). Réservé au développement et à l'usage
personnel : les conditions d'Anthropic interdisent d'offrir un abonnement claude.ai aux
utilisateurs d'un produit tiers ; en production, utiliser une clé d'API (``LiteLLMClient``).

Limites par rapport à ``LiteLLMClient`` :

- l'historique est sérialisé en transcript dans un seul message utilisateur (Claude Code
  n'accepte pas d'historique fabriqué) ; les images n'y figurent que par leur référence ;
- les outils sont exposés en MCP ; un hook ``PreToolUse`` les « diffère » pour que Claude Code
  s'arrête au premier appel sans l'exécuter : c'est bouture qui exécute les outils ;
- signatures de raisonnement non conservées, pas d'``estimate`` ; chaque appel démarre un
  processus (une à deux secondes de latence).
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from typing import Any, cast

from ..errors import ModelError
from ..messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    Usage,
    UserMessage,
)
from .base import (
    Done,
    ModelEvent,
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolSpec,
    UsageReport,
)

try:
    import claude_agent_sdk as sdk  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
    raise ImportError("installer l'extra : pip install 'bouture[claude-code]'") from exc

SERVER = "bouture"
PREFIX = f"mcp__{SERVER}__"
MODEL_PREFIX = "claude-code/"

# Erreurs d'``AssistantMessage.error`` qu'un nouvel essai peut résoudre.
_TRANSIENT = {"rate_limit", "server_error"}

PROTOCOL = """\
La conversation jusqu'ici est donnée dans le message de l'utilisateur, sous forme de transcript \
entre balises <transcript>. Les appels d'outils passés et leurs résultats y figurent pour \
mémoire. Réponds en tant qu'assistant au dernier tour uniquement, sans reprendre ces balises. \
Pour appeler un outil, utilise le mécanisme d'appel d'outils, jamais du texte."""

Query = Callable[..., AsyncIterator[Any]]


class ClaudeCodeClient:
    """``model`` : alias ou identifiant Claude Code (``"sonnet"``, ``"claude-opus-5-5"``),
    avec ou sans préfixe ``"claude-code/"`` ; ``"default"`` garde le modèle par défaut du CLI.

    ``use_subscription`` (défaut) masque ``ANTHROPIC_API_KEY`` au CLI pour qu'il utilise
    l'abonnement même si une clé traîne dans l'environnement. ``options`` est passé tel quel à
    ``ClaudeAgentOptions`` (``cli_path``, ``effort``, ``thinking``, ``env``…).
    """

    def __init__(
        self,
        *,
        use_subscription: bool = True,
        query: Query | None = None,
        **options: Any,
    ) -> None:
        self.use_subscription = use_subscription
        self.options = options
        self._query: Query = query or sdk.query
        # Répertoire vide : Claude Code n'y trouve ni CLAUDE.md ni fichiers du projet.
        self._cwd = tempfile.mkdtemp(prefix="bouture-claude-code-")

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
            "tools": [],
            "mcp_servers": servers,
            "strict_mcp_config": True,
            "setting_sources": [],
            "hooks": {"PreToolUse": [sdk.HookMatcher(hooks=[_defer])]},
            "include_partial_messages": True,
            "verbatim_prompts": True,
            "max_turns": 1,
            "cwd": self._cwd,
            "env": env,
            **settings,
        }
        return sdk.ClaudeAgentOptions(**fields)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        options = self._options(request)
        prompt = to_transcript(request.messages)
        usage = Usage(model=request.model)
        blocks: dict[int, dict[str, Any]] = {}
        calls: dict[str, ToolCallComplete] = {}
        stop: str | None = None
        try:
            async with aclosing(
                cast("AsyncGenerator[Any]", self._query(prompt=prompt, options=options))
            ) as messages:
                async for msg in messages:
                    if getattr(msg, "parent_tool_use_id", None):
                        continue
                    if isinstance(msg, sdk.StreamEvent):
                        for ev in _stream_event(msg.event, blocks, usage):
                            yield ev
                    elif isinstance(msg, sdk.AssistantMessage):
                        if msg.error:
                            raise ModelError(
                                f"Claude Code : {msg.error}",
                                transient=msg.error in _TRANSIENT,
                                code=str(msg.error),
                            )
                        for b in msg.content:
                            if isinstance(b, sdk.ToolUseBlock):
                                calls[b.id] = ToolCallComplete(b.id, _strip(b.name), b.input)
                    elif isinstance(msg, sdk.ResultMessage):
                        stop = msg.stop_reason
                        if msg.deferred_tool_use is not None:
                            d = msg.deferred_tool_use
                            calls.setdefault(d.id, ToolCallComplete(d.id, _strip(d.name), d.input))
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
    """Arrête Claude Code avant l'exécution : l'appel revient à bouture."""
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}}


def _stream_event(
    event: dict[str, Any], blocks: dict[int, dict[str, Any]], usage: Usage
) -> list[ModelEvent]:
    """Événement brut de l'API Messages → événements bouture (et cumul de l'usage)."""
    kind = event.get("type")
    if kind == "message_start":
        u = event.get("message", {}).get("usage") or {}
        usage.input_tokens += u.get("input_tokens") or 0
        usage.cache_read += u.get("cache_read_input_tokens") or 0
        usage.cache_write += u.get("cache_creation_input_tokens") or 0
        usage.output_tokens += u.get("output_tokens") or 0
    elif kind == "message_delta":
        u = event.get("usage") or {}
        usage.output_tokens = max(usage.output_tokens, u.get("output_tokens") or 0)
    elif kind == "content_block_start":
        block = event.get("content_block") or {}
        blocks[event.get("index", 0)] = block
        if block.get("type") == "tool_use":
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
        if dt == "input_json_delta" and delta.get("partial_json"):
            return [ToolCallDelta(index, None, None, delta["partial_json"])]
    return []


def to_transcript(messages: list[Message]) -> str:
    """Historique neutre → transcript texte (un seul message utilisateur pour Claude Code)."""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, UserMessage):
            body = "".join(
                f"[image : {b.ref}]" if isinstance(b, ImageBlock) else b.text for b in m.content
            )
            parts.append(f'<turn role="user">\n{body}\n</turn>')
        elif isinstance(m, AssistantMessage):
            lines: list[str] = []
            for b in m.content:
                if isinstance(b, TextBlock) and b.text:
                    lines.append(b.text)
                elif isinstance(b, ToolCallBlock):
                    args = json.dumps(b.args, ensure_ascii=False)
                    lines.append(f'<tool_call id="{b.id}" name="{b.name}">{args}</tool_call>')
                elif isinstance(b, ReasoningBlock):
                    continue
            if lines:
                parts.append('<turn role="assistant">\n' + "\n".join(lines) + "\n</turn>")
        else:
            error = ' error="true"' if m.is_error else ""
            parts.append(
                f'<tool_result id="{m.tool_call_id}" name="{m.name}"{error}>\n'
                f"{m.content}\n</tool_result>"
            )
    return "<transcript>\n" + "\n".join(parts) + "\n</transcript>"

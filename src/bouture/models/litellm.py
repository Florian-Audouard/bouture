"""Adaptateur LiteLLM (extra ``bouture[litellm]``)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from typing import Any

from ..errors import ModelError
from ..messages import (
    AssistantMessage,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    Usage,
    UserMessage,
)
from .base import (
    Done,
    Estimate,
    ModelEvent,
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    UsageReport,
)

Credentials = Callable[[ModelRequest], Awaitable[Mapping[str, Any]]]

try:
    import litellm  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
    raise ImportError("installer l'extra : pip install 'bouture[litellm]'") from exc

_TRANSIENT = tuple(
    getattr(litellm, name)
    for name in (
        "RateLimitError",
        "ServiceUnavailableError",
        "APIConnectionError",
        "Timeout",
        "InternalServerError",
    )
    if hasattr(litellm, name)
)


class LiteLLMClient:
    """``model`` au format LiteLLM (``"anthropic/claude-sonnet-5-5"``).

    ``api_keys`` est indexé par préfixe de fournisseur ; à défaut, LiteLLM lit les variables
    d'environnement standard.

    ``credentials`` (D92) : ``async (ModelRequest) -> dict`` appelé à chaque appel, avec
    ``request.context`` (owner, session, kind). Les clés renvoyées (``api_key``,
    ``api_base``, ``extra_headers``…) sont passées à LiteLLM pour ce seul appel : rien n'est
    stocké ni journalisé.
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        *,
        credentials: Credentials | None = None,
        **extra: Any,
    ) -> None:
        self.api_keys = api_keys or {}
        self.credentials = credentials
        self.extra = extra

    def _key(self, model: str) -> str | None:
        return self.api_keys.get(model.split("/", 1)[0])

    async def estimate(self, request: ModelRequest) -> Estimate:
        """Tokens d'entrée (tokenizer LiteLLM) et coût d'entrée si le modèle est tarifé (D93)."""
        messages = to_openai_messages(request.system, request.messages)
        try:
            tokens = litellm.token_counter(
                model=request.model,
                messages=messages,
                tools=_tools(request),  # pyright: ignore[reportArgumentType]
            )
        except Exception as exc:
            raise ModelError(_safe(exc), code=type(exc).__name__) from exc
        cost: float | None
        try:
            cost, _ = litellm.cost_per_token(
                model=request.model, prompt_tokens=tokens, completion_tokens=0
            )
        except Exception:
            cost = None  # modèle absent de la table de prix de LiteLLM
        return Estimate(input_tokens=tokens, cost=cost)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": to_openai_messages(request.system, request.messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.extra,
            **request.settings,
        }
        if tools := _tools(request):
            kwargs["tools"] = tools
        if key := self._key(request.model):
            kwargs["api_key"] = key
        if self.credentials is not None:
            try:
                kwargs.update(await self.credentials(request))
            except ModelError:
                raise
            except Exception as exc:
                raise ModelError(_safe(exc), code="credentials_error") from exc
        try:
            response = await litellm.acompletion(**kwargs)
        except _TRANSIENT as exc:
            raise ModelError(_safe(exc), transient=True, code=type(exc).__name__) from exc
        except Exception as exc:
            raise ModelError(_safe(exc), code=type(exc).__name__) from exc

        calls: dict[int, dict[str, Any]] = {}
        finish: str | None = None
        try:
            async with aclosing(response) as chunks:  # type: ignore[arg-type]
                async for chunk in chunks:
                    if getattr(chunk, "usage", None):
                        yield UsageReport(_usage(chunk.usage, request.model))
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if getattr(delta, "reasoning_content", None):
                        yield ReasoningDelta(delta.reasoning_content)
                    for tb in getattr(delta, "thinking_blocks", None) or []:
                        sig = tb.get("signature") if isinstance(tb, dict) else None
                        if sig:
                            yield ReasoningDelta("", signature=sig)
                    if delta.content:
                        yield TextDelta(delta.content)
                    for tc in getattr(delta, "tool_calls", None) or []:
                        slot = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                            yield ToolCallDelta(
                                tc.index,
                                tc.id or None,
                                tc.function.name or None,
                                tc.function.arguments,
                            )
                    if choice.finish_reason:
                        finish = choice.finish_reason
        except _TRANSIENT as exc:
            raise ModelError(_safe(exc), transient=True, code=type(exc).__name__) from exc
        except ModelError:
            raise
        except Exception as exc:
            if type(exc).__module__.startswith("litellm"):
                raise ModelError(_safe(exc), code=type(exc).__name__) from exc
            raise
        for index in sorted(calls):
            slot = calls[index]
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": slot["args"]}
            yield ToolCallComplete(id=slot["id"] or f"call_{index}", name=slot["name"], args=args)
        yield Done(finish)


def _tools(request: ModelRequest) -> list[dict[str, Any]] | None:
    if not request.tools:
        return None
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in request.tools
    ]


def _safe(exc: Exception) -> str:
    """Message assaini : pas de corps de réponse brut ni de clé (D64)."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")


def _usage(u: Any, model: str) -> Usage:
    details = getattr(u, "prompt_tokens_details", None)
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
        model=model,
    )


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Format neutre → format OpenAI (compris par LiteLLM pour tous les fournisseurs).

    Les messages utilisateur consécutifs sont fusionnés.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if isinstance(m, UserMessage):
            parts: list[dict[str, Any]] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    parts.append({"type": "text", "text": b.text})
                else:
                    parts.append({"type": "image_url", "image_url": {"url": b.ref}})
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].extend(parts)
            else:
                out.append({"role": "user", "content": parts})
        elif isinstance(m, AssistantMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            thinking = [
                {"type": "thinking", "thinking": b.text, "signature": b.signature}
                for b in m.content
                if isinstance(b, ReasoningBlock) and b.signature
            ]
            if thinking:
                msg["thinking_blocks"] = thinking
            tcs = [b for b in m.content if isinstance(b, ToolCallBlock)]
            if tcs:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in tcs
                ]
            if msg["content"] is None and not tcs:
                continue
            out.append(msg)
        else:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out

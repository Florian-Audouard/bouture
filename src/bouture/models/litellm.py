"""Adaptateur LiteLLM (extra ``bouture[litellm]``)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
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
    ModelEvent,
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolCallComplete,
    UsageReport,
)

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
    """

    def __init__(self, api_keys: dict[str, str] | None = None, **extra: Any) -> None:
        self.api_keys = api_keys or {}
        self.extra = extra

    def _key(self, model: str) -> str | None:
        return self.api_keys.get(model.split("/", 1)[0])

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": to_openai_messages(request.system, request.messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.extra,
            **request.settings,
        }
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        if key := self._key(request.model):
            kwargs["api_key"] = key
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

"""Adaptateur LiteLLM (extra ``bouture[litellm]``)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from typing import Any, Literal

from ..errors import ModelError
from ..messages import (
    ImageBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .base import (
    Done,
    Estimate,
    LoadedImage,
    ModelEvent,
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    UsageReport,
    image_url,
)

ToolImages = Literal["auto", "inline", "user"]

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

    ``tool_images`` (D114) : où placer les images renvoyées par un outil. ``"inline"`` dans le
    ``tool_result`` (Anthropic l'accepte) ; ``"user"`` dans un message utilisateur qui suit
    les résultats d'outils (OpenAI n'accepte que du texte dans un message ``tool``) ;
    ``"auto"`` (défaut) choisit ``"inline"`` pour le fournisseur ``anthropic``, ``"user"``
    sinon.
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        *,
        credentials: Credentials | None = None,
        tool_images: ToolImages = "auto",
        **extra: Any,
    ) -> None:
        self.api_keys = api_keys or {}
        self.credentials = credentials
        self.tool_images: ToolImages = tool_images
        self.extra = extra

    def _messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        mode = self.tool_images
        if mode == "auto":
            mode = "inline" if _provider(request.model) == "anthropic" else "user"
        return to_openai_messages(
            request.system, request.messages, images=request.images, tool_images=mode
        )

    def _key(self, model: str) -> str | None:
        return self.api_keys.get(model.split("/", 1)[0])

    async def estimate(self, request: ModelRequest) -> Estimate:
        """Tokens d'entrée (tokenizer LiteLLM) et coût d'entrée si le modèle est tarifé (D93)."""
        messages = self._messages(request)
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
            "messages": self._messages(request),
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


def _provider(model: str) -> str | None:
    try:
        return litellm.get_llm_provider(model)[1]
    except Exception:
        return None


def to_openai_messages(
    system: str,
    messages: list[Message],
    *,
    images: Mapping[str, LoadedImage] | None = None,
    tool_images: Literal["inline", "user"] = "user",
) -> list[dict[str, Any]]:
    """Format neutre → format OpenAI (compris par LiteLLM pour tous les fournisseurs).

    Les messages utilisateur consécutifs sont fusionnés. Une image chargée (``images``) part
    en URL ``data:`` ; sinon par sa référence. Images de résultats d'outils : voir
    ``LiteLLMClient.tool_images`` (D114).
    """
    loaded = images or {}
    out: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []  # images d'outils, mode "user"
    count = 0

    def add_user(parts: list[dict[str, Any]]) -> None:
        if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
            out[-1]["content"].extend(parts)
        else:
            out.append({"role": "user", "content": parts})

    def image_part(b: ImageBlock) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": image_url(b, loaded)}}

    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if deferred and not isinstance(m, ToolResultMessage):
            add_user(deferred)
            deferred = []
        if isinstance(m, UserMessage):
            add_user(
                [
                    {"type": "text", "text": b.text} if isinstance(b, TextBlock) else image_part(b)
                    for b in m.content
                ]
            )
        elif isinstance(m, ToolResultMessage):
            if isinstance(m.content, str) or not m.images:
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text})
            elif tool_images == "inline":
                parts = [
                    {"type": "text", "text": b.text} if isinstance(b, TextBlock) else image_part(b)
                    for b in m.content
                ]
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": parts})
            else:
                texts: list[str] = []
                for b in m.content:
                    if isinstance(b, TextBlock):
                        texts.append(b.text)
                        continue
                    count += 1
                    texts.append(f"[image {count} : {b.ref}, jointe au message suivant]")
                    label = f"Image {count} renvoyée par l'outil {m.name} ({m.tool_call_id}) :"
                    deferred += [{"type": "text", "text": label}, image_part(b)]
                content = "\n".join(t for t in texts if t)
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
        else:
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
    if deferred:
        add_user(deferred)
    return out

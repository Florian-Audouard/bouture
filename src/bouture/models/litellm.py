"""Adaptateur LiteLLM (extra ``bouture[litellm]``)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from typing import Any, Literal, cast

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
)
from .base import (
    Done,
    Estimate,
    LoadedImage,
    ModelEvent,
    ModelRequest,
    ProviderToolCall,
    ProviderToolResult,
    ReasoningDelta,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    UsageReport,
    clip_result,
    image_url,
    provider_tool_text,
)

ToolImages = Literal["auto", "inline", "user"]

# Outils fournisseur (D116) → outils hébergés d'Anthropic, transmis tels quels par LiteLLM (D120).
HOSTED_TOOLS = {"web_search": "web_search_20250305", "web_fetch": "web_fetch_20250910"}
_SERVER_ID = "srvtoolu_"

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

    Outils fournisseur (D120) : ``web_search`` / ``web_fetch`` pour le fournisseur
    ``anthropic`` seulement (outils hébergés, exécutés par l'API dans le même appel).
    ``provider_tool_options`` complète leur déclaration (``{"web_search": {"max_uses": 3}}``).
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        *,
        credentials: Credentials | None = None,
        tool_images: ToolImages = "auto",
        provider_tool_options: Mapping[str, Mapping[str, Any]] | None = None,
        **extra: Any,
    ) -> None:
        self.api_keys = api_keys or {}
        self.credentials = credentials
        self.tool_images: ToolImages = tool_images
        self.provider_tool_options = {k: dict(v) for k, v in (provider_tool_options or {}).items()}
        self.extra = extra

    def _messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        mode = self.tool_images
        if mode == "auto":
            mode = "inline" if _provider(request.model) == "anthropic" else "user"
        return to_openai_messages(
            request.system, request.messages, images=request.images, tool_images=mode
        )

    def provider_tools(self, model: str) -> frozenset[str]:
        return frozenset(HOSTED_TOOLS) if _provider(model) == "anthropic" else frozenset()

    def _hosted(self, request: ModelRequest) -> list[dict[str, Any]]:
        return [
            {
                "type": HOSTED_TOOLS[n],
                "name": n,
                **self.provider_tool_options.get(n, {}),
            }
            for n in request.provider_tools
            if n in HOSTED_TOOLS
        ]

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
        if tools := [*(_tools(request) or []), *self._hosted(request)]:
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
        provided: set[str] = set()  # outils fournisseur déjà émis
        answered: set[str] = set()
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
                        slot = calls.setdefault(
                            tc.index, {"id": None, "name": "", "args": "", "server": False}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                            slot["server"] = str(tc.id).startswith(_SERVER_ID)
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                            if slot["server"]:
                                continue  # exécuté par le fournisseur (D120)
                            yield ToolCallDelta(
                                tc.index,
                                tc.id or None,
                                tc.function.name or None,
                                tc.function.arguments,
                            )
                    for block in _server_results(chunk, delta):
                        call_id = str(block.get("tool_use_id", ""))
                        if not call_id or call_id in answered:
                            continue
                        answered.add(call_id)
                        slot = next((c for c in calls.values() if c["id"] == call_id), None)
                        if slot is not None and call_id not in provided:
                            provided.add(call_id)
                            yield ProviderToolCall(call_id, slot["name"], _args(slot["args"]))
                        text, is_error = summarize_server_result(block)
                        yield ProviderToolResult(call_id, text, is_error)
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
            if slot["server"]:
                if slot["id"] not in provided:  # appel sans résultat (interrompu, pause_turn)
                    yield ProviderToolCall(slot["id"], slot["name"], _args(slot["args"]))
                continue
            args = _args(slot["args"])
            yield ToolCallComplete(id=slot["id"] or f"call_{index}", name=slot["name"], args=args)
        yield Done(finish)


def _args(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {"_raw": raw}


def _server_results(chunk: Any, delta: Any) -> list[dict[str, Any]]:
    """Blocs ``*_tool_result`` des outils hébergés, cumulés par LiteLLM au fil du flux."""
    for holder in (delta, chunk):
        fields = getattr(holder, "provider_specific_fields", None)
        if isinstance(fields, dict):
            results = cast("dict[str, Any]", fields).get("web_search_results")
            if isinstance(results, list):
                return [b for b in cast("list[Any]", results) if isinstance(b, dict)]
    return []


def summarize_server_result(block: Mapping[str, Any]) -> tuple[str, bool]:
    """Résultat d'un outil hébergé Anthropic → (texte, erreur ?) (D120).

    Recherche : titres et URL ; lecture : URL, titre et texte borné. Le contenu chiffré de
    l'API n'est pas conservé (le rejeu est en texte, D118).
    """
    content: Any = block.get("content")
    if isinstance(content, dict):
        c = cast("dict[str, Any]", content)
        ctype = str(c.get("type", ""))
        if ctype.endswith("_error"):
            return f"erreur : {c.get('error_code', 'inconnue')}", True
        if ctype == "web_fetch_result":
            doc = cast("dict[str, Any]", c.get("content") or {})
            source = cast("dict[str, Any]", doc.get("source") or {})
            lines = [str(c.get("url", ""))]
            if doc.get("title"):
                lines.append(str(doc["title"]))
            if source.get("type") == "text":
                lines += ["", str(source.get("data", ""))]
            elif source:
                lines.append(f"[document {source.get('media_type', source.get('type'))}]")
            return clip_result("\n".join(lines)), False
        return clip_result(json.dumps(c, ensure_ascii=False)), False
    if isinstance(content, list):
        lines = [
            f"- {r.get('title', '')} — {r.get('url', '')}"
            for r in cast("list[dict[str, Any]]", content)
            if isinstance(r, dict)  # pyright: ignore[reportUnnecessaryIsInstance]
        ]
        return clip_result("\n".join(lines) or "aucun résultat"), False
    return "", False


def _assistant_text(m: AssistantMessage) -> str:
    """Texte d'un message assistant ; les outils fournisseur passés y sont rejoués (D118)."""
    if not m.provider_tools:
        return m.text
    parts = [
        provider_tool_text(b) if isinstance(b, ProviderToolBlock) else b.text
        for b in m.content
        if isinstance(b, TextBlock | ProviderToolBlock)
    ]
    return "\n".join(p for p in parts if p)


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
            text = _assistant_text(m)
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
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

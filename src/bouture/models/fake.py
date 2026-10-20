"""Faux client LLM scripté, pour les tests et les exemples hors ligne."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from ..errors import ModelError
from ..messages import Usage
from .base import (
    PROVIDER_TOOLS,
    Done,
    Estimate,
    ModelEvent,
    ModelRequest,
    ProviderToolCall,
    ProviderToolResult,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    UsageReport,
)


@dataclass
class FakeReply:
    """Une réponse scriptée.

    - ``text`` est streamé en morceaux de ``chunk`` caractères ;
    - ``tool_calls`` : liste de ``(nom, args)`` ;
    - ``stream_args`` : les arguments des outils sont aussi streamés (``ToolCallDelta``) ;
    - ``provider_calls`` : outils fournisseur ``(nom, args, résultat)`` émis après ``text``
      (résultat ``None`` : pas de résultat), suivis de ``after_text`` (D116) ;
    - ``hang`` : après le texte (et ``after_text``), attend indéfiniment (pour tester stop / crash) ;
    - ``error`` : lève une ``ModelError`` (``transient`` réessayable) avant tout token.
    """

    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    provider_calls: list[tuple[str, dict[str, Any], str | None]] = field(default_factory=list)
    after_text: str = ""
    chunk: int = 4
    stream_args: bool = False
    delay: float = 0.0
    hang: bool = False
    error: ModelError | None = None
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=10, output_tokens=5))


Script = FakeReply | Callable[[ModelRequest], FakeReply]


class FakeModelClient:
    def __init__(
        self,
        script: list[Script] | None = None,
        *,
        default: FakeReply | None = None,
        provider_tools: Collection[str] = PROVIDER_TOOLS,
    ) -> None:
        self.script: list[Script] = list(script or [])
        self.supported = frozenset(provider_tools)
        self.default = default
        self.requests: list[ModelRequest] = []
        self.started = asyncio.Event()
        self._call = 0

    def push(self, *replies: Script) -> None:
        self.script.extend(replies)

    def provider_tools(self, model: str) -> frozenset[str]:
        return self.supported

    async def estimate(self, request: ModelRequest) -> Estimate:
        """Grossière : un token pour quatre caractères de prompt."""
        size = len(request.system) + sum(len(m.model_dump_json()) for m in request.messages)
        return Estimate(input_tokens=size // 4)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        if self.script:
            item = self.script.pop(0)
        elif self.default is not None:
            item = self.default
        else:
            raise AssertionError("FakeModelClient : script épuisé")
        reply = item(request) if callable(item) else item
        self._call += 1
        if reply.error is not None:
            raise reply.error
        self.started.set()
        for i in range(0, len(reply.text), reply.chunk):
            if reply.delay:
                await asyncio.sleep(reply.delay)
            else:
                await asyncio.sleep(0)
            yield TextDelta(reply.text[i : i + reply.chunk])
        for n, (name, args, result) in enumerate(reply.provider_calls):
            pid = f"srv_{self._call}_{n}"
            yield ProviderToolCall(pid, name, args)
            if result is not None:
                yield ProviderToolResult(pid, result)
        for i in range(0, len(reply.after_text), reply.chunk):
            await asyncio.sleep(0)
            yield TextDelta(reply.after_text[i : i + reply.chunk])
        if reply.hang:
            await asyncio.Event().wait()
        for n, (name, args) in enumerate(reply.tool_calls):
            call_id = f"call_{self._call}_{n}"
            if reply.stream_args:
                raw = json.dumps(args)
                for i in range(0, len(raw), reply.chunk):
                    head = i == 0
                    yield ToolCallDelta(
                        n,
                        call_id if head else None,
                        name if head else None,
                        raw[i : i + reply.chunk],
                    )
            yield ToolCallComplete(id=call_id, name=name, args=args)
        yield UsageReport(reply.usage)
        yield Done("tool_use" if reply.tool_calls else "end_turn")

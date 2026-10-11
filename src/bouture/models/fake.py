"""Faux client LLM scripté, pour les tests et les exemples hors ligne."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import ModelError
from ..messages import Usage
from .base import Done, ModelEvent, ModelRequest, TextDelta, ToolCallComplete, UsageReport


@dataclass
class FakeReply:
    """Une réponse scriptée.

    - ``text`` est streamé en morceaux de ``chunk`` caractères ;
    - ``tool_calls`` : liste de ``(nom, args)`` ;
    - ``hang`` : après le texte, attend indéfiniment (pour tester stop / crash) ;
    - ``error`` : lève une ``ModelError`` (``transient`` réessayable) avant tout token.
    """

    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    chunk: int = 4
    delay: float = 0.0
    hang: bool = False
    error: ModelError | None = None
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=10, output_tokens=5))


Script = FakeReply | Callable[[ModelRequest], FakeReply]


class FakeModelClient:
    def __init__(
        self, script: list[Script] | None = None, *, default: FakeReply | None = None
    ) -> None:
        self.script: list[Script] = list(script or [])
        self.default = default
        self.requests: list[ModelRequest] = []
        self.started = asyncio.Event()
        self._call = 0

    def push(self, *replies: Script) -> None:
        self.script.extend(replies)

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
        if reply.hang:
            await asyncio.Event().wait()
        for n, (name, args) in enumerate(reply.tool_calls):
            yield ToolCallComplete(id=f"call_{self._call}_{n}", name=name, args=args)
        yield UsageReport(reply.usage)
        yield Done("tool_use" if reply.tool_calls else "end_turn")

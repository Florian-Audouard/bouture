"""``ClaudeCodeClient(persistent=True)`` (D128) : processus gardé dans un tour, sans Claude Code."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest

from bouture import ModelRequest, UserMessage
from bouture.messages import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
)
from bouture.models.base import CallContext, Done, ToolSpec

sdk = pytest.importorskip("claude_agent_sdk")

from bouture.models.claude_code import ClaudeCodeClient  # noqa: E402

TOOL = ToolSpec("set_title", "Change le titre.", {"type": "object", "properties": {}})
NAME = "mcp__bouture__set_title"


def _result(**kwargs: Any) -> Any:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s",
        "stop_reason": "end_turn",
    }
    return sdk.ResultMessage(**{**fields, **kwargs})


def _call(cid: str) -> list[Any]:
    """Réponse de Claude Code qui demande ``set_title`` (appel différé)."""
    return [
        sdk.AssistantMessage(
            content=[sdk.ToolUseBlock(id=cid, name=NAME, input={"title": "A"})], model="haiku"
        ),
        _result(
            stop_reason="tool_deferred",
            deferred_tool_use=sdk.types.DeferredToolUse(id=cid, name=NAME, input={"title": "A"}),
        ),
    ]


def _text(text: str) -> list[Any]:
    return [
        sdk.AssistantMessage(content=[sdk.TextBlock(text=text)], model="haiku"),
        _result(),
    ]


class _Fleet:
    """Fabrique de faux ``ClaudeSDKClient`` : réponses scriptées, prompts et cycle de vie gardés."""

    def __init__(self, *replies: list[Any]) -> None:
        self.replies = list(replies)
        self.clients: list[_Client] = []

    def __call__(self, options: Any) -> _Client:
        c = _Client(self, options)
        self.clients.append(c)
        return c


class _Client:
    def __init__(self, fleet: _Fleet, options: Any) -> None:
        self.fleet = fleet
        self.options = options
        self.prompts: list[Any] = []
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def query(self, prompt: str | AsyncIterator[dict[str, Any]]) -> None:
        if isinstance(prompt, str):
            self.prompts.append(prompt)
        else:
            self.prompts.append([m async for m in prompt])

    async def receive_response(self) -> AsyncGenerator[Any]:
        for m in self.fleet.replies.pop(0):
            yield m

    async def disconnect(self) -> None:
        self.closed = True


def _request(messages: list[Message], turn: UUID | None, **kwargs: Any) -> ModelRequest:
    ctx = CallContext("alice", uuid4(), "note", turn)
    fields: dict[str, Any] = {
        "model": "claude-code/haiku",
        "system": "Sois bref.",
        "messages": messages,
        "tools": [TOOL],
        "context": ctx,
    }
    return ModelRequest(**{**fields, **kwargs})


def _user(text: str) -> UserMessage:
    return UserMessage(content=[TextBlock(text=text)])


def _step(cid: str) -> list[Message]:
    """Réponse de l'assistant avec un appel, puis son résultat (tel que bouture les enregistre)."""
    return [
        AssistantMessage(content=[ToolCallBlock(id=cid, name="set_title", args={"title": "A"})]),
        ToolResultMessage(tool_call_id=cid, name="set_title", content="ok"),
    ]


async def _run(client: ClaudeCodeClient, request: ModelRequest) -> list[Any]:
    return [ev async for ev in client.stream(request)]


async def test_continuation_sends_tool_results_only() -> None:
    fleet = _Fleet(_call("t1"), _text("Fini."))
    client = ClaudeCodeClient(persistent=True, client_factory=fleet)
    turn = uuid4()
    first = [_user("titre")]
    events = await _run(client, _request(first, turn))
    assert events[-1] == Done("tool_calls")
    assert len(fleet.clients) == 1 and not fleet.clients[0].closed  # gardé

    events = await _run(client, _request([*first, *_step("t1")], turn))
    assert events[-1] == Done("end_turn")
    c = fleet.clients[0]
    assert len(fleet.clients) == 1 and c.closed  # même processus, fermé en fin de tour
    assert c.prompts[0].startswith("<transcript>")
    [msg] = c.prompts[1]
    assert msg["message"]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
    ]


async def test_state_tail_is_replaced() -> None:
    """``<etat_courant>`` en fin de requête : l'ancien reste dans l'historique, le nouveau suit."""
    fleet = _Fleet(_call("t1"), _text("Fini."))
    client = ClaudeCodeClient(persistent=True, client_factory=fleet)
    turn = uuid4()
    head = [_user("titre")]
    await _run(client, _request([*head, _user("<etat_courant>v1</etat_courant>")], turn))
    error = ToolResultMessage(tool_call_id="t1", name="set_title", content="non", is_error=True)
    nxt = [*head, _step("t1")[0], error, _user("<etat_courant>v2</etat_courant>")]
    await _run(client, _request(nxt, turn))
    [msg] = fleet.clients[0].prompts[1]
    assert msg["message"]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "non", "is_error": True},
        {"type": "text", "text": "<etat_courant>v2</etat_courant>"},
    ]


async def test_mismatch_falls_back_to_transcript() -> None:
    fleet = _Fleet(_call("t1"), _text("Fini."))
    client = ClaudeCodeClient(persistent=True, client_factory=fleet)
    turn = uuid4()
    await _run(client, _request([_user("titre")], turn))
    # Historique réécrit (autre premier message) : pas la suite de la requête précédente.
    await _run(client, _request([_user("autre"), *_step("t1")], turn))
    assert len(fleet.clients) == 2
    assert fleet.clients[0].closed and fleet.clients[1].closed
    assert fleet.clients[1].prompts[0].startswith("<transcript>")


async def test_changed_tools_fall_back() -> None:
    fleet = _Fleet(_call("t1"), _text("Fini."))
    client = ClaudeCodeClient(persistent=True, client_factory=fleet)
    turn = uuid4()
    first = [_user("titre")]
    await _run(client, _request(first, turn))
    await _run(client, _request([*first, *_step("t1")], turn, tools=[]))
    assert len(fleet.clients) == 2 and fleet.clients[0].closed


async def test_without_turn_uses_query() -> None:
    fleet = _Fleet()
    seen: list[str] = []

    async def query(*, prompt: str, options: Any) -> AsyncGenerator[Any]:
        seen.append(prompt)
        for m in _text("Salut."):
            yield m

    client = ClaudeCodeClient(persistent=True, client_factory=fleet, query=query)
    events = await _run(client, _request([_user("x")], None))
    assert events[-1] == Done("end_turn") and seen and not fleet.clients


async def test_not_persistent_by_default() -> None:
    fleet = _Fleet()

    async def query(*, prompt: str, options: Any) -> AsyncGenerator[Any]:
        for m in _call("t1"):
            yield m

    client = ClaudeCodeClient(client_factory=fleet, query=query)
    await _run(client, _request([_user("x")], uuid4()))
    assert not fleet.clients


async def test_eviction_ttl_and_aclose() -> None:
    fleet = _Fleet(*[_call(f"t{i}") for i in range(4)])
    client = ClaudeCodeClient(persistent=True, client_factory=fleet, max_live=2)
    for _ in range(3):
        await _run(client, _request([_user("x")], uuid4()))
    assert [c.closed for c in fleet.clients] == [True, False, False]  # le plus ancien évincé

    client.live_ttl = 0.0
    await _run(client, _request([_user("x")], uuid4()))  # balaie les processus expirés
    assert [c.closed for c in fleet.clients] == [True, True, True, False]

    await client.aclose()
    assert all(c.closed for c in fleet.clients)


async def test_error_closes_process() -> None:
    fleet = _Fleet([sdk.AssistantMessage(content=[], model="haiku", error="rate_limit")])
    client = ClaudeCodeClient(persistent=True, client_factory=fleet)
    with pytest.raises(Exception, match="rate_limit"):
        await _run(client, _request([_user("x")], uuid4()))
    assert fleet.clients[0].closed


async def test_agents_shutdown_closes_model_client() -> None:
    from bouture import Agents, FakeModelClient, MemoryStorage

    class Closing(FakeModelClient):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    fake = Closing()
    a = Agents[Any](storage=MemoryStorage(), model_client=fake)
    await a.setup()
    await a.shutdown()
    assert fake.closed

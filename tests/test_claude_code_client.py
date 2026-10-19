"""``ClaudeCodeClient`` (D106) : flux, appels d'outils différés et transcript, sans Claude Code."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from bouture import ModelError, ModelRequest, UserMessage
from bouture.messages import (
    AssistantMessage,
    ImageBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
)
from bouture.models.base import (
    Done,
    ModelEvent,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolSpec,
    UsageReport,
)

sdk = pytest.importorskip("claude_agent_sdk")

from bouture.models.claude_code import ClaudeCodeClient, to_transcript  # noqa: E402

TOOL = ToolSpec("set_title", "Change le titre.", {"type": "object", "properties": {}})


def _request(**kwargs: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "model": "claude-code/haiku",
        "system": "Sois bref.",
        "messages": [UserMessage(content=[TextBlock(text="Bonjour")])],
    }
    return ModelRequest(**{**fields, **kwargs})


def _event(event: dict[str, Any], parent: str | None = None) -> Any:
    return sdk.StreamEvent(uuid="u", session_id="s", event=event, parent_tool_use_id=parent)


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


class _Query:
    """Remplace ``claude_agent_sdk.query`` : rejoue des messages et garde les arguments."""

    def __init__(self, *messages: Any) -> None:
        self.messages = messages
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, prompt: str, options: Any) -> AsyncGenerator[Any]:
        self.calls.append({"prompt": prompt, "options": options})
        return self._gen()

    async def _gen(self) -> AsyncGenerator[Any]:
        for m in self.messages:
            yield m


async def _collect(client: ClaudeCodeClient, request: ModelRequest) -> list[ModelEvent]:
    return [ev async for ev in client.stream(request)]


async def test_text_stream_and_usage() -> None:
    query = _Query(
        _event(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 12, "cache_read_input_tokens": 3}},
            }
        ),
        _event({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Sal"},
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ut"},
            }
        ),
        _event({"type": "message_delta", "usage": {"output_tokens": 5}}),
        sdk.AssistantMessage(content=[sdk.TextBlock(text="Salut")], model="haiku"),
        _result(),
    )
    events = await _collect(ClaudeCodeClient(query=query), _request())
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Sal", "ut"]
    usage = next(e.usage for e in events if isinstance(e, UsageReport))
    assert (usage.input_tokens, usage.cache_read, usage.output_tokens) == (12, 3, 5)
    assert events[-1] == Done("end_turn")

    options = query.calls[0]["options"]
    assert options.model == "haiku"
    assert options.tools == [] and options.setting_sources == [] and options.max_turns == 1
    assert options.env["ANTHROPIC_API_KEY"] == ""
    assert options.system_prompt.startswith("Sois bref.")
    assert query.calls[0]["prompt"].startswith("<transcript>")


async def test_deferred_tool_call() -> None:
    name = "mcp__bouture__set_title"
    query = _Query(
        _event(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "t1", "name": name},
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"title"'},
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "x"},
            },
            parent="t0",
        ),
        sdk.AssistantMessage(
            content=[sdk.ToolUseBlock(id="t1", name=name, input={"title": "A"})], model="haiku"
        ),
        _result(
            stop_reason="tool_use",
            deferred_tool_use=sdk.types.DeferredToolUse(id="t1", name=name, input={"title": "A"}),
        ),
    )
    events = await _collect(ClaudeCodeClient(query=query), _request(tools=[TOOL]))
    assert events[0] == ToolCallDelta(1, "t1", "set_title", "")
    assert events[1] == ToolCallDelta(1, None, None, '{"title"')
    assert not any(isinstance(e, TextDelta) for e in events)  # message d'un sous-agent ignoré
    assert [e for e in events if isinstance(e, ToolCallComplete)] == [
        ToolCallComplete("t1", "set_title", {"title": "A"})
    ]
    assert events[-1] == Done("tool_calls")
    assert "bouture" in query.calls[0]["options"].mcp_servers


async def test_errors() -> None:
    query = _Query(sdk.AssistantMessage(content=[], model="haiku", error="rate_limit"))
    with pytest.raises(ModelError) as info:
        await _collect(ClaudeCodeClient(query=query), _request())
    assert info.value.transient and info.value.code == "rate_limit"

    query = _Query(_result(subtype="error_during_execution", is_error=True, api_error_status=400))
    with pytest.raises(ModelError) as info:
        await _collect(ClaudeCodeClient(query=query), _request())
    assert not info.value.transient


async def test_api_key_kept_without_subscription() -> None:
    query = _Query(_result())
    client = ClaudeCodeClient(use_subscription=False, query=query, env={"FOO": "1"})
    await _collect(client, _request(model="claude-code/default"))
    options = query.calls[0]["options"]
    assert "ANTHROPIC_API_KEY" not in options.env and options.env["FOO"] == "1"
    assert options.model is None


def test_transcript() -> None:
    text = to_transcript(
        [
            UserMessage(content=[TextBlock(text="Regarde "), ImageBlock(ref="img://1")]),
            AssistantMessage(
                content=[
                    ReasoningBlock(text="secret"),
                    TextBlock(text="Je change."),
                    ToolCallBlock(id="t1", name="set_title", args={"title": "Été"}),
                ]
            ),
            ToolResultMessage(tool_call_id="t1", name="set_title", content="refusé", is_error=True),
        ]
    )
    assert "[image : img://1]" in text and "secret" not in text
    assert '<tool_call id="t1" name="set_title">{"title": "Été"}</tool_call>' in text
    assert '<tool_result id="t1" name="set_title" error="true">\nrefusé\n</tool_result>' in text

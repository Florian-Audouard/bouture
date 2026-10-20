"""Outils exécutés par le fournisseur (D116-D120), sans appel réel."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from types import SimpleNamespace
from typing import Any

import pytest

from bouture import (
    AgentDefinition,
    Agents,
    FakeModelClient,
    FakeReply,
    ModelRequest,
    ProviderToolBlock,
    ToolContext,
    UnsupportedProviderTool,
)
from bouture.messages import AssistantMessage, TextBlock, UserMessage
from bouture.models.base import (
    PROVIDER_RESULT_MAX,
    Done,
    ProviderToolCall,
    ProviderToolResult,
    TextDelta,
    ToolCallComplete,
    ToolCallDelta,
    UsageReport,
    clip_result,
    provider_tool_text,
)

from .conftest import AgentsFactory, Note, collect

OWNER = "alice"


def researcher(tools: Collection[str] = ("web_search",)) -> AgentDefinition[Note, Any]:
    d = AgentDefinition[Note, Any](
        "researcher", model="fake", state_type=Note, provider_tools=tools
    )

    @d.tool
    async def set_title(ctx: ToolContext[Note, Any], title: str) -> str:
        """Change le titre."""
        ctx.state.title = title
        return "ok"

    return d


class ByModel(FakeModelClient):
    """Faux client dont les outils fournisseur dépendent du modèle, comme ``LiteLLMClient``."""

    def provider_tools(self, model: str) -> frozenset[str]:
        return frozenset({"web_search"}) if model.startswith("fake") else frozenset()


# -- déclaration et capacités (D117) ---------------------------------------------------
def test_definition_rejects_unknown_provider_tools() -> None:
    with pytest.raises(ValueError, match="bash"):
        AgentDefinition[None, Any]("x", model="fake", provider_tools=["web_search", "bash"])
    d = AgentDefinition[None, Any]("x", model="fake", provider_tools=["web_fetch", "web_fetch"])
    assert d.provider_tools == ("web_fetch",)


async def test_register_rejects_unsupported_provider_tools(make_agents: AgentsFactory) -> None:
    a = make_agents(FakeModelClient(provider_tools=()))
    with pytest.raises(UnsupportedProviderTool, match="web_search"):
        a.register(researcher())
    # Un client qui ne déclare rien ne supporte rien.
    plain = SimpleNamespace(stream=None)
    b = Agents[Any](storage=a.storage, model_client=plain)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedProviderTool):
        b.register(researcher())
    b.register(AgentDefinition[None, Any]("sans", model="fake"))


async def test_model_override_is_checked(make_agents: AgentsFactory) -> None:
    a = make_agents(ByModel())
    a.register(researcher())
    await a.setup()
    with pytest.raises(UnsupportedProviderTool, match="autre"):
        await a.create_session("researcher", OWNER, model="autre")
    s = await a.create_session("researcher", OWNER, model="fake-2")
    with pytest.raises(UnsupportedProviderTool):
        await a.set_model(s.id, OWNER, "autre")
    assert (await a.set_model(s.id, OWNER, None)).model == "fake"


# -- tour : trace dans l'arbre et événements (D116) -------------------------------------
async def test_provider_calls_are_recorded_not_executed(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient(
        [
            FakeReply(
                "Je cherche. ",
                provider_calls=[("web_search", {"query": "bouture"}, "- Bouture — https://b.fr")],
                after_text="Trouvé.",
                tool_calls=[("set_title", {"title": "Bouture"})],
            ),
            FakeReply("Fini."),
            FakeReply("Encore."),
        ]
    )
    a = make_agents(fake)
    a.register(researcher())
    await a.setup()
    s = await a.create_session("researcher", OWNER)
    evs = await collect((await a.send_message(s.id, OWNER, "cherche")).events())

    assert fake.requests[0].provider_tools == ["web_search"]
    kinds = [e.type for e in evs]
    assert kinds.index("provider_tool_call") < kinds.index("provider_tool_result")
    assert kinds.index("provider_tool_result") < kinds.index("tool_call")
    call = next(e for e in evs if e.type == "provider_tool_call")
    assert call.data["name"] == "web_search" and call.data["args"] == {"query": "bouture"}
    result = next(e for e in evs if e.type == "provider_tool_result")
    assert result.data["id"] == call.data["id"] and not result.data["is_error"]
    # Un seul résultat d'outil : celui de set_title, exécuté par bouture.
    assert [e.data["name"] for e in evs if e.type == "tool_result"] == ["set_title"]
    assert evs[-1].data["reason"] == "completed"

    branch = await a.get_branch(s.id, OWNER)
    first = AssistantMessage.model_validate(
        next(e for e in branch if e.type == "assistant").content
    )
    assert [b.type for b in first.content] == ["text", "provider_tool", "text", "tool_call"]
    assert first.text == "Je cherche. Trouvé."
    assert first.provider_tools[0].result == "- Bouture — https://b.fr"
    assert first.tool_calls[0].name == "set_title"

    # Le tour suivant voit la recherche passée dans l'historique.
    await (await a.send_message(s.id, OWNER, "et ?")).wait()
    history = fake.requests[2].messages
    replayed = [m for m in history if isinstance(m, AssistantMessage) and m.provider_tools]
    assert replayed and replayed[0].provider_tools[0].args == {"query": "bouture"}


async def test_custom_builder_inherits_provider_tools(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient([FakeReply("ok")])
    a = make_agents(fake)
    d = researcher(("web_search", "web_fetch"))

    @d.build_context
    async def build(ctx: Any) -> ModelRequest:
        return ModelRequest(model=ctx.model, system="", messages=[])

    a.register(d)
    await a.setup()
    s = await a.create_session("researcher", OWNER)
    await (await a.send_message(s.id, OWNER, "x")).wait()
    assert fake.requests[0].provider_tools == ["web_search", "web_fetch"]


async def test_recovery_replays_provider_calls(make_agents: AgentsFactory) -> None:
    hung = asyncio.Event()

    def hang(request: ModelRequest) -> FakeReply:
        hung.set()
        return FakeReply(
            "avant ", provider_calls=[("web_fetch", {"url": "https://b.fr"}, "page")], hang=True
        )

    a1 = make_agents(FakeModelClient([hang]), lock_ttl=0.3)
    a1.register(researcher(("web_fetch",)))
    await a1.setup()
    s = await a1.create_session("researcher", OWNER)
    await a1.send_message(s.id, OWNER, "lis")
    await hung.wait()
    await asyncio.sleep(0.15)
    for ex in list(a1._executors.values()):  # pyright: ignore[reportPrivateUsage]
        ex.cancel("abandon")
    await a1.runner.join(1)
    await asyncio.sleep(0.4)

    a2 = make_agents(FakeModelClient([FakeReply("reprise")]), lock_ttl=0.3)
    a2.register(researcher(("web_fetch",)))
    await a2.setup()
    end = await (await a2.send_message(s.id, OWNER, "encore")).wait()
    assert end.data["reason"] == "completed"
    tree = await a2.get_tree(s.id, OWNER)
    partial = next(e for e in tree if e.type == "assistant")
    msg = AssistantMessage.model_validate(partial.content)
    assert msg.interrupted
    assert msg.provider_tools == [
        ProviderToolBlock(
            id="srv_1_0", name="web_fetch", args={"url": "https://b.fr"}, result="page"
        )
    ]


# -- rejeu en texte (D118) ------------------------------------------------------------
def _searched() -> AssistantMessage:
    return AssistantMessage(
        content=[
            TextBlock(text="Je cherche."),
            ProviderToolBlock(id="s1", name="web_search", args={"query": "q"}, result="- A — u"),
            TextBlock(text="Trouvé."),
        ]
    )


def test_provider_tool_text_and_clip() -> None:
    b = ProviderToolBlock(
        id="s1", name="web_search", args={"query": "é"}, is_error=True, result="x"
    )
    text = provider_tool_text(b)
    assert text.startswith('<provider_tool id="s1" name="web_search">{"query": "é"}')
    assert '<provider_tool_result error="true">' in text
    assert "result" not in provider_tool_text(ProviderToolBlock(id="s", name="web_fetch", args={}))
    long = "a" * (PROVIDER_RESULT_MAX + 5)
    assert clip_result(long).endswith(f"[… tronqué, {PROVIDER_RESULT_MAX + 5} caractères]")


def test_claude_code_transcript_replays_provider_tools() -> None:
    pytest.importorskip("claude_agent_sdk")
    from bouture.models.claude_code import to_transcript

    text = to_transcript([UserMessage(content=[TextBlock(text="q")]), _searched()])
    assert "Je cherche.\n<provider_tool" in text and "</provider_tool_result>\nTrouvé." in text


def test_openai_messages_replay_provider_tools() -> None:
    pytest.importorskip("litellm")
    from bouture.models.litellm import to_openai_messages

    out = to_openai_messages("", [UserMessage(content=[TextBlock(text="q")]), _searched()])
    content = out[-1]["content"]
    assert content.startswith("Je cherche.\n<provider_tool") and content.endswith("Trouvé.")


# -- ClaudeCodeClient (D119) ----------------------------------------------------------
async def test_claude_code_allows_web_tools_and_records_them() -> None:
    sdk = pytest.importorskip("claude_agent_sdk")
    from bouture.models.claude_code import ClaudeCodeClient

    from .test_claude_code_client import _event, _Query, _result  # pyright: ignore[reportPrivateUsage]

    web = {"type": "tool_use", "id": "w1", "name": "WebSearch"}
    query = _Query(
        _event({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
        _event({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Je cherche."},
            }
        ),
        _event({"type": "content_block_start", "index": 1, "content_block": web}),
        _event(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"query": "q"}'},
            }
        ),
        _event({"type": "message_delta", "usage": {"output_tokens": 7}}),
        sdk.AssistantMessage(
            content=[sdk.ToolUseBlock(id="w1", name="WebSearch", input={"query": "q"})],
            model="haiku",
        ),
        sdk.UserMessage(
            content=[
                sdk.ToolResultBlock(
                    tool_use_id="w1", content=[{"type": "text", "text": "- A — https://a"}]
                )
            ]
        ),
        _event({"type": "message_start", "message": {"usage": {"input_tokens": 30}}}),
        _event({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Trouvé."},
            }
        ),
        _event({"type": "message_delta", "usage": {"output_tokens": 4}}),
        _result(num_turns=2),
    )
    client = ClaudeCodeClient(query=query, provider_tool_turns=3)
    assert client.provider_tools("sonnet") == {"web_search", "web_fetch"}
    request = ModelRequest(
        model="claude-code/haiku",
        system="",
        messages=[UserMessage(content=[TextBlock(text="q")])],
        provider_tools=["web_search"],
    )
    events = [ev async for ev in client.stream(request)]

    options = query.calls[0]["options"]
    assert options.tools == ["WebSearch"] and options.max_turns == 4
    assert not any(isinstance(e, ToolCallDelta | ToolCallComplete) for e in events)
    kinds = [type(e) for e in events]
    assert kinds == [TextDelta, ProviderToolCall, ProviderToolResult, TextDelta, UsageReport, Done]
    assert events[1] == ProviderToolCall("w1", "web_search", {"query": "q"})
    assert events[2] == ProviderToolResult("w1", "- A — https://a", False)
    usage = next(e for e in events if isinstance(e, UsageReport)).usage
    assert usage.input_tokens == 40 and usage.output_tokens == 11

    # Le hook laisse passer la recherche et diffère les outils de bouture.
    hook = options.hooks["PreToolUse"][0].hooks[0]
    allow = await hook({"tool_name": "WebSearch"}, "w1", None)
    defer = await hook({"tool_name": "mcp__bouture__set_title"}, "t1", None)
    assert allow["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert defer["hookSpecificOutput"]["permissionDecision"] == "defer"


async def test_claude_code_budget_exhausted_ends_cleanly() -> None:
    pytest.importorskip("claude_agent_sdk")
    from bouture.models.claude_code import ClaudeCodeClient

    from .test_claude_code_client import _Query, _result  # pyright: ignore[reportPrivateUsage]

    query = _Query(_result(subtype="error_max_turns", is_error=True, stop_reason=None))
    request = ModelRequest(
        model="claude-code/haiku", system="", messages=[], provider_tools=["web_fetch"]
    )
    events = [ev async for ev in ClaudeCodeClient(query=query).stream(request)]
    assert events[-1] == Done("max_turns")
    # Sans outil fournisseur, rien ne change : un seul tour, pas d'outil du CLI.
    query2 = _Query(_result())
    plain = ModelRequest(model="claude-code/haiku", system="", messages=[])
    _ = [ev async for ev in ClaudeCodeClient(query=query2).stream(plain)]
    assert query2.calls[0]["options"].tools == [] and query2.calls[0]["options"].max_turns == 1


# -- LiteLLMClient (D120) -------------------------------------------------------------
class _Chunks:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks

    def __aiter__(self) -> _Chunks:
        return self

    async def __anext__(self) -> Any:
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def aclose(self) -> None:
        pass


def _chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    fields: dict[str, Any] | None = None,
    finish: str | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, provider_specific_fields=fields)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def _tc(index: int, id: str | None, name: str | None, args: str) -> Any:
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=args))


async def test_litellm_hosted_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    litellm = pytest.importorskip("litellm")
    from bouture.models.litellm import LiteLLMClient

    search_result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "title": "A", "url": "https://a"}],
    }
    chunks = [
        _chunk("Je cherche."),
        _chunk(tool_calls=[_tc(0, "srvtoolu_1", "web_search", "")]),
        _chunk(tool_calls=[_tc(0, None, None, '{"query": "q"}')]),
        _chunk(fields={"web_search_results": [search_result]}),
        _chunk("Trouvé."),
        _chunk(fields={"web_search_results": [search_result]}),  # liste cumulative
        _chunk(tool_calls=[_tc(1, "toolu_2", "set_title", '{"title": "A"}')], finish="tool_use"),
    ]
    seen: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _Chunks:
        seen.update(kwargs)
        return _Chunks(chunks)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    client = LiteLLMClient(provider_tool_options={"web_search": {"max_uses": 2}})
    assert client.provider_tools("anthropic/claude-sonnet-5-5") == {"web_search", "web_fetch"}
    assert client.provider_tools("openai/gpt-4o-mini") == frozenset()
    request = ModelRequest(
        model="anthropic/claude-sonnet-5-5",
        system="",
        messages=[UserMessage(content=[TextBlock(text="q")])],
        provider_tools=["web_search"],
    )
    events = [ev async for ev in client.stream(request)]

    assert {"type": "web_search_20250305", "name": "web_search", "max_uses": 2} in seen["tools"]
    assert [type(e) for e in events if not isinstance(e, ToolCallDelta)] == [
        TextDelta,
        ProviderToolCall,
        ProviderToolResult,
        TextDelta,
        ToolCallComplete,
        Done,
    ]
    assert all(e.id != "srvtoolu_1" for e in events if isinstance(e, ToolCallDelta))
    assert ProviderToolCall("srvtoolu_1", "web_search", {"query": "q"}) in events
    assert ProviderToolResult("srvtoolu_1", "- A — https://a", False) in events
    complete = next(e for e in events if isinstance(e, ToolCallComplete))
    assert complete.name == "set_title"


def test_litellm_summarizes_fetch_and_errors() -> None:
    pytest.importorskip("litellm")
    from bouture.models.litellm import summarize_server_result

    fetched = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_2",
        "content": {
            "type": "web_fetch_result",
            "url": "https://b.fr",
            "content": {
                "type": "document",
                "title": "B",
                "source": {"type": "text", "media_type": "text/plain", "data": "corps"},
            },
        },
    }
    assert summarize_server_result(fetched) == ("https://b.fr\nB\n\ncorps", False)
    error = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_3",
        "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
    }
    assert summarize_server_result(error) == ("erreur : max_uses_exceeded", True)
    assert summarize_server_result({"content": []}) == ("aucun résultat", False)

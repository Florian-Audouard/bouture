"""Images et contenu riche dans les résultats d'outils (D112-D115), sans appel réel."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from bouture import (
    AgentDefinition,
    Agents,
    FakeModelClient,
    FakeReply,
    ImageContext,
    ModelRequest,
    ToolContext,
)
from bouture.messages import ImageBlock, TextBlock, ToolResultMessage, UserMessage
from bouture.models.base import LoadedImage
from bouture.turn import to_result

from .conftest import AgentsFactory, Note, collect

OWNER = "alice"
PNG = b"\x89PNG\r\n\x1a\nfake"


def studio() -> AgentDefinition[Note, Any]:
    d = AgentDefinition[Note, Any]("studio", model="fake", state_type=Note)

    @d.tool
    async def render(ctx: ToolContext[Note, Any], ref: str) -> list[TextBlock | ImageBlock]:
        """Génère une image."""
        ctx.state.title = f"rendu {ref}"
        return [TextBlock(text=f"Image {ref} générée."), ImageBlock(ref=ref)]

    @d.tool
    async def listing(ctx: ToolContext[Note, Any]) -> list[dict[str, str]]:
        """Renvoie des données ordinaires."""
        return [{"ref": "a"}]

    return d


def with_loader(a: Agents[Any], store: dict[str, bytes], seen: list[str] | None = None) -> None:
    @a.image_loader
    async def _load(ref: str, ctx: ImageContext[Any]) -> tuple[bytes, str]:
        assert ctx.owner == OWNER and ctx.kind == "studio"
        if seen is not None:
            seen.append(ref)
        if ref not in store:
            raise FileNotFoundError(ref)
        return store[ref], "image/png"


async def _setup(make_agents: AgentsFactory, fake: FakeModelClient, **kw: Any) -> Agents[Any]:
    a = make_agents(fake, **kw)
    a.register(studio())
    return a


def _tool_results(request: ModelRequest) -> list[ToolResultMessage]:
    return [m for m in request.messages if isinstance(m, ToolResultMessage)]


async def test_tool_returns_image_blocks(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient(
        [FakeReply("", tool_calls=[("render", {"ref": "img/1.png"})]), FakeReply("Voilà.")]
    )
    a = await _setup(make_agents, fake)
    seen: list[str] = []
    with_loader(a, {"img/1.png": PNG}, seen)
    await a.setup()
    s = await a.create_session("studio", OWNER)
    evs = await collect((await a.send_message(s.id, OWNER, "dessine")).events())

    tr = next(e for e in evs if e.type == "tool_result")
    assert not tr.data["is_error"]
    assert tr.data["content"] == "Image img/1.png générée."  # texte : compatibilité
    assert tr.data["blocks"] == [
        {"type": "text", "text": "Image img/1.png générée."},
        {"type": "image", "ref": "img/1.png", "media_type": "image/png"},
    ]
    # D45 : l'arbre ne garde que la référence, jamais les octets.
    branch = await a.get_branch(s.id, OWNER)
    entry = next(e for e in branch if e.type == "tool_result")
    assert entry.content["content"][1] == {
        "type": "image",
        "ref": "img/1.png",
        "media_type": "image/png",
    }
    assert "data" not in str(entry.content)
    # La requête suivante porte les octets, chargés une seule fois pour le tour.
    second = fake.requests[1]
    assert _tool_results(second)[0].images[0].ref == "img/1.png"
    assert second.images["img/1.png"] == LoadedImage(PNG, "image/png")
    assert seen == ["img/1.png"]
    assert (await a.get_state(s.id, OWNER)).state.title == "rendu img/1.png"


async def test_missing_image_is_tool_error(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient(
        [FakeReply("", tool_calls=[("render", {"ref": "perdue.png"})]), FakeReply("Tant pis.")]
    )
    a = await _setup(make_agents, fake)
    with_loader(a, {})
    await a.setup()
    s = await a.create_session("studio", OWNER)
    evs = await collect((await a.send_message(s.id, OWNER, "dessine")).events())

    tr = next(e for e in evs if e.type == "tool_result")
    assert tr.data["is_error"] and "perdue.png" in tr.data["content"]
    assert "FileNotFoundError" in tr.data["content"] and "blocks" not in tr.data
    assert evs[-1].data["reason"] == "completed"
    # L'état modifié par l'outil est annulé, comme pour une exception.
    assert (await a.get_state(s.id, OWNER)).state.title == ""
    assert "state_changed" not in [e.type for e in evs]


async def test_without_loader_images_go_by_ref(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient(
        [FakeReply("", tool_calls=[("render", {"ref": "https://x/1.png"})]), FakeReply("ok")]
    )
    a = await _setup(make_agents, fake)
    await a.setup()
    s = await a.create_session("studio", OWNER)
    end = await (await a.send_message(s.id, OWNER, "dessine")).wait()
    assert end.data["reason"] == "completed"
    second = fake.requests[1]
    assert second.images == {}
    assert _tool_results(second)[0].images[0].ref == "https://x/1.png"


async def test_user_images_use_the_loader(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient([FakeReply("Jolie photo.")])
    a = await _setup(make_agents, fake)
    with_loader(a, {"upload/chat.png": PNG})
    await a.setup()
    s = await a.create_session("studio", OWNER)
    content = [TextBlock(text="Regarde"), ImageBlock(ref="upload/chat.png")]
    await (await a.send_message(s.id, OWNER, content)).wait()
    assert fake.requests[0].images == {"upload/chat.png": LoadedImage(PNG, "image/png")}


async def test_image_deleted_later_becomes_placeholder(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient(
        [
            FakeReply("", tool_calls=[("render", {"ref": "img/1.png"})]),
            FakeReply("Voilà."),
            FakeReply("Je ne la vois plus."),
        ]
    )
    a = await _setup(make_agents, fake)
    store = {"img/1.png": PNG}
    with_loader(a, store)
    await a.setup()
    s = await a.create_session("studio", OWNER)
    await (await a.send_message(s.id, OWNER, "dessine")).wait()
    del store["img/1.png"]  # supprimée par l'application hôte entre deux tours
    end = await (await a.send_message(s.id, OWNER, "et maintenant ?")).wait()
    assert end.data["reason"] == "completed"
    third = fake.requests[2]
    assert third.images == {}
    result = _tool_results(third)[0]
    assert result.images == [] and "[image indisponible : img/1.png]" in result.text
    # L'arbre, lui, garde la référence d'origine.
    branch = await a.get_branch(s.id, OWNER)
    entry = next(e for e in branch if e.type == "tool_result")
    assert entry.content["content"][1]["ref"] == "img/1.png"


async def test_recovery_replays_blocks(make_agents: AgentsFactory) -> None:
    hung = asyncio.Event()

    def hang(request: ModelRequest) -> FakeReply:
        hung.set()
        return FakeReply("partiel", hang=True)

    fake = FakeModelClient([FakeReply("", tool_calls=[("render", {"ref": "img/1.png"})]), hang])
    a1 = await _setup(make_agents, fake, lock_ttl=0.3)
    await a1.setup()
    s = await a1.create_session("studio", OWNER)
    await a1.send_message(s.id, OWNER, "x")
    await hung.wait()  # second appel : celui qui reste bloqué
    await asyncio.sleep(0.15)
    for ex in list(a1._executors.values()):  # pyright: ignore[reportPrivateUsage]
        ex.cancel("abandon")
    await a1.runner.join(1)
    await asyncio.sleep(0.4)

    a2 = await _setup(make_agents, FakeModelClient([FakeReply("reprise")]), lock_ttl=0.3)
    await a2.setup()
    end = await (await a2.send_message(s.id, OWNER, "encore")).wait()
    assert end.data["reason"] == "completed"
    tree = await a2.get_tree(s.id, OWNER)
    entry = next(e for e in tree if e.type == "tool_result")
    assert entry.content["content"][1]["ref"] == "img/1.png"


def test_old_string_results_stay_readable() -> None:
    old = ToolResultMessage.model_validate(
        {"tool_call_id": "c1", "name": "t", "content": "ok", "is_error": False}
    )
    assert old.content == "ok" and old.text == "ok" and old.images == []
    rich = ToolResultMessage.model_validate(
        {
            "tool_call_id": "c1",
            "name": "t",
            "content": [{"type": "text", "text": "a"}, {"type": "image", "ref": "r"}],
        }
    )
    assert rich.text == "a" and [b.ref for b in rich.images] == ["r"]


def test_to_result_keeps_plain_data_as_json() -> None:
    plain = to_result([{"ref": "a"}])
    assert isinstance(plain, str) and json.loads(plain) == [{"ref": "a"}]
    assert to_result([]) == "[]"
    assert to_result(ImageBlock(ref="r")) == [ImageBlock(ref="r")]
    assert to_result("texte") == "texte"


# -- clients -----------------------------------------------------------------------


def _history() -> list[Any]:
    from bouture.messages import AssistantMessage, ToolCallBlock

    return [
        UserMessage(content=[TextBlock(text="dessine")]),
        AssistantMessage(content=[ToolCallBlock(id="c1", name="render", args={})]),
        ToolResultMessage(
            tool_call_id="c1",
            name="render",
            content=[TextBlock(text="fait"), ImageBlock(ref="img/1.png")],
        ),
    ]


def test_litellm_tool_images_inline_and_user() -> None:
    pytest.importorskip("litellm")
    from bouture.models.litellm import to_openai_messages

    images = {"img/1.png": LoadedImage(PNG, "image/png")}
    inline = to_openai_messages("", _history(), images=images, tool_images="inline")
    assert inline[-1]["role"] == "tool"
    assert inline[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    user = to_openai_messages("", _history(), images=images, tool_images="user")
    tool, follow = user[-2], user[-1]
    assert tool["role"] == "tool" and isinstance(tool["content"], str)
    assert "[image 1 : img/1.png, jointe au message suivant]" in tool["content"]
    assert follow["role"] == "user"
    assert "Image 1 renvoyée par l'outil render (c1)" in follow["content"][0]["text"]
    assert follow["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    # Sans octets chargés, l'image part par sa référence.
    by_ref = to_openai_messages("", _history(), tool_images="inline")
    assert by_ref[-1]["content"][1]["image_url"]["url"] == "img/1.png"


async def test_claude_code_sends_images_as_content_blocks() -> None:
    pytest.importorskip("claude_agent_sdk")
    from bouture.models.claude_code import ClaudeCodeClient, to_transcript

    from .test_claude_code_client import _Query, _result  # pyright: ignore[reportPrivateUsage]

    query = _Query(_result())
    history = _history()
    request = ModelRequest(
        model="claude-code/haiku",
        system="",
        messages=history,
        images={"img/1.png": LoadedImage(PNG, "image/png")},
    )
    _ = [ev async for ev in ClaudeCodeClient(query=query).stream(request)]
    prompt = query.calls[0]["prompt"]
    assert not isinstance(prompt, str)
    messages = [m async for m in prompt]
    assert len(messages) == 1 and messages[0]["type"] == "user"
    content = messages[0]["message"]["content"]
    assert [b["type"] for b in content] == ["text", "image", "text"]
    assert content[0]["text"].endswith("[image : img/1.png]")
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[2]["text"].endswith("</transcript>")
    # Le texte reste identique au transcript sans image.
    assert content[0]["text"] + content[2]["text"] == to_transcript(history)

    # Sans image chargée : prompt texte, comme avant.
    query2 = _Query(_result())
    request2 = ModelRequest(model="claude-code/haiku", system="", messages=history)
    _ = [ev async for ev in ClaudeCodeClient(query=query2).stream(request2)]
    assert isinstance(query2.calls[0]["prompt"], str)


def test_litellm_auto_mode_follows_provider() -> None:
    pytest.importorskip("litellm")
    from bouture.models.litellm import LiteLLMClient

    images = {"img/1.png": LoadedImage(PNG, "image/png")}
    client = LiteLLMClient()

    def last(model: str) -> dict[str, Any]:
        request = ModelRequest(model=model, system="", messages=_history(), images=images)
        return client._messages(request)[-1]  # pyright: ignore[reportPrivateUsage]

    assert last("anthropic/claude-sonnet-5-5")["role"] == "tool"
    assert last("openai/gpt-4o-mini")["role"] == "user"

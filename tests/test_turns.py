from __future__ import annotations

import asyncio
from typing import Annotated, Any

import pytest
from pydantic import Field

from bouture import (
    AgentDefinition,
    Agents,
    FakeModelClient,
    FakeReply,
    ModelError,
    NotAwaitingInput,
    NotFound,
    SessionBusy,
    ToolContext,
)
from bouture.events import EventSink

from .conftest import AgentsFactory, Note, collect, types

OWNER = "alice"


async def test_simple_turn(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("Bonjour Alice"))
    s = await agents.create_session("note", OWNER, title="Ma note")
    h = await agents.send_message(s.id, OWNER, "Salut")
    evs = await collect(h.events())
    assert types(evs)[0] == "turn_start"
    assert "user_message" in types(evs)
    text = "".join(e.data["text"] for e in evs if e.type == "text_delta")
    assert text == "Bonjour Alice"
    end = evs[-1]
    assert end.type == "turn_end" and end.data["reason"] == "completed"
    assert end.data["usage"]["input_tokens"] == 10

    branch = await agents.get_branch(s.id, OWNER)
    assert [e.type for e in branch] == ["root", "user", "assistant"]
    assert branch[-1].is_checkpoint
    info = await agents.get_session(s.id, OWNER)
    assert info.status == "idle" and info.leaf_entry_id == branch[-1].id
    # Un seul appel LLM pour un chat sans outil, avec l'état courant en fin de contexte.
    assert len(fake.requests) == 1
    req = fake.requests[0]
    assert "Tu édites une note." in req.system
    last = req.messages[-1]
    assert last.role == "user" and "<etat_courant>" in last.content[0].text  # type: ignore[union-attr]


async def test_tool_modifies_state(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("", tool_calls=[("set_title", {"title": "Dragons"})]), FakeReply("Fait."))
    s = await agents.create_session("note", OWNER)
    evs = await collect((await agents.send_message(s.id, OWNER, "Titre ?")).events())
    sc = next(e for e in evs if e.type == "state_changed")
    assert sc.data["patch"] == [{"op": "replace", "path": "/title", "value": "Dragons"}]
    tr = next(e for e in evs if e.type == "tool_result")
    assert tr.data["name"] == "set_title" and not tr.data["is_error"]
    st = await agents.get_state(s.id, OWNER)
    assert st.state.title == "Dragons"
    branch = await agents.get_branch(s.id, OWNER)
    assert [e.type for e in branch] == ["root", "user", "assistant", "tool_result", "assistant"]
    # Snapshot créé seulement au changement d'état.
    assert branch[1].state_id == branch[0].state_id
    assert branch[3].state_id != branch[2].state_id
    assert len(fake.requests) == 2


async def test_invalid_state_is_reverted(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("", tool_calls=[("set_score", {"score": -3})]), FakeReply("ok"))
    s = await agents.create_session("note", OWNER)
    evs = await collect((await agents.send_message(s.id, OWNER, "x")).events())
    tr = next(e for e in evs if e.type == "tool_result")
    assert tr.data["is_error"] and "invalide" in tr.data["content"]
    assert (await agents.get_state(s.id, OWNER)).state.score == 0
    assert "state_changed" not in types(evs)


async def test_tool_exception_restores_state(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("", tool_calls=[("boom", {})]), FakeReply("désolé"))
    s = await agents.create_session("note", OWNER)
    evs = await collect((await agents.send_message(s.id, OWNER, "x")).events())
    tr = next(e for e in evs if e.type == "tool_result")
    assert tr.data["is_error"] and "panne" in tr.data["content"]
    assert (await agents.get_state(s.id, OWNER)).state.title == ""


async def test_unknown_tool_and_bad_args(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(
        FakeReply("", tool_calls=[("nope", {}), ("set_title", {"wrong": 1})]),
        FakeReply("ok"),
    )
    s = await agents.create_session("note", OWNER)
    evs = await collect((await agents.send_message(s.id, OWNER, "x")).events())
    results = [e for e in evs if e.type == "tool_result"]
    assert [r.data["is_error"] for r in results] == [True, True]
    assert evs[-1].data["reason"] == "completed"


async def test_pause_and_resume_after_restart(make_agents: AgentsFactory, storage: Any) -> None:
    fake1 = FakeModelClient([FakeReply("", tool_calls=[("ask_tag", {"proposal": "épique"})])])
    a1 = make_agents(fake1)
    await a1.setup()
    s = await a1.create_session("note", OWNER)
    evs = await collect((await a1.send_message(s.id, OWNER, "Propose un tag")).events())
    ask = next(e for e in evs if e.type == "ask_user")
    assert ask.data["choices"] == ["oui", "non"] and ask.data["tool_name"] == "ask_tag"
    assert evs[-1].data["reason"] == "awaiting_input"
    info = await a1.get_session(s.id, OWNER)
    assert info.status == "awaiting_input" and info.pending[0].question.startswith("Ajouter")
    # L'état modifié par l'outil qui pose la question est bien persisté.
    assert (await a1.get_state(s.id, OWNER)).state.title == "brouillon"
    await a1.shutdown()

    # « Redémarrage » : nouvelle instance sur le même stockage.
    fake2 = FakeModelClient([FakeReply("Tag ajouté !")])
    a2 = make_agents(fake2)
    await a2.setup()
    evs = await collect((await a2.answer(s.id, OWNER, "oui")).events())
    tr = next(e for e in evs if e.type == "tool_result")
    assert tr.data["content"] == "tag épique ajouté"
    assert evs[-1].data["reason"] == "completed"
    assert (await a2.get_state(s.id, OWNER)).state.tags == ["épique"]
    assert (await a2.get_session(s.id, OWNER)).status == "idle"
    with pytest.raises(NotAwaitingInput):
        await a2.answer(s.id, OWNER, "oui")


async def test_message_during_pause_cancels_question(
    agents: Agents[Any], fake: FakeModelClient
) -> None:
    fake.push(FakeReply("", tool_calls=[("ask_tag", {"proposal": "x"})]), FakeReply("Autre chose"))
    s = await agents.create_session("note", OWNER)
    await (await agents.send_message(s.id, OWNER, "a")).wait()
    evs = await collect((await agents.send_message(s.id, OWNER, "Laisse tomber")).events())
    tr = next(e for e in evs if e.type == "tool_result")
    assert tr.data["content"].startswith("Question annulée")
    branch = await agents.get_branch(s.id, OWNER)
    assert [e.type for e in branch] == [
        "root",
        "user",
        "assistant",
        "tool_result",
        "user",
        "assistant",
    ]
    assert (await agents.get_session(s.id, OWNER)).pending == []


async def test_session_busy_and_owner_isolation(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("lent", hang=True))
    s = await agents.create_session("note", OWNER)
    with pytest.raises(NotFound):
        await agents.get_session(s.id, "bob")
    with pytest.raises(NotFound):
        await agents.send_message(s.id, "bob", "intrus")
    assert (await agents.list_sessions("bob")).items == []
    h = await agents.send_message(s.id, OWNER, "x")
    await fake.started.wait()
    with pytest.raises(SessionBusy):
        await agents.send_message(s.id, OWNER, "encore")
    with pytest.raises(NotFound):
        await agents.stop(s.id, "bob")
    await agents.stop(s.id, OWNER)
    end = await h.wait()
    assert end.data["reason"] == "interrupted"


async def test_stop_during_streaming_keeps_partial_text(
    agents: Agents[Any], fake: FakeModelClient
) -> None:
    fake.push(FakeReply("Il était une fois", hang=True))
    s = await agents.create_session("note", OWNER)
    h = await agents.send_message(s.id, OWNER, "Raconte")
    await fake.started.wait()
    await asyncio.sleep(0.05)
    await agents.stop(s.id, OWNER)
    evs = await collect(h.events())
    assert evs[-1].data["reason"] == "interrupted"
    branch = await agents.get_branch(s.id, OWNER)
    last = branch[-1]
    assert last.type == "assistant" and last.interrupted and last.is_checkpoint
    assert last.content["content"][0]["text"] == "Il était une fois"
    # On peut reprendre la conversation après un stop.
    fake.push(FakeReply("Suite"))
    assert (await (await agents.send_message(s.id, OWNER, "continue")).wait()).data[
        "reason"
    ] == "completed"


async def test_stop_during_tool_restores_state(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("", tool_calls=[("slow", {})]))
    s = await agents.create_session("note", OWNER)
    h = await agents.send_message(s.id, OWNER, "x")
    async for ev in h.events():
        if ev.type == "tool_call":
            break
    await asyncio.sleep(0.05)
    await agents.stop(s.id, OWNER)
    end = await h.wait()
    assert end.data["reason"] == "interrupted"
    branch = await agents.get_branch(s.id, OWNER)
    assert branch[-1].type == "tool_result"
    assert branch[-1].content["is_error"] and branch[-1].content["content"].startswith("Interrompu")
    assert (await agents.get_state(s.id, OWNER)).state.title == ""


async def test_cancel_right_after_tool_calls_closes_them(
    agents: Agents[Any], fake: FakeModelClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interruption pendant ``message_end`` : le message assistant est enregistré, ses appels
    reçoivent quand même « Interrompu » (D126)."""
    emit = EventSink.emit

    async def cancelling_emit(self: EventSink, type: str, data: dict[str, Any]) -> None:
        await emit(self, type, data)
        if type == "message_end" and not data["interrupted"]:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

    monkeypatch.setattr(EventSink, "emit", cancelling_emit)
    fake.push(FakeReply("", tool_calls=[("set_title", {"title": "T"})]))
    s = await agents.create_session("note", OWNER)
    end = await (await agents.send_message(s.id, OWNER, "x")).wait()
    assert end.data["reason"] != "completed"
    branch = await agents.get_branch(s.id, OWNER)
    assert [e.type for e in branch][-2:] == ["assistant", "tool_result"]
    assert branch[-1].content["content"].startswith("Interrompu")
    assert (await agents.get_session(s.id, OWNER)).status == "idle"


async def test_disconnect_does_not_stop(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("Une longue réponse", delay=0.01))
    s = await agents.create_session("note", OWNER)
    h = await agents.send_message(s.id, OWNER, "x")
    it = h.events()
    await it.__anext__()
    await it.aclose()  # le client se déconnecte (F5)
    # Un nouvel abonné reprend le flux depuis le début du tour.
    end = await h.wait()
    assert end.data["reason"] == "completed"
    branch = await agents.get_branch(s.id, OWNER)
    assert branch[-1].content["content"][0]["text"] == "Une longue réponse"


async def test_subscribe_with_last_event_id(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("abc"))
    s = await agents.create_session("note", OWNER)
    evs = await collect((await agents.send_message(s.id, OWNER, "x")).events())
    replay: list[str] = []
    sub = agents.subscribe(s.id, OWNER, after=evs[1].id)
    async for ev in sub:
        replay.append(ev.type)
        if ev.type == "turn_end":
            break
    assert replay == types(evs)[2:]


async def test_transient_model_error_retried(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(
        FakeReply(error=ModelError("surcharge", transient=True, code="overloaded")), FakeReply("ok")
    )
    s = await agents.create_session("note", OWNER)
    assert (await (await agents.send_message(s.id, OWNER, "x")).wait()).data[
        "reason"
    ] == "completed"


async def test_fatal_model_error(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply(error=ModelError("clé invalide", transient=False, code="auth")))
    s = await agents.create_session("note", OWNER)
    end = await (await agents.send_message(s.id, OWNER, "x")).wait()
    assert end.data["reason"] == "error" and end.data["error"]["code"] == "auth"
    assert (await agents.get_session(s.id, OWNER)).status == "idle"


async def test_max_steps(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient(default=FakeReply("", tool_calls=[("set_title", {"title": "t"})]))
    a = make_agents(fake)
    a.definition("note").max_steps = 3
    await a.setup()
    s = await a.create_session("note", OWNER)
    end = await (await a.send_message(s.id, OWNER, "x")).wait()
    assert end.data["reason"] == "max_steps"
    assert len(fake.requests) == 3


async def test_expired_lease_is_recovered_as_crashed(
    make_agents: AgentsFactory, storage: Any
) -> None:
    fake = FakeModelClient([FakeReply("partiel", hang=True)])
    a1 = make_agents(fake, lock_ttl=0.3)
    await a1.setup()
    s = await a1.create_session("note", OWNER)
    await a1.send_message(s.id, OWNER, "x")
    await fake.started.wait()
    await asyncio.sleep(0.15)
    # Le processus « meurt » : la tâche disparaît sans rien écrire, le bail n'est plus renouvelé.
    for ex in list(a1._executors.values()):  # pyright: ignore[reportPrivateUsage]
        ex.cancel("abandon")
    await a1.runner.join(1)
    await asyncio.sleep(0.4)

    a2 = make_agents(FakeModelClient([FakeReply("reprise")]), lock_ttl=0.3)
    await a2.setup()
    end = await (await a2.send_message(s.id, OWNER, "encore")).wait()
    assert end.data["reason"] == "completed"
    branch = await a2.get_branch(s.id, OWNER)
    assert [e.type for e in branch] == ["root", "user", "assistant", "user", "assistant"]
    assert branch[2].interrupted and branch[2].content["content"][0]["text"] == "partiel"
    tree = await a2.get_tree(s.id, OWNER)
    assert len(tree) == 5


async def test_turn_end_hook(make_agents: AgentsFactory) -> None:
    seen: list[str] = []
    fake = FakeModelClient([FakeReply("ok")])
    a = make_agents(fake)

    @a.definition("note").on_turn_end
    async def _hook(ctx: Any, result: Any) -> None:
        seen.append(result.reason)

    await a.setup()
    s = await a.create_session("note", OWNER)
    await (await a.send_message(s.id, OWNER, "x")).wait()
    for _ in range(20):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == ["completed"]


async def test_state_is_note_type(agents: Agents[Any]) -> None:
    s = await agents.create_session("note", OWNER)
    assert isinstance((await agents.get_state(s.id, OWNER)).state, Note)


async def test_stop_and_subscribe_from_another_process(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient([FakeReply("abc", hang=True)])
    a1 = make_agents(fake)
    a2 = make_agents(FakeModelClient())
    await a1.setup()
    await a2.setup()
    s = await a1.create_session("note", OWNER)
    sub = a2.subscribe(s.id, OWNER)
    await a1.send_message(s.id, OWNER, "x")
    seen: list[str] = []
    async for ev in sub:
        seen.append(ev.type)
        if ev.type == "text_delta":
            await a2.stop(s.id, OWNER)  # LISTEN/NOTIFY → annulation dans a1
        if ev.type == "turn_end":
            assert ev.data["reason"] == "interrupted"
            break
    assert seen[0] == "turn_start" and "text_delta" in seen


async def test_ask_with_invalid_state_is_not_asked(
    agents: Agents[Any], fake: FakeModelClient
) -> None:
    fake.push(FakeReply("", tool_calls=[("ask_bad", {})]), FakeReply("pardon"))
    s = await agents.create_session("note", OWNER)
    evs = await collect((await agents.send_message(s.id, OWNER, "x")).events())
    assert "ask_user" not in types(evs)
    tr = next(e for e in evs if e.type == "tool_result")
    assert tr.data["is_error"] and "invalide" in tr.data["content"]
    assert evs[-1].data["reason"] == "completed"
    assert (await agents.get_state(s.id, OWNER)).state.score == 0


def test_tool_schema_keeps_annotated_description() -> None:
    d = AgentDefinition[None, None]("schema", model="m")

    @d.tool
    async def pick(
        ctx: ToolContext[None, None], n: Annotated[int, Field(description="Combien")] = 1
    ) -> str:
        """Choisit."""
        return str(n)

    assert pick.spec().parameters["properties"]["n"]["description"] == "Combien"

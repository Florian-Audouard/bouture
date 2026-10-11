"""Besoins d'intégration dans une application existante (D92–D105)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from bouture import (
    AgentDefinition,
    Agents,
    Data,
    EntriesContext,
    EntryView,
    FakeModelClient,
    FakeReply,
    ImportedEntry,
    ImportedPublication,
    InitContext,
    InvalidSettings,
    InvalidTarget,
    Ref,
    ResolveError,
)

from .conftest import AgentsFactory, Note, collect

OWNER = "alice"


async def chat(a: Agents[Any], fake: FakeModelClient, sid: Any, text: str, reply: str) -> None:
    fake.push(FakeReply(reply))
    end = await (await a.send_message(sid, OWNER, text)).wait()
    assert end.data["reason"] == "completed"


# -- D92 : réglages par session, contexte d'appel -----------------------------------------
async def test_model_settings_and_call_context(agents: Agents[Any], fake: FakeModelClient) -> None:
    with pytest.raises(InvalidSettings):
        await agents.create_session("note", OWNER, model_settings={"api_key": "sk-x"})
    s = await agents.create_session("note", OWNER, model_settings={"temperature": 0.2})
    assert s.model_settings == {"temperature": 0.2}
    await chat(agents, fake, s.id, "a", "b")
    req = fake.requests[-1]
    assert req.settings["temperature"] == 0.2
    assert req.context is not None
    assert (req.context.owner, req.context.session_id, req.context.kind) == (OWNER, s.id, "note")
    assert req.context.turn_id is not None

    info = await agents.set_model_settings(s.id, OWNER, {"top_p": 0.5, "seed": None})
    assert info.model_settings == {"top_p": 0.5}
    with pytest.raises(InvalidSettings):
        await agents.set_model_settings(s.id, OWNER, {"api_base": "http://x"})
    await chat(agents, fake, s.id, "c", "d")
    assert "temperature" not in fake.requests[-1].settings
    clone = await agents.clone(s.id, OWNER)
    assert clone.model_settings == {"top_p": 0.5}


# -- D93 : aperçu de requête --------------------------------------------------------------
async def test_preview_does_not_run_a_turn(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    await chat(agents, fake, s.id, "bonjour", "salut")
    calls, tree = len(fake.requests), await agents.get_tree(s.id, OWNER)

    p = await agents.preview(s.id, OWNER, "et maintenant ?")
    assert p.model == "fake" and p.input_tokens and p.input_tokens > 0
    assert p.system and "Tu édites une note." in p.system
    assert p.messages is not None
    assert "et maintenant ?" in "".join(m.model_dump_json() for m in p.messages[-2:])
    assert p.tools and {t["name"] for t in p.tools} >= {"set_title"}
    assert len(fake.requests) == calls
    assert await agents.get_tree(s.id, OWNER) == tree

    hidden = await agents.preview(s.id, OWNER, expose_prompt=False)
    assert hidden.system is None and hidden.messages is None and hidden.tools is None
    assert hidden.input_tokens and hidden.input_tokens < p.input_tokens

    # Ce que l'aperçu montre est ce que le tour enverra.
    await chat(agents, fake, s.id, "et maintenant ?", "voilà")
    assert fake.requests[-1].system == p.system
    assert fake.requests[-1].messages == p.messages


# -- D94 : sauvegarde automatique ---------------------------------------------------------
async def test_autosave_amends_last_state_edit(agents: Agents[Any]) -> None:
    s = await agents.create_session("note", OWNER)
    first = await agents.edit_state(s.id, OWNER, {"title": "D"}, amend=True)
    for title in ("Dr", "Dra", "Dragons"):
        e = await agents.edit_state(s.id, OWNER, {"title": title}, amend=True)
        assert e.id == first.id
    assert e.content["patch"] == [{"op": "replace", "path": "/title", "value": "Dragons"}]
    assert [x.type for x in await agents.get_branch(s.id, OWNER)] == ["root", "state_edit"]
    assert (await agents.get_state(s.id, OWNER)).state.title == "Dragons"

    # Revenir à l'état du parent réutilise son snapshot.
    back = await agents.edit_state(s.id, OWNER, {}, amend=True)
    root = (await agents.get_branch(s.id, OWNER))[0]
    assert back.id == first.id and back.state_id == root.state_id and back.content["patch"] == []

    # Un point de sauvegarde nommé ou publié n'est jamais réécrit.
    await agents.edit_state(s.id, OWNER, {"title": "v1"}, amend=True)
    await agents.label_entry(s.id, OWNER, first.id, "Avant refonte")
    second = await agents.edit_state(s.id, OWNER, {"title": "v2"}, amend=True)
    assert second.id != first.id
    await agents.publish(s.id, OWNER)
    third = await agents.edit_state(s.id, OWNER, {"title": "v3"}, amend=True)
    assert third.id != second.id
    assert (await agents.get_state(s.id, OWNER, entry_id=second.id)).state.title == "v2"
    # amend=False garde le comportement historique.
    assert (await agents.edit_state(s.id, OWNER, {"title": "v4"})).id != third.id


# -- D95 : points de sauvegarde nommés -----------------------------------------------------
async def test_labels(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    await chat(agents, fake, s.id, "a", "b")
    branch = await agents.get_branch(s.id, OWNER)
    with pytest.raises(InvalidTarget):
        await agents.label_entry(s.id, OWNER, branch[1].id, "pas un checkpoint")
    v = await agents.label_entry(s.id, OWNER, branch[2].id, "  Version 1  ")
    assert v.label == "Version 1"
    labeled = await agents.get_tree(s.id, OWNER, labeled=True)
    assert [e.id for e in labeled] == [branch[2].id]
    assert (await agents.label_entry(s.id, OWNER, branch[2].id, "")).label is None
    assert await agents.get_tree(s.id, OWNER, labeled=True) == []


# -- D96 : messages sans appel LLM --------------------------------------------------------
async def test_initial_messages_and_append(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient()
    a = make_agents(fake)
    greeter = AgentDefinition[Note, Any]("greeter", model="fake", state_type=Note)

    @greeter.initial_messages
    async def _greet(ctx: InitContext[Note, Any]) -> list[tuple[str, Any]]:
        assert ctx.owner == OWNER and ctx.state is not None
        return [("assistant", f"Bonjour {ctx.owner} !")]

    a.register(greeter)
    await a.setup()
    s = await a.create_session("greeter", OWNER)
    branch = await a.get_branch(s.id, OWNER)
    assert [e.type for e in branch] == ["root", "assistant"]
    assert branch[1].is_checkpoint and s.leaf_entry_id == branch[1].id
    assert fake.requests == []

    await chat(a, fake, s.id, "salut", "re")
    assert "Bonjour alice !" in fake.requests[-1].messages[0].model_dump_json()

    e = await a.append_message(s.id, OWNER, "assistant", "Une note du narrateur.")
    assert e.type == "assistant" and e.is_checkpoint
    assert (await a.get_session(s.id, OWNER)).leaf_entry_id == e.id
    u = await a.append_message(s.id, OWNER, "user", [{"type": "text", "text": "ok"}])
    assert u.parent_id == e.id
    user_entry = (await a.get_branch(s.id, OWNER))[2]
    with pytest.raises(InvalidTarget):
        await a.append_message(s.id, OWNER, "user", "x", parent_entry_id=user_entry.id)
    with pytest.raises(InvalidTarget):
        await a.append_message(s.id, OWNER, "assistant", [{"type": "image", "ref": "http://i"}])
    # Sur un checkpoint ancien : nouvelle branche.
    side = await a.append_message(s.id, OWNER, "user", "autre", parent_entry_id=branch[1].id)
    assert side.parent_id == branch[1].id


async def test_append_cancels_pending_question(agents: Agents[Any], fake: FakeModelClient) -> None:
    fake.push(FakeReply("", tool_calls=[("ask_tag", {"proposal": "x"})]))
    s = await agents.create_session("note", OWNER)
    await (await agents.send_message(s.id, OWNER, "tag ?")).wait()
    assert (await agents.get_session(s.id, OWNER)).status == "awaiting_input"
    await agents.append_message(s.id, OWNER, "assistant", "Laissons ça.")
    info = await agents.get_session(s.id, OWNER)
    assert info.status == "idle" and info.pending == []
    types_ = [e.type for e in await agents.get_branch(s.id, OWNER)]
    assert types_[-2:] == ["tool_result", "assistant"]


# -- D97 : éditer un message sans régénérer ------------------------------------------------
async def test_edit_entry_keeps_original(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    fake.push(FakeReply("", tool_calls=[("set_title", {"title": "T"})]), FakeReply("Fait."))
    await (await agents.send_message(s.id, OWNER, "titre")).wait()
    await chat(agents, fake, s.id, "encore", "oui")
    old = await agents.get_branch(s.id, OWNER)
    calls = len(fake.requests)

    first_user = old[1]
    e = await agents.edit_entry(s.id, OWNER, first_user.id, "titre, stp")
    assert e.derived_from == first_user.id and e.parent_id == first_user.parent_id
    assert "titre, stp" in json.dumps(e.content)
    new = await agents.get_branch(s.id, OWNER)
    assert [x.type for x in new] == [x.type for x in old]
    assert [x.state_id for x in new] == [x.state_id for x in old]
    assert new[1].id == e.id and new[-1].is_checkpoint
    assert len(fake.requests) == calls
    sib = await agents.siblings(s.id, OWNER, e.id)
    assert {x.id for x in sib.entries} == {first_user.id, e.id}

    # Assistant : seul le texte change, l'appel d'outil reste.
    assistant = new[2]
    a2 = await agents.edit_entry(s.id, OWNER, assistant.id, "Je change le titre.")
    blocks = a2.content["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_call"]
    assert blocks[0]["text"] == "Je change le titre."

    # Hors du chemin actif : la suite est recopiée jusqu'au dernier checkpoint.
    original = await agents.edit_entry(s.id, OWNER, old[-2].id, "encore, édité")
    branch = await agents.get_branch(s.id, OWNER)
    assert branch[-2].id == original.id and branch[-1].type == "assistant"
    assert branch[-3].id == old[-3].id

    root = old[0]
    with pytest.raises(InvalidTarget):
        await agents.edit_entry(s.id, OWNER, root.id, "x")


# -- D98 : clone depuis un point choisi ----------------------------------------------------
async def test_clone_from_entry(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    await chat(agents, fake, s.id, "un", "1")
    await chat(agents, fake, s.id, "deux", "2")
    old = await agents.get_branch(s.id, OWNER)
    c = await agents.clone(s.id, OWNER, entry_id=old[2].id)
    assert [e.content for e in await agents.get_branch(c.id, OWNER)] == [e.content for e in old[:3]]
    # Un message utilisateur : coupé juste après, pour régénérer dans la copie.
    c2 = await agents.clone(s.id, OWNER, entry_id=old[3].id)
    b2 = await agents.get_branch(c2.id, OWNER)
    assert [e.type for e in b2] == ["root", "user", "assistant", "user"] and b2[-1].is_checkpoint
    fake.push(FakeReply("2 bis"))
    await (await agents.regenerate(c2.id, OWNER, b2[-1].id)).wait()
    assert (await agents.get_branch(s.id, OWNER))[-1].id == old[-1].id


# -- D99 : changer les références ---------------------------------------------------------
async def test_update_inputs(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient()
    a = make_agents(fake)
    a.register(AgentDefinition[None, Any]("reader", model="fake", inputs={"note": "note"}))

    @a.resolver("note")
    async def _resolve(inp: Any, ctx: Any) -> str:
        return f"Note : {(await a.get_published_state(inp)).title}"

    await a.setup()
    n = await a.create_session("note", OWNER)
    await a.edit_state(n.id, OWNER, {"title": "Atlas"})
    ref = await a.publish(n.id, OWNER)
    await a.edit_state(n.id, OWNER, {"title": "Atlas 2"})
    await a.publish(n.id, OWNER)

    r = await a.create_session("reader", OWNER, inputs={"note": Ref("note", ref.object_id, 1)})
    await chat(a, fake, r.id, "lis", "ok")
    before = (await a.get_branch(r.id, OWNER))[-1]
    with pytest.raises(ResolveError):
        await a.update_inputs(r.id, OWNER, {"autre": Data("brief", {})})
    e = await a.update_inputs(r.id, OWNER, {"note": Ref("note", ref.object_id)})
    assert e.type == "input_update" and e.is_checkpoint
    assert (await a.get_inputs(r.id, OWNER))["note"].version == 2
    assert (await a.get_inputs(r.id, OWNER, entry_id=before.id))["note"].version == 1

    await chat(a, fake, r.id, "relis", "ok")
    req = fake.requests[-1]
    assert "Note : Atlas 2" in req.system
    assert "Les références suivantes ont changé : note." in str(req.messages)

    await a.rollback(r.id, OWNER, before.id)
    await chat(a, fake, r.id, "et là ?", "ok")
    assert "Note : Atlas\n" in fake.requests[-1].system + "\n"


# -- D100 : corbeille ---------------------------------------------------------------------
async def test_hide_and_restore_branch(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    await chat(agents, fake, s.id, "a", "b")
    old = await agents.get_branch(s.id, OWNER)
    fake.push(FakeReply("b2"))
    await (await agents.fork(s.id, OWNER, old[1].id, "a2")).wait()
    active = await agents.get_branch(s.id, OWNER)

    with pytest.raises(InvalidTarget):
        await agents.hide_branch(s.id, OWNER, active[1].id)
    with pytest.raises(InvalidTarget):
        await agents.hide_branch(s.id, OWNER, old[0].id)
    hidden = await agents.hide_branch(s.id, OWNER, old[1].id)
    assert set(hidden) == {old[1].id, old[2].id}
    tree_ids = {e.id for e in await agents.get_tree(s.id, OWNER)}
    assert old[1].id not in tree_ids
    full = await agents.get_tree(s.id, OWNER, include_hidden=True)
    assert all(e.hidden_at is not None for e in full if e.id in hidden)
    assert len((await agents.siblings(s.id, OWNER, active[1].id)).entries) == 1
    with pytest.raises(InvalidTarget):
        await agents.switch_branch(s.id, OWNER, old[2].id)
    with pytest.raises(InvalidTarget):
        await agents.regenerate(s.id, OWNER, old[1].id)

    # Restaurer l'enfant seul est refusé tant que le parent est masqué.
    with pytest.raises(InvalidTarget):
        await agents.restore_branch(s.id, OWNER, old[2].id)
    assert set(await agents.restore_branch(s.id, OWNER, old[1].id)) == set(hidden)
    assert len((await agents.siblings(s.id, OWNER, active[1].id)).entries) == 2
    await agents.switch_branch(s.id, OWNER, old[2].id)


# -- D101 : métadonnées d'entrée ----------------------------------------------------------
async def test_entry_metadata(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient()
    a = make_agents(fake)
    tagged = AgentDefinition[Note, Any]("tagged", model="fake", state_type=Note)
    seen: list[str] = []

    @tagged.on_entries
    async def _meta(
        ctx: EntriesContext[Any], entries: list[EntryView]
    ) -> dict[Any, dict[str, Any]]:
        assert ctx.owner == OWNER
        seen.extend(e.type for e in entries)
        return {e.id: {"app_message_id": f"m-{e.type}"} for e in entries if e.type == "assistant"}

    a.register(tagged)
    await a.setup()
    s = await a.create_session("tagged", OWNER)
    fake.push(FakeReply("hi"))
    evs = await collect((await a.send_message(s.id, OWNER, "yo")).events())
    assert seen == ["user", "assistant"]
    meta = next(e for e in evs if e.type == "entry_metadata")
    assert evs.index(meta) == len(evs) - 2
    last = (await a.get_branch(s.id, OWNER))[-1]
    assert meta.data["entries"] == {str(last.id): {"app_message_id": "m-assistant"}}
    assert last.metadata == {"app_message_id": "m-assistant"}

    v = await a.set_entry_metadata(s.id, OWNER, last.id, {"liked": True, "app_message_id": None})
    assert v.metadata == {"liked": True}
    assert (await a.get_branch(s.id, OWNER))[-1].metadata == {"liked": True}


async def test_entry_metadata_hook_failure_is_ignored(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient()
    a = make_agents(fake)
    broken = AgentDefinition[Note, Any]("broken", model="fake", state_type=Note)

    @broken.on_entries
    async def _meta(ctx: EntriesContext[Any], entries: list[EntryView]) -> dict[Any, Any]:
        raise RuntimeError("panne")

    a.register(broken)
    await a.setup()
    s = await a.create_session("broken", OWNER)
    await chat(a, fake, s.id, "a", "b")


# -- D102 : retrouver une session ---------------------------------------------------------
async def test_list_sessions_by_object_and_metadata(agents: Agents[Any]) -> None:
    s1 = await agents.create_session("note", OWNER, metadata={"app": {"character": "c1"}, "x": 1})
    s2 = await agents.create_session("note", OWNER, metadata={"app": {"character": "c2"}})
    await agents.edit_state(s1.id, OWNER, {"title": "A"})
    ref = await agents.publish(s1.id, OWNER)
    page = await agents.list_sessions(OWNER, object_id=ref.object_id)
    assert [s.id for s in page.items] == [s1.id]
    page = await agents.list_sessions(OWNER, metadata={"app": {"character": "c2"}})
    assert [s.id for s in page.items] == [s2.id]
    assert (await agents.list_sessions("bob", metadata={"x": 1})).items == []


# -- D104 : import d'historique -----------------------------------------------------------
async def test_import_session(agents: Agents[Any], fake: FakeModelClient) -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    u1, a1, e1, u2, a2 = (uuid4() for _ in range(5))
    entries = [
        ImportedEntry(u1, None, "user", t0, content="Crée une note", checkpoint=False),
        ImportedEntry(a1, u1, "assistant", t0 + timedelta(seconds=1), content="Voilà"),
        ImportedEntry(e1, a1, "state_edit", t0 + timedelta(seconds=2), state={"title": "Vieux"}),
        ImportedEntry(u2, a1, "user", t0 + timedelta(seconds=3), content="Autre piste"),
        ImportedEntry(
            a2, u2, "assistant", t0 + timedelta(seconds=4), content="Ok", metadata={"legacy": 7}
        ),
    ]
    oid = uuid4()
    s = await agents.import_session(
        OWNER,
        "note",
        entries=entries,
        leaf_id=e1,
        title="Importée",
        object_id=oid,
        publications=[ImportedPublication(e1, 3, t0 + timedelta(seconds=5))],
    )
    assert s.leaf_entry_id == e1 and s.object_id == oid and s.created_at == t0
    assert fake.requests == []
    branch = await agents.get_branch(s.id, OWNER)
    assert [e.id for e in branch[1:]] == [u1, a1, e1]
    assert branch[2].created_at == t0 + timedelta(seconds=1)
    assert (await agents.get_state(s.id, OWNER)).state.title == "Vieux"
    pubs = await agents.list_publications(OWNER, "note", oid)
    assert [p.version for p in pubs] == [3]
    assert (await agents.get_published_state(Ref("note", oid))).title == "Vieux"
    sib = await agents.siblings(s.id, OWNER, u2)
    assert [e.id for e in sib.entries] == [e1, u2]
    assert (await agents.get_branch(s.id, OWNER, entry_id=a2))[-1].metadata == {"legacy": 7}

    # La session importée est une session comme une autre.
    await chat(agents, fake, s.id, "suite", "ok")
    await agents.edit_state(s.id, OWNER, {"title": "Neuf"})
    assert (await agents.publish(s.id, OWNER)).version == 4

    with pytest.raises(InvalidTarget):
        await agents.import_session(OWNER, "note", entries=[entries[1]])
    with pytest.raises(InvalidTarget):
        await agents.import_session(OWNER, "note", entries=entries[:1])
    with pytest.raises(InvalidTarget):
        await agents.import_session(
            OWNER,
            "note",
            entries=entries[:3],
            object_id=oid,
            publications=[ImportedPublication(e1, 1, t0)],
        )


# -- D105 : streaming des arguments d'outil -----------------------------------------------
async def test_tool_call_delta_is_opt_in(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient()
    a = make_agents(fake)
    streamed = AgentDefinition[Note, Any]("streamed", model="fake", state_type=Note)
    streamed.stream_tool_args = True

    @streamed.tool
    async def set_title(ctx: Any, title: str) -> str:
        """Change le titre."""
        ctx.state.title = title
        return "ok"

    a.register(streamed)
    await a.setup()
    args = {"title": "Un long titre streamé"}
    for kind, expected in (("streamed", True), ("note", False)):
        s = await a.create_session(kind, OWNER)
        fake.push(
            FakeReply("", tool_calls=[("set_title", args)], stream_args=True), FakeReply("ok")
        )
        evs = await collect((await a.send_message(s.id, OWNER, "titre")).events())
        deltas = [e for e in evs if e.type == "tool_call_delta"]
        assert bool(deltas) is expected
        if expected:
            assert json.loads("".join(d.data["args_delta"] for d in deltas)) == args
            assert deltas[0].data["name"] == "set_title"
            assert deltas[0].data["tool_call_id"] == "call_1_0" or deltas[0].data["tool_call_id"]
            done = next(e for e in evs if e.type == "tool_call")
            assert evs.index(done) > evs.index(deltas[-1])
        assert (await a.get_state(s.id, OWNER)).state.title == args["title"]

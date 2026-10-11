from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from bouture import (
    AgentDefinition,
    Agents,
    Data,
    FakeModelClient,
    FakeReply,
    InvalidTarget,
    NotFound,
    Ref,
    ResolveError,
    StateValidationError,
    StateVersionError,
)

from .conftest import AgentsFactory, collect, types

OWNER = "alice"


async def chat(a: Agents[Any], fake: FakeModelClient, sid: Any, text: str, reply: str) -> None:
    fake.push(FakeReply(reply))
    end = await (await a.send_message(sid, OWNER, text)).wait()
    assert end.data["reason"] == "completed"


async def test_rollback(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    fake.push(FakeReply("", tool_calls=[("set_title", {"title": "A"})]), FakeReply("ok"))
    await (await agents.send_message(s.id, OWNER, "titre A")).wait()
    first = await agents.get_branch(s.id, OWNER)
    fake.push(FakeReply("", tool_calls=[("set_title", {"title": "B"})]), FakeReply("ok"))
    await (await agents.send_message(s.id, OWNER, "titre B")).wait()
    assert (await agents.get_state(s.id, OWNER)).state.title == "B"

    with pytest.raises(InvalidTarget):
        await agents.rollback(s.id, OWNER, first[1].id)  # message user : pas un checkpoint
    info = await agents.rollback(s.id, OWNER, first[-1].id)
    assert info.leaf_entry_id == first[-1].id
    assert (await agents.get_state(s.id, OWNER)).state.title == "A"
    # Rien n'est supprimé.
    assert len(await agents.get_tree(s.id, OWNER)) == 9
    # La suite repart de la feuille déplacée.
    await chat(agents, fake, s.id, "suite", "ok")
    branch = await agents.get_branch(s.id, OWNER)
    assert [e.type for e in branch][-2:] == ["user", "assistant"]
    assert branch[-3].id == first[-1].id
    # Le contexte envoyé ne contient pas la branche abandonnée.
    texts = [m.content[0].text for m in fake.requests[-1].messages if m.role == "user"]  # type: ignore[union-attr]
    assert "titre B" not in texts


async def test_fork_and_siblings_and_switch(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    await chat(agents, fake, s.id, "question 1", "réponse 1")
    await chat(agents, fake, s.id, "question 2", "réponse 2")
    original = await agents.get_branch(s.id, OWNER)
    q2 = original[3]
    assert q2.type == "user"

    fake.push(FakeReply("réponse 2bis"))
    evs = await collect((await agents.fork(s.id, OWNER, q2.id, "question 2 éditée")).events())
    assert evs[0].data["trigger"] == "fork"
    branch = await agents.get_branch(s.id, OWNER)
    assert branch[3].content["content"][0]["text"] == "question 2 éditée"
    assert branch[3].parent_id == q2.parent_id

    sibs = await agents.siblings(s.id, OWNER, branch[3].id)
    assert [e.id for e in sibs.entries] == [q2.id, branch[3].id] and sibs.index == 1

    with pytest.raises(InvalidTarget):
        await agents.fork(s.id, OWNER, branch[2].id, "pas un message user")

    info = await agents.switch_branch(s.id, OWNER, q2.id)
    assert info.leaf_entry_id == original[-1].id


async def test_regenerate(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    await chat(agents, fake, s.id, "question", "réponse A")
    branch = await agents.get_branch(s.id, OWNER)
    fake.push(FakeReply("réponse B"))
    await (await agents.regenerate(s.id, OWNER, branch[-1].id)).wait()
    new = await agents.get_branch(s.id, OWNER)
    assert new[1].id == branch[1].id  # même message user
    assert new[2].id != branch[2].id
    assert new[2].content["content"][0]["text"] == "réponse B"
    sibs = await agents.siblings(s.id, OWNER, new[2].id)
    assert len(sibs.entries) == 2
    # Le message user n'est pas dupliqué dans le contexte.
    users = [m for m in fake.requests[-1].messages if m.role == "user"]
    assert len(users) == 2  # la question + l'état courant


async def test_clone_copies_active_branch_only(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER, title="Original")
    fake.push(FakeReply("", tool_calls=[("set_title", {"title": "A"})]), FakeReply("ok"))
    await (await agents.send_message(s.id, OWNER, "x")).wait()
    b1 = await agents.get_branch(s.id, OWNER)
    await chat(agents, fake, s.id, "y", "ok")
    await agents.rollback(s.id, OWNER, b1[-1].id)
    await chat(agents, fake, s.id, "z", "ok")

    c = await agents.clone(s.id, OWNER, title="Copie")
    assert c.id != s.id and c.title == "Copie" and c.cloned_from_session_id == s.id
    tree = await agents.get_tree(c.id, OWNER)
    assert len(tree) == len(await agents.get_branch(s.id, OWNER))
    assert {e.id for e in tree}.isdisjoint({e.id for e in await agents.get_tree(s.id, OWNER)})
    assert (await agents.get_state(c.id, OWNER)).state.title == "A"
    # Indépendance : un tour sur le clone ne touche pas l'original.
    await chat(agents, fake, c.id, "sur le clone", "ok")
    assert len(await agents.get_tree(s.id, OWNER)) == 9


async def test_edit_state(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    e = await agents.edit_state(s.id, OWNER, {"title": "Manuel", "tags": ["a"]})
    assert e.type == "state_edit" and e.is_checkpoint
    assert {"op": "replace", "path": "/title", "value": "Manuel"} in e.content["patch"]
    assert (await agents.get_state(s.id, OWNER)).state.title == "Manuel"
    with pytest.raises(StateValidationError):
        await agents.edit_state(s.id, OWNER, {"score": -1})
    # Sans changement : pas de nouvelle entrée.
    same = await agents.edit_state(s.id, OWNER, {"title": "Manuel", "tags": ["a"]})
    assert same.id == e.id
    # L'agent voit la note d'édition et le nouvel état.
    await chat(agents, fake, s.id, "tu vois ?", "oui")
    texts = [
        m.content[0].text
        for m in fake.requests[-1].messages
        if m.role == "user"  # type: ignore[union-attr]
    ]
    assert any("modifié l'état" in t for t in texts)
    assert "Manuel" in texts[-1]
    # Rollback avant l'édition.
    root = (await agents.get_branch(s.id, OWNER))[0]
    await agents.rollback(s.id, OWNER, root.id)
    assert (await agents.get_state(s.id, OWNER)).state.title == ""


async def test_edit_state_cancels_pending(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    fake.push(FakeReply("", tool_calls=[("ask_tag", {"proposal": "x"})]))
    await (await agents.send_message(s.id, OWNER, "x")).wait()
    e = await agents.edit_state(s.id, OWNER, {"title": "brouillon", "score": 2})
    branch = await agents.get_branch(s.id, OWNER)
    assert [x.type for x in branch][-2:] == ["tool_result", "state_edit"]
    assert branch[-1].id == e.id
    assert (await agents.get_session(s.id, OWNER)).status == "idle"


async def test_publish_and_idempotence(agents: Agents[Any], fake: FakeModelClient) -> None:
    published: list[Any] = []

    @agents.definition("note").on_publish
    async def _on_publish(ctx: Any, ref: Any, state: Any) -> None:
        published.append((ref.version, state.title))

    s = await agents.create_session("note", OWNER)
    await agents.edit_state(s.id, OWNER, {"title": "v1"})
    r1 = await agents.publish(s.id, OWNER)
    assert r1.version == 1
    assert (await agents.publish(s.id, OWNER)) == r1  # même état → même version
    await agents.edit_state(s.id, OWNER, {"title": "v2"})
    r2 = await agents.publish(s.id, OWNER)
    assert r2.version == 2 and r2.object_id == r1.object_id
    assert published == [(1, "v1"), (2, "v2")]
    assert (await agents.get_published_state(r1.as_ref())).title == "v1"
    assert (await agents.get_published_state(Ref("note", r1.object_id))).title == "v2"
    pubs = await agents.list_publications(OWNER, "note", r1.object_id)
    assert [p.version for p in pubs] == [1, 2]
    assert await agents.list_publications("bob", "note", r1.object_id) == []
    with pytest.raises(NotFound):
        await agents.get_published_state(r1, owner="bob")


async def test_on_publish_failure_rolls_back(agents: Agents[Any]) -> None:
    @agents.definition("note").on_publish
    async def _fail(ctx: Any, ref: Any, state: Any) -> None:
        raise RuntimeError("table projet indisponible")

    s = await agents.create_session("note", OWNER)
    with pytest.raises(RuntimeError):
        await agents.publish(s.id, OWNER)
    info = await agents.get_session(s.id, OWNER)
    assert info.object_id is None


async def test_publish_refused_while_awaiting(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    fake.push(FakeReply("", tool_calls=[("ask_tag", {"proposal": "x"})]))
    await (await agents.send_message(s.id, OWNER, "x")).wait()
    with pytest.raises(InvalidTarget):
        await agents.publish(s.id, OWNER)


async def wait_for(cond: Any, timeout: float = 2.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if await cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition jamais vraie")


async def test_child_session_result_applied(agents: Agents[Any], fake: FakeModelClient) -> None:
    s = await agents.create_session("note", OWNER)
    fake.push(FakeReply("", tool_calls=[("make_image", {"prompt": "un chat"})]), FakeReply("lancé"))
    evs = await collect((await agents.send_message(s.id, OWNER, "image")).events())
    spawned = next(e for e in evs if e.type == "child_spawned")
    child_id = spawned.data["session_id"]
    children = (await agents.list_sessions(OWNER, origin_session_id=s.id)).items
    assert [str(c.id) for c in children] == [child_id]

    await agents.edit_state(children[0].id, OWNER, {"url": "https://img/1.png"})
    await agents.publish(children[0].id, OWNER)
    # Parent inactif : un mini-tour applique le résultat.

    async def applied() -> bool:
        return (await agents.get_state(s.id, OWNER)).state.image == "https://img/1.png"

    await wait_for(applied)
    branch = await agents.get_branch(s.id, OWNER)
    assert branch[-1].type == "child_result" and branch[-1].is_checkpoint
    assert branch[-1].content["error"] is None


async def test_child_result_queued_while_parent_busy(
    agents: Agents[Any], fake: FakeModelClient
) -> None:
    s = await agents.create_session("note", OWNER)
    fake.push(FakeReply("", tool_calls=[("make_image", {"prompt": "x"})]), FakeReply("ok"))
    await (await agents.send_message(s.id, OWNER, "image")).wait()
    child = (await agents.list_sessions(OWNER, origin_session_id=s.id)).items[0]
    fake.started.clear()
    fake.push(FakeReply("lent", hang=True))
    h = await agents.send_message(s.id, OWNER, "parle")
    await fake.started.wait()
    await asyncio.sleep(0.05)
    await agents.edit_state(child.id, OWNER, {"url": "u"})
    await agents.publish(child.id, OWNER)
    assert (await agents.get_state(s.id, OWNER)).state.image is None
    await agents.stop(s.id, OWNER)
    await h.wait()

    async def applied() -> bool:
        return (await agents.get_state(s.id, OWNER)).state.image == "u"

    await wait_for(applied)
    branch = await agents.get_branch(s.id, OWNER)
    assert [e.type for e in branch][-3:] == ["user", "assistant", "child_result"], [
        (e.type, e.content) for e in branch
    ]


class WorldRef(BaseModel):
    name: str


async def test_inputs_and_resolver(make_agents: AgentsFactory) -> None:
    fake = FakeModelClient()
    a = make_agents(fake)
    reader = AgentDefinition[None, Any](
        "reader", model="fake", inputs={"note": "note", "brief": "brief"}
    )
    a.register(reader)
    a.data_type("brief", WorldRef)

    @a.resolver("note")
    async def _resolve(inp: Any, ctx: Any) -> str:
        st = await a.get_published_state(inp)
        return f"Note publiée : {st.title}"

    await a.setup()
    n = await a.create_session("note", OWNER)
    await a.edit_state(n.id, OWNER, {"title": "Atlas"})
    ref = await a.publish(n.id, OWNER)
    with pytest.raises(ResolveError):
        await a.create_session("reader", OWNER, inputs={"note": Ref("note", ref.object_id)})
    with pytest.raises(ResolveError):
        await a.create_session(
            "reader",
            OWNER,
            inputs={"note": Ref("note", ref.object_id), "brief": Data("brief", {"nom": 1})},
        )
    r = await a.create_session(
        "reader",
        OWNER,
        inputs={"note": Ref("note", ref.object_id), "brief": Data("brief", {"name": "Brief"})},
    )
    # Une publication ultérieure ne change pas l'input figé à la création.
    await a.edit_state(n.id, OWNER, {"title": "Atlas 2"})
    await a.publish(n.id, OWNER)
    fake.push(FakeReply("lu"))
    await (await a.send_message(r.id, OWNER, "lis")).wait()
    system = fake.requests[-1].system
    assert "Note publiée : Atlas" in system and "Atlas 2" not in system
    assert '"name": "Brief"' in system
    # Pas d'état pour ce kind : pas de bloc <etat_courant>.
    assert all("<etat_courant>" not in str(m) for m in fake.requests[-1].messages)


async def test_state_migration(make_agents: AgentsFactory, storage: Any) -> None:
    class V1(BaseModel):
        name: str = ""

    class V2(BaseModel):
        title: str = ""

    d1 = AgentDefinition[V1, Any]("doc", model="fake", state_type=V1)
    a1 = make_agents(FakeModelClient())
    a1.register(d1)
    await a1.setup()
    s = await a1.create_session("doc", OWNER)
    await a1.edit_state(s.id, OWNER, {"name": "ancien"})

    d2 = AgentDefinition[V2, Any](
        "doc",
        model="fake",
        state_type=V2,
        state_version=2,
        state_migrations={1: lambda d: {"title": d["name"]}},
    )
    a2 = make_agents(FakeModelClient())
    a2.register(d2)
    await a2.setup()
    st = await a2.get_state(s.id, OWNER)
    assert st.state.title == "ancien" and st.state_version == 2

    # Version stockée plus récente que le code → erreur explicite.
    await a2.edit_state(s.id, OWNER, {"title": "nouveau"})
    with pytest.raises(StateVersionError):
        await a1.get_state(s.id, OWNER)


async def test_list_sessions_pagination_and_archive(agents: Agents[Any]) -> None:
    ids = [(await agents.create_session("note", OWNER, title=f"n{i}")).id for i in range(5)]
    page1 = await agents.list_sessions(OWNER, limit=2)
    page2 = await agents.list_sessions(OWNER, limit=2, cursor=page1.next_cursor)
    page3 = await agents.list_sessions(OWNER, limit=2, cursor=page2.next_cursor)
    got = [s.id for s in page1.items + page2.items + page3.items]
    assert sorted(got) == sorted(ids) and page3.next_cursor is None
    await agents.archive(ids[0], OWNER)
    assert ids[0] not in [s.id for s in (await agents.list_sessions(OWNER)).items]
    assert [s.id for s in (await agents.list_sessions(OWNER, archived=True)).items] == [ids[0]]
    info = await agents.rename(ids[1], OWNER, "renommée")
    assert info.title == "renommée"
    info = await agents.set_model(ids[1], OWNER, "autre-modele")
    assert info.model == "autre-modele"
    info = await agents.update_metadata(ids[1], OWNER, {"color": "red"})
    assert info.metadata == {"color": "red"}


async def test_purge(agents: Agents[Any], fake: FakeModelClient) -> None:
    purged: list[Any] = []

    @agents.on_purge
    async def _hook(ctx: Any, owner: str, ids: list[Any]) -> None:
        purged.append((owner, ids))

    s1 = await agents.create_session("note", OWNER)
    await chat(agents, fake, s1.id, "x", "y")
    s2 = await agents.create_session("note", "bob")
    await agents.purge_owner(OWNER)
    assert purged == [(OWNER, [s1.id])]
    with pytest.raises(NotFound):
        await agents.get_session(s1.id, OWNER)
    assert (await agents.get_session(s2.id, "bob")).id == s2.id


def test_types_helper() -> None:
    assert types([]) == []

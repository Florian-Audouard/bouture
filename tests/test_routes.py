from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Header

from bouture import Agents, FakeModelClient, FakeReply
from bouture.fastapi import make_router


def parse_sse(text: str) -> list[dict[str, Any]]:
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        ev = json.loads(fields["data"])
        assert fields["event"] == ev["type"] and int(fields["id"]) == ev["id"]
        out.append(ev)
    return out


@pytest.fixture
async def client(agents: Agents[Any]) -> AsyncIterator[httpx.AsyncClient]:
    async def current_user(x_user: str = Header()) -> str:
        return x_user

    app = FastAPI()
    app.include_router(make_router(agents, current_user), prefix="/agents")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


ALICE = {"x-user": "alice"}
BOB = {"x-user": "bob"}


async def test_routes_flow(client: httpx.AsyncClient, fake: FakeModelClient) -> None:
    r = await client.post("/agents/sessions", json={"kind": "note", "title": "N"}, headers=ALICE)
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    fake.push(FakeReply("", tool_calls=[("set_title", {"title": "Route"})]), FakeReply("ok"))
    r = await client.post(
        f"/agents/sessions/{sid}/messages", json={"content": "hey"}, headers=ALICE
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = parse_sse(r.text)
    assert evs[-1]["type"] == "turn_end" and evs[-1]["data"]["reason"] == "completed"
    assert r.headers["x-turn-id"] == evs[0]["turn_id"]

    r = await client.get(f"/agents/sessions/{sid}/state", headers=ALICE)
    assert r.json()["state"]["title"] == "Route"
    r = await client.put(
        f"/agents/sessions/{sid}/state", json={"state": {"score": -1}}, headers=ALICE
    )
    assert r.status_code == 422
    r = await client.put(
        f"/agents/sessions/{sid}/state", json={"state": {"title": "Main"}}, headers=ALICE
    )
    assert r.status_code == 200 and r.json()["type"] == "state_edit"

    branch = (await client.get(f"/agents/sessions/{sid}/branch", headers=ALICE)).json()
    user_entry = branch[1]["id"]
    r = await client.post(
        f"/agents/sessions/{sid}/rollback", json={"entry_id": user_entry}, headers=ALICE
    )
    assert r.status_code == 409
    r = await client.post(
        f"/agents/sessions/{sid}/rollback", json={"entry_id": branch[0]["id"]}, headers=ALICE
    )
    assert r.status_code == 200 and r.json()["leaf_entry_id"] == branch[0]["id"]

    fake.push(FakeReply("bis"))
    r = await client.post(
        f"/agents/sessions/{sid}/entries/{user_entry}/fork",
        json={"content": "édité"},
        headers=ALICE,
    )
    assert parse_sse(r.text)[-1]["data"]["reason"] == "completed"
    sibs = (
        await client.get(f"/agents/sessions/{sid}/entries/{user_entry}/siblings", headers=ALICE)
    ).json()
    assert len(sibs["entries"]) == 2

    r = await client.post(f"/agents/sessions/{sid}/publish", headers=ALICE)
    assert r.status_code == 200 and r.json()["version"] == 1
    obj = r.json()["object_id"]
    r = await client.get(f"/agents/publications/note/{obj}", headers=ALICE)
    assert [p["version"] for p in r.json()] == [1]

    r = await client.patch(f"/agents/sessions/{sid}", json={"title": "Renommée"}, headers=ALICE)
    assert r.json()["title"] == "Renommée"
    r = await client.post(f"/agents/sessions/{sid}/clone", headers=ALICE)
    assert r.status_code == 201
    r = await client.get("/agents/sessions", headers=ALICE)
    assert len(r.json()["items"]) == 2
    assert (await client.post(f"/agents/sessions/{sid}/stop", headers=ALICE)).status_code == 204


async def test_routes_owner_isolation(client: httpx.AsyncClient) -> None:
    sid = (await client.post("/agents/sessions", json={"kind": "note"}, headers=ALICE)).json()["id"]
    for method, path, body in [
        ("GET", f"/agents/sessions/{sid}", None),
        ("GET", f"/agents/sessions/{sid}/tree", None),
        ("GET", f"/agents/sessions/{sid}/state", None),
        ("GET", f"/agents/sessions/{sid}/events", None),
        ("POST", f"/agents/sessions/{sid}/messages", {"content": "x"}),
        ("POST", f"/agents/sessions/{sid}/stop", None),
        ("POST", f"/agents/sessions/{sid}/publish", None),
        ("PUT", f"/agents/sessions/{sid}/state", {"state": {}}),
    ]:
        r = await client.request(method, path, json=body, headers=BOB)
        assert r.status_code == 404, (method, path, r.status_code)
    assert (await client.get("/agents/sessions", headers=BOB)).json()["items"] == []
    r = await client.post("/agents/sessions", json={"kind": "inconnu"}, headers=ALICE)
    assert r.status_code == 404


async def test_routes_answer_conflict(client: httpx.AsyncClient) -> None:
    sid = (await client.post("/agents/sessions", json={"kind": "note"}, headers=ALICE)).json()["id"]
    r = await client.post(f"/agents/sessions/{sid}/answer", json={"answer": "oui"}, headers=ALICE)
    assert r.status_code == 409 and r.json()["error"] == "NotAwaitingInput"


async def test_routes_integration(client: httpx.AsyncClient, fake: FakeModelClient) -> None:
    """Routes des besoins d'intégration (D92–D105)."""
    r = await client.post(
        "/agents/sessions",
        json={"kind": "note", "model_settings": {"api_key": "x"}},
        headers=ALICE,
    )
    assert r.status_code == 422 and r.json()["error"] == "InvalidSettings"
    r = await client.post(
        "/agents/sessions",
        json={"kind": "note", "metadata": {"app": "a1"}, "model_settings": {"temperature": 0.3}},
        headers=ALICE,
    )
    sid = r.json()["id"]
    assert r.json()["model_settings"] == {"temperature": 0.3}
    base = f"/agents/sessions/{sid}"

    r = await client.patch(base, json={"model_settings": {"top_p": 0.9}}, headers=ALICE)
    assert r.json()["model_settings"] == {"top_p": 0.9}
    r = await client.post(f"{base}/preview", json={"content": "salut"}, headers=ALICE)
    assert r.status_code == 200 and r.json()["input_tokens"] > 0 and r.json()["system"]
    assert r.json()["settings"] == {"top_p": 0.9}

    r = await client.post(
        f"{base}/entries", json={"role": "assistant", "content": "Bonjour !"}, headers=ALICE
    )
    assert r.status_code == 201
    greeting = r.json()["id"]
    r = await client.patch(f"{base}/entries/{greeting}", json={"content": "Salut !"}, headers=ALICE)
    assert r.json()["derived_from"] == greeting
    edited = r.json()["id"]
    r = await client.put(f"{base}/entries/{edited}/label", json={"label": "Départ"}, headers=ALICE)
    assert r.json()["label"] == "Départ"
    r = await client.patch(
        f"{base}/entries/{edited}/metadata", json={"metadata": {"k": 1}}, headers=ALICE
    )
    assert r.json()["metadata"] == {"k": 1}
    r = await client.get(f"{base}/tree", params={"labeled": True}, headers=ALICE)
    assert [e["id"] for e in r.json()] == [edited]

    r = await client.post(f"{base}/entries/{greeting}/hide", headers=ALICE)
    assert r.json()["entry_ids"] == [greeting]
    assert greeting not in {
        e["id"] for e in (await client.get(f"{base}/tree", headers=ALICE)).json()
    }
    r = await client.get(f"{base}/tree", params={"include_hidden": True}, headers=ALICE)
    assert greeting in {e["id"] for e in r.json()}
    r = await client.post(f"{base}/entries/{edited}/hide", headers=ALICE)
    assert r.status_code == 409
    r = await client.post(f"{base}/entries/{greeting}/restore", headers=ALICE)
    assert r.json()["entry_ids"] == [greeting]

    r = await client.put(
        f"{base}/state", json={"state": {"title": "A"}, "amend": True}, headers=ALICE
    )
    first = r.json()["id"]
    r = await client.put(
        f"{base}/state", json={"state": {"title": "AB"}, "amend": True}, headers=ALICE
    )
    assert r.json()["id"] == first

    assert (await client.get(f"{base}/inputs", headers=ALICE)).json() == {}
    r = await client.patch(f"{base}/inputs", json={"inputs": {"x": {"kind": "y"}}}, headers=ALICE)
    assert r.status_code == 422

    r = await client.post(f"{base}/clone", json={"entry_id": edited}, headers=ALICE)
    assert r.status_code == 201
    clone_id = r.json()["id"]
    clone_branch = (await client.get(f"/agents/sessions/{clone_id}/branch", headers=ALICE)).json()
    assert [e["type"] for e in clone_branch] == ["root", "assistant"]

    # Le clone garde les métadonnées : il est retrouvé aussi.
    r = await client.get("/agents/sessions", params={"metadata": '{"app": "a1"}'}, headers=ALICE)
    assert {s["id"] for s in r.json()["items"]} == {sid, clone_id}
    r = await client.get("/agents/sessions", params={"metadata": "[1]"}, headers=ALICE)
    assert r.status_code == 422
    obj = (await client.post(f"{base}/publish", headers=ALICE)).json()["object_id"]
    r = await client.get("/agents/sessions", params={"object_id": obj}, headers=ALICE)
    assert [s["id"] for s in r.json()["items"]] == [sid]


async def test_preview_route_can_hide_prompt(agents: Agents[Any]) -> None:
    async def current_user() -> str:
        return "alice"

    app = FastAPI()
    app.include_router(make_router(agents, current_user, expose_prompt=False))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        sid = (await c.post("/sessions", json={"kind": "note"})).json()["id"]
        body = (await c.post(f"/sessions/{sid}/preview")).json()
    assert body["system"] is None and body["messages"] is None and body["input_tokens"] > 0

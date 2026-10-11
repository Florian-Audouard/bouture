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

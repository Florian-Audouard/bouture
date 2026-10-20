"""L'exemple personnage/monde, piloté par ses routes avec le modèle de démo."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bouture import ModelRequest
from examples.character_world.app import build_agents, create_app

from .conftest import PG_URL
from .test_routes import parse_sse

H = {"x-user": "alice"}


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_character_reads_published_world(backend: str, tmp_path: Path) -> None:
    if backend == "postgres" and not PG_URL:
        pytest.skip("BOUTURE_TEST_PG_URL non défini")
    db = tmp_path / "sous dossier" / "monde.db"  # dossier absent, espace dans le chemin
    urls = {"memory": None, "sqlite": f"sqlite:///{db}", "postgres": PG_URL}
    agents = build_agents(urls[backend])
    app = create_app(agents)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
    ):

        async def say(sid: str, content: str) -> list[dict]:
            r = await c.post(
                f"/agents/sessions/{sid}/messages", json={"content": content}, headers=H
            )
            assert r.status_code == 200, r.text
            return parse_sse(r.text)

        # Un monde, édité puis publié.
        world = (await c.post("/agents/sessions", json={"kind": "world"}, headers=H)).json()["id"]
        await say(world, "nom: Aldéra")
        await say(world, "lieu: Port-Brume")
        r = await c.post(f"/agents/sessions/{world}/publish", headers=H)
        assert r.status_code == 200, r.text
        pub = r.json()

        # Un personnage lié à ce monde (version épinglée).
        r = await c.post(
            "/agents/sessions",
            json={
                "kind": "character",
                "inputs": {"world": {"kind": "world", "id": pub["object_id"]}},
            },
            headers=H,
        )
        assert r.status_code == 201, r.text
        char = r.json()["id"]

        # Le monde évolue ensuite : le personnage garde la version 1.
        await say(world, "lieu: Hautes-Terres")
        await c.post(f"/agents/sessions/{world}/publish", headers=H)

        evs = await say(char, "monde")
        result = next(e for e in evs if e["type"] == "tool_result")
        assert "Aldéra" in result["data"]["content"] and "Port-Brume" in result["data"]["content"]
        assert "Hautes-Terres" not in result["data"]["content"]

        # Pause (AskUser) puis reprise.
        evs = await say(char, "portrait: aquarelle")
        assert evs[-1]["data"]["reason"] == "awaiting_input"
        r = await c.post(f"/agents/sessions/{char}/answer", json={"answer": "oui"}, headers=H)
        assert parse_sse(r.text)[-1]["data"]["reason"] == "completed"
        state = (await c.get(f"/agents/sessions/{char}/state", headers=H)).json()["state"]
        assert state["portrait_style"] == "aquarelle"

        # Bob ne voit pas les publications d'Alice.
        r = await c.get(f"/agents/publications/world/{pub['object_id']}", headers={"x-user": "bob"})
        assert r.json() == []

    if backend == "sqlite":
        import sqlite3

        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT version, name FROM app_worlds WHERE id = ? ORDER BY version",
                (pub["object_id"],),
            ).fetchall()
        assert rows == [(1, "Aldéra"), (2, "Aldéra")]

    if backend == "postgres":
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(PG_URL)  # type: ignore[arg-type]
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT version, name FROM app_worlds WHERE id = :id ORDER BY version"),
                {"id": pub["object_id"]},
            )
            assert [tuple(r) for r in rows] == [(1, "Aldéra"), (2, "Aldéra")]
        await engine.dispose()


async def test_character_site() -> None:
    from examples.character_site.app import build_agents as build_site
    from examples.character_site.app import create_app as create_site

    app = create_site(build_site())
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
    ):
        r = await c.get("/")
        assert r.status_code == 200 and "Atelier de personnages" in r.text
        assert (await c.get("/info")).json() == {"model": "demo"}
        sid = (await c.post("/agents/sessions", json={"kind": "character"}, headers=H)).json()["id"]
        for msg in ["nom: Lyra", "âge: 27 ans", "aime: la pluie", "compétence: crochetage"]:
            r = await c.post(f"/agents/sessions/{sid}/messages", json={"content": msg}, headers=H)
            assert parse_sse(r.text)[-1]["data"]["reason"] == "completed"
        state = (await c.get(f"/agents/sessions/{sid}/state", headers=H)).json()["state"]
        assert (state["name"], state["age"]) == ("Lyra", "27 ans")
        assert state["likes"] == ["la pluie"] and state["skills"] == ["crochetage"]

        # Valeur trop longue : refusée dès les arguments (maxLength du schéma), état intact.
        msg = "âge: " + "très " * 30
        r = await c.post(f"/agents/sessions/{sid}/messages", json={"content": msg}, headers=H)
        result = next(e["data"] for e in parse_sse(r.text) if e["type"] == "tool_result")
        assert result["is_error"] and "arguments invalides" in result["content"]
        state = (await c.get(f"/agents/sessions/{sid}/state", headers=H)).json()["state"]
        assert state["age"] == "27 ans"

        # Question posée par l'outil « ask » : choix proposés, réponse libre acceptée.
        msg = "question: Elfe ou humaine ? | elfe | humaine"
        r = await c.post(f"/agents/sessions/{sid}/messages", json={"content": msg}, headers=H)
        ask = next(e["data"] for e in parse_sse(r.text) if e["type"] == "ask_user")
        assert (ask["tool_name"], ask["choices"]) == ("ask", ["elfe", "humaine"])
        r = await c.post(
            f"/agents/sessions/{sid}/answer",
            json={"answer": "demi-elfe", "tool_call_id": ask["tool_call_id"]},
            headers=H,
        )
        result = next(e["data"] for e in parse_sse(r.text) if e["type"] == "tool_result")
        assert result["content"] == "demi-elfe"


def test_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.character_world.demo_model import DemoModelClient
    from examples.llm import OPENROUTER, client_from_env

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client, model = client_from_env()
    assert isinstance(client, DemoModelClient) and model == "demo"

    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        client_from_env()

    pytest.importorskip("litellm")
    from bouture.models.litellm import LiteLLMClient

    monkeypatch.setenv("LLM_MODEL", "vendor/some-model")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    client, model = client_from_env()
    assert isinstance(client, LiteLLMClient) and model == "openai/vendor/some-model"
    assert client.extra == {"api_base": OPENROUTER, "api_key": "sk-test"}


def test_studio_integration(tmp_path: Path) -> None:
    """Exemple d'intégration complète (README) : SQLite, BackgroundAgents, outil image."""
    from bouture import FakeModelClient, FakeReply
    from examples.studio import build, run

    seen: list[int] = []

    def check(request: ModelRequest) -> FakeReply:
        seen.extend(len(i.data) for i in request.images.values())
        return FakeReply("Voilà.")

    fake = FakeModelClient([FakeReply("", tool_calls=[("render", {"color": "bleu"})]), check])
    with build(fake, tmp_path / "studio") as bg:
        state = run(bg, "un carré bleu")
    assert state.renders == ["render-1.png"]
    assert seen and seen[0] > 0  # l'image a été chargée pour l'appel suivant
    assert (tmp_path / "studio" / "bouture.db").exists()

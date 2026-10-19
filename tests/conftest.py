from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from bouture import (
    AgentDefinition,
    Agents,
    AskUser,
    ChildResultContext,
    Event,
    FakeModelClient,
    MemoryStorage,
    PublishedRef,
    Storage,
    ToolContext,
)


class Note(BaseModel):
    title: str = ""
    tags: list[str] = Field(default_factory=list)
    score: int = Field(default=0, ge=0)
    image: str | None = None


class Image(BaseModel):
    url: str = ""


def make_definitions() -> tuple[AgentDefinition[Note, Any], AgentDefinition[Image, Any]]:
    note = AgentDefinition[Note, Any](
        "note", model="fake", state_type=Note, system_prompt="Tu édites une note."
    )

    @note.tool
    async def set_title(ctx: ToolContext[Note, Any], title: str) -> str:
        """Change le titre."""
        ctx.state.title = title
        return "ok"

    @note.tool
    async def set_score(ctx: ToolContext[Note, Any], score: int) -> str:
        """Change le score (doit rester positif)."""
        ctx.state.score = score
        return "ok"

    @note.tool
    async def boom(ctx: ToolContext[Note, Any]) -> str:
        """Modifie l'état puis échoue."""
        ctx.state.title = "corrompu"
        raise RuntimeError("panne")

    @note.tool
    async def slow(ctx: ToolContext[Note, Any]) -> str:
        """Outil qui ne se termine jamais."""
        ctx.state.title = "pendant"
        await asyncio.Event().wait()
        return "jamais"

    @note.tool
    async def ask_tag(ctx: ToolContext[Note, Any], proposal: str) -> AskUser:
        """Propose un tag à l'utilisateur."""
        ctx.state.title = ctx.state.title or "brouillon"
        return AskUser(f"Ajouter le tag {proposal} ?", choices=["oui", "non"], payload=proposal)

    @ask_tag.on_answer
    async def _ask_tag_answer(ctx: ToolContext[Note, Any], answer: str, payload: str) -> str:
        if answer == "oui":
            ctx.state.tags.append(payload)
            return f"tag {payload} ajouté"
        return "refusé"

    @note.tool
    async def ask_bad(ctx: ToolContext[Note, Any]) -> AskUser:
        """Pose une question en laissant un état invalide."""
        ctx.state.score = -1
        return AskUser("Vraiment ?")

    @note.tool
    async def make_image(ctx: ToolContext[Note, Any], prompt: str) -> str:
        """Lance une session enfant image."""
        info = await ctx.spawn_child("image", on_result="image_done", payload={"prompt": prompt})
        return f"session {info.id}"

    @note.child_handler("image_done")
    async def _image_done(
        ctx: ChildResultContext[Note, Any], ref: PublishedRef, child: Image, payload: Any
    ) -> None:
        ctx.state.image = child.url

    image = AgentDefinition[Image, Any]("image", model="fake", state_type=Image)

    @image.tool
    async def set_url(ctx: ToolContext[Image, Any], url: str) -> str:
        """Fixe l'URL."""
        ctx.state.url = url
        return "ok"

    return note, image


PG_URL = os.environ.get("BOUTURE_TEST_PG_URL")


if sys.platform == "win32" and PG_URL and "psycopg" in PG_URL:
    # psycopg async ne fonctionne pas avec la boucle Proactor de Windows.
    @pytest.fixture(scope="session")
    def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
        return asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]


@pytest.fixture(params=["memory", "postgres", "sqlite"])
async def storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Storage]:
    """Chaque test tourne sur la mémoire, sur SQLite (fichier temporaire, chemin avec espaces)
    et, si ``BOUTURE_TEST_PG_URL`` est défini, sur Postgres."""
    if request.param == "memory":
        yield MemoryStorage()
        return
    if request.param == "sqlite":
        from bouture.storage.sqlite import SQLiteStorage

        lite = SQLiteStorage(tmp_path / "dossier avec espaces" / "bouture.db")
        await lite.setup()
        yield lite
        await lite.close()
        return
    if not PG_URL:
        pytest.skip("BOUTURE_TEST_PG_URL non défini")
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from bouture.storage.postgres import PostgresStorage

    engine = create_async_engine(PG_URL)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS bouture CASCADE"))
    st = PostgresStorage(engine)
    await st.setup()
    yield st
    await st.close()
    await engine.dispose()


@pytest.fixture
def fake() -> FakeModelClient:
    return FakeModelClient()


AgentsFactory = Callable[..., Agents[Any]]


@pytest.fixture
async def make_agents(storage: Storage) -> AsyncIterator[AgentsFactory]:
    created: list[Agents[Any]] = []

    def factory(
        model_client: FakeModelClient, *, store: Storage | None = None, **kw: Any
    ) -> Agents[Any]:
        kw.setdefault("retry_base_delay", 0)
        kw.setdefault("poll_interval", 0.05)
        a = Agents[Any](storage=store or storage, model_client=model_client, **kw)
        a.register(*make_definitions())
        created.append(a)
        return a

    yield factory
    for a in created:
        await a.shutdown(timeout=2)


@pytest.fixture
async def agents(make_agents: AgentsFactory, fake: FakeModelClient) -> Agents[Any]:
    a = make_agents(fake)
    await a.setup()
    return a


async def collect(it: AsyncIterator[Event]) -> list[Event]:
    return [e async for e in it]


def types(events: list[Event]) -> list[str]:
    return [e.type for e in events]

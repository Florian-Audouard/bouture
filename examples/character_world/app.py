"""Application FastAPI de l'exemple.

    uv run uvicorn examples.character_world.app:app --reload

Configuration (environnement ou ``.env`` à la racine) :
- ``LLM_API_KEY``, ``LLM_MODEL``, ``LLM_BASE_URL`` : LLM compatible OpenAI (voir ``examples/llm.py``) ;
  sans clé → modèle de démo ;
- ``DATABASE_URL`` (ex. ``postgresql+asyncpg://user:pw@localhost/db``) ; absent → stockage mémoire.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header
from sqlalchemy import text

from bouture import Agents, MemoryStorage, ModelClient
from bouture.fastapi import make_router

from ..llm import client_from_env, load_env
from .definitions import AppDeps, register
from .demo_model import DemoModelClient

APP_TABLES = """
CREATE TABLE IF NOT EXISTS app_worlds (
    id uuid NOT NULL,
    version integer NOT NULL,
    owner text NOT NULL,
    name text NOT NULL,
    genre text NOT NULL,
    description text NOT NULL,
    PRIMARY KEY (id, version)
)
"""


def build_agents(
    database_url: str | None = None, client: ModelClient | None = None, model: str = "demo"
) -> Agents[AppDeps]:
    deps = AppDeps()
    client = client or DemoModelClient()
    if database_url:
        from sqlalchemy.ext.asyncio import create_async_engine

        deps.engine = create_async_engine(database_url)
        agents = Agents[AppDeps](engine=deps.engine, deps=deps, model_client=client)
    else:
        agents = Agents[AppDeps](storage=MemoryStorage(), deps=deps, model_client=client)
    deps.agents = agents
    register(agents, model)
    return agents


async def current_user(x_user: str = Header(default="demo")) -> str:
    """À remplacer par l'authentification du projet."""
    return x_user


def create_app(agents: Agents[AppDeps]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        engine = agents.deps.engine
        if engine is not None:
            async with engine.begin() as conn:
                await conn.execute(text(APP_TABLES))
        await agents.setup()
        try:
            yield
        finally:
            await agents.shutdown()
            if engine is not None:
                await engine.dispose()

    app = FastAPI(title="bouture — personnage & monde", lifespan=lifespan)
    app.include_router(make_router(agents, current_user), prefix="/agents")
    return app


load_env()
app = create_app(build_agents(os.environ.get("DATABASE_URL"), *client_from_env()))

"""Site de démonstration : un seul kind, ``character``, et une page web.

    uv run uvicorn examples.character_site.app:app --reload

puis ouvrir http://localhost:8000. Configuration (environnement ou ``.env`` à la racine) :
- ``LLM_API_KEY``, ``LLM_MODEL``, ``LLM_BASE_URL`` : LLM compatible OpenAI (voir ``examples/llm.py``) ;
  sans clé → modèle de démo ;
- ``DATABASE_URL`` (ex. ``postgresql+asyncpg://user:pw@localhost/db``) ; absent → stockage mémoire.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bouture import AgentDefinition, Agents, AskUser, MemoryStorage, ModelClient, ToolContext
from bouture.fastapi import make_router

from ..character_world.demo_model import DemoModelClient
from ..llm import client_from_env, load_env

STATIC = Path(__file__).parent / "static"


LISTS = {
    "quirks": "manies",
    "likes": "ce qu'il ou elle aime",
    "dislikes": "ce qu'il ou elle n'aime pas",
    "skills": "compétences",
}
ListName = Literal["quirks", "likes", "dislikes", "skills"]
MAX_ITEMS = 12

# Longueurs maximales : partagées par l'état et par les paramètres des outils, pour que le modèle
# les voie dans le schéma (maxLength) au lieu de les découvrir par une erreur.
Short = Annotated[str, Field(max_length=80)]
Text = Annotated[str, Field(max_length=2000)]
Story = Annotated[str, Field(max_length=4000)]
Item = Annotated[str, Field(max_length=120)]
Items = list[Item]


class Character(BaseModel):
    name: Short = ""
    age: Short = ""
    gender: Short = ""
    appearance: Text = ""
    personality: Text = ""
    backstory: Story = ""
    quirks: Items = Field(default_factory=list, max_length=MAX_ITEMS)
    likes: Items = Field(default_factory=list, max_length=MAX_ITEMS)
    dislikes: Items = Field(default_factory=list, max_length=MAX_ITEMS)
    skills: Items = Field(default_factory=list, max_length=MAX_ITEMS)
    portrait_style: str | None = None


def _v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Version 1 : une simple liste ``traits``, reprise dans ``personality``."""
    traits = data.pop("traits", [])
    return {**data, "personality": ", ".join(traits)}


SYSTEM_PROMPT = """Tu aides l'utilisateur à créer un personnage de fiction, en français.

La fiche contient : nom, âge, genre, apparence, personnalité, histoire, et quatre listes :
manies, ce qu'il ou elle aime, ce qu'il ou elle n'aime pas, compétences.

Règles :
- Toute question à l'utilisateur passe par l'outil `ask`, jamais par du texte. Une question par
  appel, avec 2 à 5 choix courts quand c'est possible (l'utilisateur peut aussi répondre
  librement). Tu peux appeler `ask` plusieurs fois de suite : les questions sont posées une à une.
- Enregistre chaque élément validé dès que possible : `set_fields` pour les champs texte (plusieurs
  à la fois si besoin), `add_items` et `remove_items` pour les listes.
- Respecte les longueurs maximales indiquées dans les outils. Si un outil renvoie une erreur,
  corrige les arguments (par exemple en résumant) et rappelle-le aussitôt, sans en parler à
  l'utilisateur.
- Si l'utilisateur te laisse carte blanche, invente et enregistre directement.
- Tes messages restent courts ; le Markdown est accepté (gras, listes).
- Quand la fiche est bien remplie, propose un style de portrait avec `propose_portrait`.
"""


def make_definition(model: str) -> AgentDefinition[Character, None]:
    character = AgentDefinition[Character, None](
        "character",
        model=model,
        state_type=Character,
        state_version=2,
        state_migrations={1: _v1_to_v2},
        system_prompt=SYSTEM_PROMPT,
    )

    @character.tool
    async def set_fields(
        ctx: ToolContext[Character, None],
        name: Short | None = None,
        age: Annotated[
            Short | None, Field(description="Court, ex. « 27 ans » ou « trois siècles »")
        ] = None,
        gender: Short | None = None,
        appearance: Annotated[Text | None, Field(description="Description physique")] = None,
        personality: Annotated[Text | None, Field(description="Caractère, tempérament")] = None,
        backstory: Annotated[Story | None, Field(description="Histoire, passé")] = None,
    ) -> str:
        """Remplace un ou plusieurs champs texte de la fiche. Les champs omis ne changent pas."""
        values = {
            "name": name,
            "age": age,
            "gender": gender,
            "appearance": appearance,
            "personality": personality,
            "backstory": backstory,
        }
        changed = [k for k, v in values.items() if v is not None]
        if not changed:
            return "Aucun champ fourni."
        for k in changed:
            setattr(ctx.state, k, values[k])
        return "Champs mis à jour : " + ", ".join(changed)

    @character.tool
    async def add_items(ctx: ToolContext[Character, None], list: ListName, items: Items) -> str:
        """Ajoute des éléments à une liste : quirks (manies), likes (aime), dislikes (n'aime
        pas), skills (compétences). 12 éléments au maximum par liste, 120 caractères par élément."""
        target: Items = getattr(ctx.state, list)
        added = [i for i in items if i not in target]
        if len(target) + len(added) > MAX_ITEMS:
            return (
                f"Refusé : {LISTS[list]} contient déjà {len(target)} éléments sur {MAX_ITEMS}. "
                "Retire-en d'abord avec remove_items."
            )
        target.extend(added)
        return f"{LISTS[list]} : ajouté {', '.join(added) or 'rien'}"

    @character.tool
    async def remove_items(ctx: ToolContext[Character, None], list: ListName, items: Items) -> str:
        """Retire des éléments d'une liste (quirks, likes, dislikes, skills)."""
        target: Items = getattr(ctx.state, list)
        removed = [i for i in items if i in target]
        for i in removed:
            target.remove(i)
        unknown = [i for i in items if i not in removed]
        return f"{LISTS[list]} : retiré {', '.join(removed) or 'rien'}" + (
            f" (inconnus : {', '.join(unknown)})" if unknown else ""
        )

    @character.tool
    async def ask(
        ctx: ToolContext[Character, None],
        question: str,
        choices: Annotated[
            Items | None, Field(description="2 à 5 réponses courtes proposées")
        ] = None,
    ) -> AskUser:
        """Pose une question à l'utilisateur et attend sa réponse (choix proposé ou texte libre)."""
        return AskUser(question, choices=choices or None)

    @ask.on_answer
    async def _ask_answer(ctx: ToolContext[Character, None], answer: str, payload: None) -> str:
        return answer

    @character.tool
    async def propose_portrait(ctx: ToolContext[Character, None], style: str) -> AskUser:
        """Propose un style de portrait ; attend la confirmation de l'utilisateur."""
        return AskUser(
            f"Retenir un portrait de style « {style} » ?", choices=["oui", "non"], payload=style
        )

    @propose_portrait.on_answer
    async def _portrait_answer(ctx: ToolContext[Character, None], answer: str, payload: str) -> str:
        if answer == "oui":
            ctx.state.portrait_style = payload
            return f"Style de portrait retenu : {payload}"
        return "Pas de portrait pour l'instant."

    return character


def build_agents(
    database_url: str | None = None, client: ModelClient | None = None, model: str = "demo"
) -> Agents[None]:
    client = client or DemoModelClient()
    if database_url:
        from sqlalchemy.ext.asyncio import create_async_engine

        agents = Agents[None](engine=create_async_engine(database_url), model_client=client)
    else:
        agents = Agents[None](storage=MemoryStorage(), model_client=client)
    agents.register(make_definition(model))
    return agents


async def current_user(x_user: str = Header(default="demo")) -> str:
    """Démo : la page envoie un identifiant aléatoire. À remplacer par une vraie authentification."""
    return x_user


def create_app(agents: Agents[None]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        await agents.setup()
        try:
            yield
        finally:
            await agents.shutdown()

    app = FastAPI(title="bouture — création de personnage", lifespan=lifespan)
    app.include_router(make_router(agents, current_user), prefix="/agents")

    @app.get("/info", include_in_schema=False)
    async def info() -> dict[str, str]:
        return {"model": agents.definition("character").model}

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


load_env()
app = create_app(build_agents(os.environ.get("DATABASE_URL"), *client_from_env()))

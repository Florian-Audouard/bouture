"""Les deux kinds de l'exemple : ``world`` (un monde) et ``character`` (un personnage lié à un monde)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from bouture import (
    AgentDefinition,
    Agents,
    AskUser,
    Data,
    PublishContext,
    PublishedRef,
    Ref,
    ResolveContext,
    ToolContext,
)


# -- états métier ------------------------------------------------------------------------
class World(BaseModel):
    name: str = ""
    genre: str = ""
    description: str = ""
    places: list[str] = Field(default_factory=list)


class Character(BaseModel):
    name: str = ""
    backstory: str = ""
    traits: list[str] = Field(default_factory=list)
    portrait_style: str | None = None  # une référence ou un réglage, jamais un binaire


@dataclass
class AppDeps:
    """Dépendances passées aux outils (``ctx.deps``)."""

    agents: Agents[AppDeps] | None = None
    engine: Any = None


def render_world(w: World) -> str:
    places = ", ".join(w.places) or "aucun"
    head = f"{w.name or '(sans nom)'} — {w.genre or 'genre non précisé'}"
    return f"{head}\n{w.description}\nLieux : {places}"


def make_definitions(
    model: str,
) -> tuple[AgentDefinition[World, AppDeps], AgentDefinition[Character, AppDeps]]:
    # -- monde ---------------------------------------------------------------------------
    world = AgentDefinition[World, AppDeps](
        "world",
        model=model,
        state_type=World,
        system_prompt="Tu aides l'utilisateur à concevoir un monde de fiction. "
        "Modifie la fiche avec les outils.",
    )

    @world.tool
    async def set_name(ctx: ToolContext[World, AppDeps], name: str) -> str:
        """Change le nom du monde."""
        ctx.state.name = name
        return f"Nom : {name}"

    @world.tool
    async def set_description(ctx: ToolContext[World, AppDeps], description: str) -> str:
        """Remplace la description du monde."""
        ctx.state.description = description
        return "Description mise à jour."

    @world.tool
    async def add_place(ctx: ToolContext[World, AppDeps], place: str) -> str:
        """Ajoute un lieu au monde."""
        if place not in ctx.state.places:
            ctx.state.places.append(place)
        return f"Lieu ajouté : {place}"

    @world.on_publish
    async def _copy_world(ctx: PublishContext[AppDeps], ref: PublishedRef, state: World) -> None:
        # Copie dans les tables du projet, dans la transaction de publication.
        if ctx.db is None:  # stockage mémoire : pas de tables projet
            return
        await ctx.db.execute(
            text(
                "INSERT INTO app_worlds (id, version, owner, name, genre, description) "
                "VALUES (:id, :version, :owner, :name, :genre, :description)"
            ),
            {
                "id": ref.object_id,
                "version": ref.version,
                "owner": ctx.owner,
                **state.model_dump(exclude={"places"}),
            },
        )

    # -- personnage ----------------------------------------------------------------------
    character = AgentDefinition[Character, AppDeps](
        "character",
        model=model,
        state_type=Character,
        inputs={"world": "world"},
        system_prompt="Tu aides l'utilisateur à écrire un personnage cohérent avec son monde. "
        "Appelle read_world pour consulter la fiche complète du monde.",
    )

    @character.tool(name="set_name")
    async def set_character_name(ctx: ToolContext[Character, AppDeps], name: str) -> str:
        """Change le nom du personnage."""
        ctx.state.name = name
        return f"Nom : {name}"

    @character.tool
    async def set_backstory(ctx: ToolContext[Character, AppDeps], backstory: str) -> str:
        """Remplace l'histoire du personnage."""
        ctx.state.backstory = backstory
        return "Histoire mise à jour."

    @character.tool
    async def add_trait(ctx: ToolContext[Character, AppDeps], trait: str) -> str:
        """Ajoute un trait de caractère."""
        ctx.state.traits.append(trait)
        return f"Trait ajouté : {trait}"

    @character.tool
    async def read_world(ctx: ToolContext[Character, AppDeps]) -> str:
        """Lit la fiche complète du monde publié lié à ce personnage."""
        ref = ctx.inputs.get("world")
        if not isinstance(ref, Ref):
            return "Aucun monde lié."
        assert ctx.deps.agents is not None
        # Lecture de l'objet publié (version épinglée à la création), jamais de la session du monde.
        w: World = await ctx.deps.agents.get_published_state(ref, owner=ctx.owner)
        return render_world(w)

    @character.tool
    async def propose_portrait(ctx: ToolContext[Character, AppDeps], style: str) -> AskUser:
        """Propose un style de portrait ; met le tour en pause jusqu'à la réponse."""
        return AskUser(
            f"Retenir un portrait de style « {style} » ?", choices=["oui", "non"], payload=style
        )

    @propose_portrait.on_answer
    async def _portrait_answer(
        ctx: ToolContext[Character, AppDeps], answer: str, payload: str
    ) -> str:
        if answer == "oui":
            ctx.state.portrait_style = payload
            return f"Style de portrait retenu : {payload}"
        return "Pas de portrait."

    return world, character


def register(agents: Agents[AppDeps], model: str) -> None:
    agents.register(*make_definitions(model))

    @agents.resolver("world")
    async def _resolve_world(inp: Ref | Data, ctx: ResolveContext[Any]) -> str:
        # Résumé injecté dans le prompt système du personnage.
        if isinstance(inp, Data):
            return render_world(World.model_validate(inp.value))
        w: World = await agents.get_published_state(inp)
        return render_world(w)

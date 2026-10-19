# bouture

Agents LLM pour backends Python async (FastAPI ou autre) dont les sessions sont des **arbres
versionnés** : fork, rollback, regenerate, clone, stop. Tout est persisté en Postgres (schéma
`bouture`, créé automatiquement) ; un stockage en mémoire sert aux tests.

- Boucle agent maison (LLM → outils → LLM) avec streaming d'événements.
- État métier typé (Pydantic), versionné à chaque changement, migrable.
- Outils Python qui lisent et modifient l'état, peuvent mettre le tour en pause (`AskUser`).
  La reprise fonctionne après un redémarrage.
- Publication versionnée d'objets, inputs entre agents, sessions enfants.
- Serveur sans état entre les tours, un seul tour à la fois par session (verrou par bail).
- Un F5 ou une déconnexion n'arrête jamais un tour : seul `stop` l'arrête.
- Routes FastAPI optionnelles ; le cœur n'importe jamais FastAPI.

Conception : [PROPOSITION.md](PROPOSITION.md) (API, schéma, événements) et
[DECISIONS.md](DECISIONS.md) (décisions numérotées `Dxx`).

## Installation

Python ≥ 3.12.

```bash
uv add "bouture[litellm,fastapi,asyncpg]"
```

Extras : `litellm` (fournisseurs LLM), `claude-code` (abonnement Claude via Claude Code, voir
plus bas), `fastapi` (routes), `asyncpg` ou `psycopg` (pilote Postgres ; LISTEN/NOTIFY requis).

## Démarrage rapide

```python
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine

from bouture import AgentDefinition, Agents, ToolContext
from bouture.models.litellm import LiteLLMClient


class Note(BaseModel):
    title: str = ""
    tags: list[str] = []


note = AgentDefinition[Note, None](
    "note",
    model="anthropic/claude-sonnet-5-5",
    state_type=Note,
    system_prompt="Tu aides à rédiger une note.",
)


@note.tool
async def set_title(ctx: ToolContext[Note, None], title: str) -> str:
    """Change le titre de la note."""
    ctx.state.title = title
    return "ok"


agents = Agents(engine=create_async_engine("postgresql+asyncpg://…"), model_client=LiteLLMClient())
agents.register(note)
await agents.setup()  # crée / migre le schéma `bouture`

s = await agents.create_session("note", owner="alice")
turn = await agents.send_message(s.id, "alice", "Intitule la note « Courses »")
async for ev in turn.events():
    print(ev.type, ev.data)

await agents.shutdown()
```

Le tour tourne en tâche de fond : ne plus lire `turn.events()` ne l'arrête pas. Pour se
rattacher (autre onglet, après F5) : `agents.subscribe(session_id, owner, after=last_event_id)`.

## Concepts

**Définition (kind)** — `AgentDefinition[S, D]` : nom, type d'état `S` (modèle Pydantic ou
`None` pour un chat sans état), état initial, prompt système (chaîne ou fonction), modèle
(chaîne LiteLLM, surchargeable par session), outils, inputs attendus. `D` est le type des
dépendances applicatives passées à `Agents(deps=…)` et reçues dans `ctx.deps`.

**Session** — instance d'un kind appartenant à un `owner`. C'est un arbre d'entrées
(`root`, `user`, `assistant`, `tool_result`, `state_edit`, `child_result`, `input_update`,
`summary`) avec un pointeur de feuille. Rien n'est jamais supprimé : on déplace la feuille, on
crée une branche ou on met une branche à la corbeille.
Toutes les méthodes prennent `owner` ; la session d'un autre propriétaire donne `NotFound`.

**État** — snapshot enregistré à chaque changement, réinjecté dans le contexte à chaque tour.
Après chaque outil, l'état est revalidé : s'il est invalide (ou si l'outil lève), il est restauré
et l'erreur revient au LLM. L'état ne contient que des **références** vers les binaires.

## Outils et pause

```python
from bouture import AskUser


@note.tool
async def propose_tag(ctx: ToolContext[Note, None], tag: str) -> AskUser:
    """Propose un tag à l'utilisateur."""
    return AskUser(f"Ajouter le tag {tag} ?", choices=["oui", "non"], payload=tag)


@propose_tag.on_answer
async def _(ctx: ToolContext[Note, None], answer: str, payload: str) -> str:
    if answer == "oui":
        ctx.state.tags.append(payload)
        return "tag ajouté"
    return "refusé"
```

Le tour se termine avec `turn_end{reason: "awaiting_input"}`. `agents.answer(session_id, owner,
"oui")` le reprend, même sur un autre processus. Envoyer un message à la place annule la question.

Pas d'outil intégré : pas de bash, pas d'accès aux fichiers locaux. Options : `@note.tool(name=…,
timeout=…)`, `ctx.emit(name, data)` (événement `custom`), `ToolFatal` pour arrêter le tour.

## Arbre

| Méthode | Effet |
|---|---|
| `fork(sid, owner, entry_id, content)` | nouvelle branche depuis le parent d'un message utilisateur, avec un message édité |
| `regenerate(sid, owner, entry_id)` | relance le tour depuis le message utilisateur de ce tour |
| `rollback(sid, owner, entry_id)` | déplace la feuille sur un checkpoint (conversation + état) |
| `switch_branch(sid, owner, entry_id)` | bascule sur une autre branche |
| `siblings(sid, owner, entry_id)` | variantes d'une entrée (pour « < 2/3 > ») |
| `clone(sid, owner, entry_id=None)` | nouvelle session avec un seul chemin (la branche active par défaut) |
| `edit_state(sid, owner, state, amend=False)` | édition manuelle → entrée `state_edit` (le LLM en est informé) ; `amend=True` pour la sauvegarde automatique |
| `append_message(sid, owner, role, content)` | message posé sans appel au modèle (accueil, note) |
| `edit_entry(sid, owner, entry_id, content)` | corrige un message sans régénérer ; l'original reste en frère |
| `label_entry(sid, owner, entry_id, label)` | nomme un checkpoint (point de sauvegarde) |
| `hide_branch` · `restore_branch` | corbeille d'une branche (hors chemin actif) |
| `stop(sid, owner)` | arrête le tour en cours ; le texte partiel est conservé |
| `get_tree(include_hidden=, labeled=)` · `get_branch` · `get_state(entry_id=…)` | lecture |

## Publication, inputs, sessions enfants

```python
ref = await agents.publish(world_session_id, owner)  # nouvelle version, idempotent


@world.on_publish  # dans la transaction de publication
async def _(ctx, ref, state):
    await ctx.db.execute(...)  # copie dans les tables du projet


s = await agents.create_session("character", owner, inputs={"world": Ref("world", ref.object_id)})
```

- Un `Ref` sans version est épinglé sur la dernière version à la création ; `Data(kind, value)`
  passe un snapshot validé par `agents.data_type(kind, Model)`.
- `@agents.resolver("world")` transforme un input en texte ajouté au prompt système.
- Les outils lisent les objets publiés (`agents.get_published_state(ref, owner=…)` ou les tables
  du projet), jamais les sessions des autres agents.
- `ctx.spawn_child(kind, on_result="handler", payload=…)` crée une session enfant ; à sa
  publication, `@definition.child_handler("handler")` applique le résultat à l'état du parent
  (entrée `child_result`).

## Intégration dans une application

Ce qui sert quand bouture remplace l'historique d'une application existante (D92–D105) :

```python
# Clé de l'utilisateur fournie à chaque appel, jamais stockée.
async def credentials(request: ModelRequest) -> dict[str, Any]:
    return {"api_key": await vault.key_for(request.context.owner)}


agents = Agents(storage=storage, model_client=LiteLLMClient(credentials=credentials))
s = await agents.create_session("character", owner, model_settings={"temperature": 0.9})


@character.initial_messages  # message d'accueil, sans appel au modèle
async def _(ctx: InitContext[Char, Deps]) -> list[tuple[str, Any]]:
    return [("assistant", "Bonjour, voyageur.")]


@character.on_entries  # lier les entrées du tour à des objets du projet, dans le commit
async def _(ctx: EntriesContext[Deps], entries: list[EntryView]) -> dict[UUID, dict]:
    return {e.id: {"app_message_id": ...} for e in entries if e.type == "assistant"}
```

- **Aperçu** : `await agents.preview(sid, owner, "message à venir")` renvoie le modèle, les
  réglages, l'estimation de tokens et de coût, et le prompt exact, sans lancer de tour.
- **Réglages par session** : seuls ceux de `AgentDefinition(overridable_settings=…)` sont
  acceptés (échantillonnage par défaut) ; toute autre clé lève `InvalidSettings`.
- **Références** : `update_inputs(sid, owner, {"world": Ref(...)})` change un input à partir
  de ce point de la branche ; `get_inputs` lit les inputs effectifs.
- **Retrouver une session** : `list_sessions(owner, object_id=…)` ou `metadata={"app": …}`.
- **Métadonnées d'entrée** : `EntryView.metadata`, `set_entry_metadata` pour les modifier.
- **Import** : `import_session(owner, kind, entries=[ImportedEntry(...)], publications=[...])`
  reprend un historique existant (ids et dates conservés), en une transaction.
- **Arguments d'outil en direct** : `AgentDefinition(stream_tool_args=True)` émet
  `tool_call_delta`.

**Migrations avec Alembic.** Une révision du projet applique le schéma de bouture, et
l'application vérifie au démarrage sans rien modifier :

```python
# alembic/versions/xxxx_bouture.py
from alembic import op
from bouture.storage.postgres import apply_migrations


def upgrade() -> None:
    apply_migrations(op.get_bind())  # idempotent ; ``target=`` pour figer une version


# env.py : exclure le schéma de bouture de l'autogénération
def include_name(name, type_, parent_names):
    return not (type_ == "schema" and name == "bouture")


# démarrage : lecture seule, SchemaOutdated si une révision manque
await storage.setup(auto_migrate=False)
```

## Événements

Chaque événement : `{id, session_id, turn_id, type, data, ts}`. Types : `turn_start`,
`user_message`, `message_start`, `text_delta`, `reasoning_delta`, `tool_call_delta` (opt-in),
`tool_call`, `message_end`, `tool_result`, `state_changed` (JSON Patch), `ask_user`,
`child_spawned`, `child_result_applied`, `entry_metadata`, `custom`, `turn_end`. `turn_end.reason` ∈ `completed | awaiting_input | interrupted | error |
max_steps | crashed | server_shutdown`. Détails : [PROPOSITION.md §3](PROPOSITION.md).

## Routes FastAPI

```python
from bouture.fastapi import make_router


async def current_user(...) -> str:  # dépendance FastAPI du projet
    ...


app.include_router(make_router(agents, current_user), prefix="/agents")
```

Les routes vérifient l'appartenance via `owner` et ne contiennent aucune logique. Les tours
répondent en SSE (en-tête `X-Turn-Id`) ; `GET /sessions/{id}/events` accepte `Last-Event-ID`.
Erreurs : 404 (`NotFound`, `UnknownKind`), 409 (`SessionBusy`, `InvalidTarget`,
`NotAwaitingInput`), 422 (`StateValidationError`, `ResolveError`, `InvalidSettings`). La purge
RGPD (`purge_session`, `purge_owner`) n'est pas exposée. `make_router(..., expose_prompt=False)`
masque le prompt dans `POST /sessions/{id}/preview`. Liste complète :
[PROPOSITION.md §4](PROPOSITION.md).

## Exemple

[examples/character_world](examples/character_world) : deux kinds, `world` et `character`.
Le personnage reçoit un monde publié en input, l'outil `read_world` en lit la fiche, et
`propose_portrait` montre la pause. Sans clé d'API, un modèle de démo répond à des commandes
(`nom: …`, `lieu: …`, `monde`, `portrait: …`).

```bash
uv run uvicorn examples.character_world.app:app --reload
```

[examples/character_site](examples/character_site) : un site de démonstration (une page, sans
build) pour créer un personnage en discutant avec l'agent. La fiche (nom, âge, genre, apparence,
personnalité, histoire, manies, goûts, aversions, compétences) se met à jour en direct. L'agent
pose ses questions avec l'outil `ask` (choix proposés ou réponse libre), et une question en attente
reste affichée après un rechargement. Les réponses de l'agent sont rendues en Markdown, et la page
permet aussi l'édition manuelle et la publication.

```bash
uv run uvicorn examples.character_site.app:app --reload
```

Les deux exemples tournent sans clé avec un modèle de démonstration (commandes `nom: …`,
`trait: …`). Pour un vrai LLM compatible OpenAI (OpenRouter par défaut), créez un fichier `.env` à
la racine. Il est ignoré par git et ne doit jamais être commité.

```dotenv
LLM_API_KEY=sk-or-...                       # https://openrouter.ai/keys
LLM_MODEL=anthropic/claude-sonnet-4.5       # un modèle qui gère les appels d'outils
# LLM_BASE_URL=https://openrouter.ai/api/v1  # défaut ; tout autre endpoint compatible OpenAI
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/db
```

Pour utiliser un abonnement Claude (Pro / Max) au lieu d'une clé d'API, installez l'extra
`claude-code`, connectez Claude Code une fois (`claude setup-token`, puis
`CLAUDE_CODE_OAUTH_TOKEN=…` dans l'environnement, ou `claude login`), et mettez dans le `.env` :

```dotenv
LLM_PROVIDER=claude-code
LLM_MODEL=sonnet                            # alias ou identifiant Claude Code ; défaut sonnet
```

`ClaudeCodeClient` ([src/bouture/models/claude_code.py](src/bouture/models/claude_code.py)) lance
Claude Code à chaque appel : l'historique lui est passé en transcript, les outils en MCP, et
l'exécution s'arrête au premier appel d'outil pour que bouture l'exécute. Comptez une à deux
secondes de latence par appel. Réservé au développement et à l'usage personnel : les conditions
d'Anthropic interdisent d'offrir un abonnement claude.ai aux utilisateurs d'un produit ; en
production, utilisez une clé d'API avec `LiteLLMClient`.

Les variables déjà présentes dans l'environnement priment sur le `.env` (voir
[examples/llm.py](examples/llm.py)). `DATABASE_URL` active Postgres, sinon le stockage est en mémoire.
L'utilisateur courant est lu dans l'en-tête `X-User`, à remplacer par votre authentification.

## Tests

```bash
uv run pytest
```

Faux LLM (`FakeModelClient`) et stockage mémoire. Pour exécuter aussi chaque test sur Postgres :

```bash
BOUTURE_TEST_PG_URL=postgresql+asyncpg://postgres:test@localhost:55432/bouture_test uv run pytest
```

La base de test est réinitialisée (`DROP SCHEMA bouture CASCADE`) : utilisez une base dédiée.
Vérifications : `uv run ruff check .` et `uv run pyright`.

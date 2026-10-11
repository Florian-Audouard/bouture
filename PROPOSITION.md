# bouture — Proposition : API publique, schéma, événements

Statut : **validée et implémentée**. Les écarts et précisions apparus à l'implémentation
sont consignés dans [DECISIONS.md](DECISIONS.md) (D80–D91).
Les références `Dxx` renvoient à [DECISIONS.md](DECISIONS.md).

---

## 1. API publique

### 1.1 Définition d'un agent (kind)

```python
from pydantic import BaseModel
from bouture import AgentDefinition, ToolContext, AskUser, Ref


class Character(BaseModel):
    name: str = ""
    backstory: str = ""
    portrait: ImageRef | None = None


character = AgentDefinition[Character, AppDeps](
    kind="character",
    state_type=Character,  # None → agent sans état (chat simple)
    initial_state=lambda inputs: Character(),  # ou une instance
    state_version=2,
    state_migrations={1: migrate_v1_to_v2},  # dict → dict
    system_prompt="Tu aides à écrire un personnage…",  # str | (ctx) -> str
    model="anthropic/claude-sonnet-5-5",
    model_settings={"temperature": 0.7, "max_tokens": 4096},
    inputs={"world": "world"},  # nom → kind attendu (validé à la création)
    max_steps=25,
    tool_timeout=120,
)
```

**Outils** — signature typée, schéma JSON dérivé des annotations et de la docstring :

```python
@character.tool
async def set_name(ctx: ToolContext[Character, AppDeps], name: str) -> str:
    """Change le nom du personnage."""
    ctx.state.name = name
    return "ok"


@character.tool
async def read_world(ctx: ToolContext[Character, AppDeps]) -> str:
    """Lit la description du monde publié lié à ce personnage."""
    ref = ctx.inputs["world"]  # Ref épinglée
    async with AsyncSession(ctx.deps.engine) as db:
        return (await db.get(WorldRow, (ref.id, ref.version))).description


@character.tool
async def propose_portrait(ctx: ToolContext[Character, AppDeps], style: str) -> str | AskUser:
    """Propose de générer un portrait."""
    return AskUser(
        question="Je génère un portrait ?", choices=["oui", "non"], payload={"style": style}
    )


@propose_portrait.on_answer
async def _(ctx: ToolContext[Character, AppDeps], answer: str, payload: dict) -> str:
    if answer == "oui":
        await ctx.spawn_child(
            "image",
            inputs={"brief": Data("portrait_brief", payload)},
            on_result="set_portrait",
            initial_message="Génère le portrait",
        )
        return "Session portrait lancée."
    return "Pas de portrait."
```

**Crochets** (tous optionnels) :

```python
@character.on_publish  # dans la transaction de la lib (D53)
async def _(ctx: PublishContext[AppDeps], ref: PublishedRef, state: Character) -> None:
    ctx.db.add(
        CharacterRow(id=ref.object_id, version=ref.version, owner=ctx.owner, **state.model_dump())
    )


@character.child_handler("set_portrait")  # D58
async def _(
    ctx: ChildResultContext[Character, AppDeps],
    ref: PublishedRef,
    child_state: ImageState,
    payload: dict,
) -> None:
    ctx.state.portrait = ImageRef(id=ref.object_id, version=ref.version, url=child_state.url)


@character.render_state  # D42 — défaut : JSON indenté
def _(state: Character) -> str: ...


@character.build_context  # D21 — défaut : voir §1.5
async def _(ctx: BuildContext[Character, AppDeps]) -> ModelRequest: ...


@character.on_turn_end  # D67 — après commit, hors transaction
async def _(ctx: TurnEndContext[AppDeps], result: TurnResult) -> None: ...
```

**`ToolContext[S, D]`** : `state: S` (copie de travail), `deps: D`, `owner: str`,
`session_id: UUID`, `turn_id: UUID`, `inputs: Mapping[str, Ref | Data]`,
`spawn_child(...) -> SessionInfo`, `emit(name, data)` (événement `custom`).
Exceptions dédiées : `ToolFatal` (arrête le tour, D63).

### 1.2 Façade `Agents`

```python
from bouture import Agents
from bouture.models.litellm import LiteLLMClient

agents = Agents[AppDeps](
    engine=engine,  # AsyncEngine (postgresql+asyncpg / +psycopg)
    deps=AppDeps(engine=engine, storage=blob_storage),
    model_client=LiteLLMClient(api_keys={"anthropic": "..."}),
    # storage=MemoryStorage()               # pour les tests (remplace engine)
    # runner=InProcessRunner()              # défaut
)
agents.register(character, world)


@agents.resolver("world")  # D55
async def _(ref: Ref, ctx: ResolveContext[AppDeps]) -> str: ...


agents.data_type("portrait_brief", PortraitBrief, version=1, migrations={})  # inputs Data (D54)


@agents.on_purge  # D68
async def _(ctx: PurgeContext[AppDeps], owner: str, session_ids: list[UUID]) -> None: ...


await agents.setup(auto_migrate=True)  # crée/migre le schéma, démarre LISTEN
await agents.shutdown()  # clôt les tours en cours (server_shutdown)
```

### 1.3 Opérations sur les sessions

Toutes prennent `owner` ; session d'un autre propriétaire → `NotFound` (D61).

| Méthode | Retour | Notes |
|---|---|---|
| `create_session(kind, owner, *, inputs=None, title=None, metadata=None, model=None, model_settings=None)` | `SessionInfo` | crée `root` + snapshot initial, puis les messages de `initial_messages` (D96) |
| `get_session(id, owner)` | `SessionInfo` | inclut `status`, `pending`, `leaf_entry_id` |
| `list_sessions(owner, *, kind=None, origin_session_id=None, archived=False, limit=50, cursor=None, object_id=None, metadata=None)` | `Page[SessionInfo]` | tri `updated_at desc` ; `metadata` = inclusion JSON (D102) |
| `rename(id, owner, title)` / `update_metadata(id, owner, metadata)` / `set_model(id, owner, model\|None)` | `SessionInfo` | |
| `set_model_settings(id, owner, settings)` | `SessionInfo` | liste blanche `overridable_settings`, sinon `InvalidSettings` (D92) |
| `archive(id, owner)` / `unarchive(id, owner)` | `SessionInfo` | |
| `clone(id, owner, *, title=None, entry_id=None)` | `SessionInfo` | un seul chemin (D17, D98) |
| `import_session(owner, kind, *, entries, leaf_id=None, publications=(), …)` | `SessionInfo` | import d'historique, ids et dates conservés (D104) |
| **Tours** | | |
| `send_message(id, owner, content)` | `TurnHandle` | `content: str \| list[ContentBlock]` |
| `answer(id, owner, answer)` | `TurnHandle` | réponse à `pending[0]` |
| `fork(id, owner, entry_id, content)` | `TurnHandle` | D14 |
| `regenerate(id, owner, entry_id)` | `TurnHandle` | D15 |
| `stop(id, owner)` | `None` | idempotent |
| `subscribe(id, owner, *, after=None)` | `AsyncIterator[Event]` | rejoue puis suit en direct (D24) |
| **Arbre** | | |
| `get_tree(id, owner, *, include_hidden=False, labeled=False)` | `list[EntryView]` | arbre complet, sans contenu d'état ni corbeille |
| `get_branch(id, owner, *, entry_id=None)` | `list[EntryView]` | root → entry (défaut : feuille) |
| `siblings(id, owner, entry_id)` | `Siblings(entries, index)` | |
| `switch_branch(id, owner, entry_id)` | `SessionInfo` | D16 |
| `rollback(id, owner, entry_id)` | `SessionInfo` | checkpoint requis sinon `InvalidTarget` |
| `append_message(id, owner, role, content, *, parent_entry_id=None)` | `EntryView` | sans appel LLM, checkpoint (D96) |
| `edit_entry(id, owner, entry_id, content)` | `EntryView` | copie sœur `derived_from`, suite recopiée (D97) |
| `label_entry(id, owner, entry_id, label)` | `EntryView` | point de sauvegarde nommé (D95) |
| `set_entry_metadata(id, owner, entry_id, metadata)` | `EntryView` | fusion, `None` supprime (D101) |
| `hide_branch(id, owner, entry_id)` / `restore_branch(…)` | `list[UUID]` | corbeille (D100) |
| **Inputs** | | |
| `update_inputs(id, owner, inputs)` | `EntryView` | entrée `input_update` (D99) |
| `get_inputs(id, owner, *, entry_id=None)` | `dict[str, InputView]` | inputs effectifs sur la branche |
| **Aperçu** | | |
| `preview_request(id, owner, content=None)` | `ModelRequest` | sans tour ni écriture (D93) |
| `preview(id, owner, content=None, *, expose_prompt=True)` | `RequestPreview` | + `estimate` du client |
| **État** | | |
| `get_state(id, owner, *, entry_id=None)` | `StateView(state, state_version, entry_id)` | migré à la lecture |
| `edit_state(id, owner, state, *, amend=False)` | `EntryView` | D43 ; `amend` réécrit le dernier `state_edit` (D94) |
| **Publication** | | |
| `publish(id, owner)` | `PublishedRef(kind, object_id, version)` | D50–D53 |
| `list_publications(owner, kind, object_id)` | `list[PublicationInfo]` | |
| `get_published_state(ref, owner)` | `BaseModel` | pratique pour un résolveur ; l'autorisation reste au projet |
| **Admin** (non exposé en routes) | | |
| `purge_session(id)` / `purge_owner(owner)` | `None` | D68 |

`TurnHandle` : `turn_id`, `session_id`, `events()` (raccourci de `subscribe(after=<début du tour>)`).

**Erreurs** (`bouture.errors`) : `NotFound`, `SessionBusy`, `InvalidTarget`,
`NotAwaitingInput`, `StateValidationError`, `StateVersionError`, `ResolveError`,
`UnknownKind`, `SchemaOutdated`, `InvalidSettings`.

**Crochets d'intégration** (D96, D101) : `@definition.initial_messages`
(`InitContext` → `[(role, content)]`) et `@definition.on_entries`
(`EntriesContext` avec `db`, entrées du tour → `{entry_id: metadata}`), ce dernier appelé dans
la transaction du commit.

### 1.4 Format neutre des messages (D2)

```python
ContentBlock = TextBlock(text) | ImageBlock(ref: str, media_type)   # ref → résolu par le projet
AssistantBlock = TextBlock | ReasoningBlock(text, signature?) | ToolCallBlock(id, name, args: dict)
UserMessage(content: list[ContentBlock])
AssistantMessage(content: list[AssistantBlock], interrupted: bool, usage: Usage | None)
ToolResult(tool_call_id, content: str | list[ContentBlock], is_error: bool)
```

`ModelClient.stream(request: ModelRequest) -> AsyncIterator[ModelEvent]`, avec
`ModelEvent = TextDelta | ReasoningDelta | ToolCallDelta | ToolCallComplete | Usage | Done(stop_reason)`.
`ModelRequest.context` (`CallContext(owner, session_id, kind, turn_id)`) identifie l'appel ;
méthode optionnelle `estimate(request) -> Estimate(input_tokens, cost)` (D92, D93).
`LiteLLMClient(credentials=async (request) -> dict)` fournit les identifiants à chaque appel.
`FakeModelClient(script=[...])` pour les tests : réponses scriptées, y compris tool_calls et pauses artificielles.

### 1.5 Contexte par défaut (D21, D42, D56)

```
system      : system_prompt
messages    : [inputs résolus — un bloc par input, stable]
              + branche active convertie (state_edit / child_result → notes courtes)
              + [bloc d'état courant via render_state — éphémère, en queue]
tools       : outils de la définition
```

---

## 2. Schéma Postgres (schéma `bouture`)

```sql
CREATE TABLE bouture.schema_version (
  version     int PRIMARY KEY,
  applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE bouture.sessions (
  id                  uuid PRIMARY KEY,
  kind                text NOT NULL,
  owner               text NOT NULL,
  title               text,
  metadata            jsonb NOT NULL DEFAULT '{}',
  model_override      text,
  model_settings      jsonb NOT NULL DEFAULT '{}',  -- réglages surchargés (D92)
  leaf_entry_id       uuid,                       -- FK différée vers entries
  object_id           uuid UNIQUE,                -- attribué à la 1re publication (D51)
  -- clone (D17)
  cloned_from_session_id uuid REFERENCES bouture.sessions(id),
  cloned_from_entry_id   uuid,
  -- origine enfant (D57)
  origin_session_id   uuid REFERENCES bouture.sessions(id),
  origin_entry_id     uuid,
  origin_handler      text,
  origin_payload      jsonb,
  -- pause (D30, D32)
  pending             jsonb NOT NULL DEFAULT '[]',  -- [{tool_call_id, tool_name, question, choices, payload}]
  -- verrou / stop (D25, D26)
  running_turn_id     uuid,
  lock_expires_at     timestamptz,
  cancel_requested    boolean NOT NULL DEFAULT false,
  archived_at         timestamptz,
  created_at          timestamptz NOT NULL DEFAULT now(),
  updated_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON bouture.sessions (owner, kind, archived_at, updated_at DESC);
CREATE INDEX ON bouture.sessions (origin_session_id);
CREATE INDEX ON bouture.sessions USING gin (metadata jsonb_path_ops);  -- D102

CREATE TABLE bouture.session_inputs (            -- immuables (D54)
  session_id   uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  name         text NOT NULL,
  mode         text NOT NULL CHECK (mode IN ('ref','data')),
  kind         text NOT NULL,
  object_id    uuid,            -- mode = ref
  version      int,             -- mode = ref (toujours épinglée)
  data         jsonb,           -- mode = data
  data_version int,             -- mode = data (D44)
  PRIMARY KEY (session_id, name)
);

CREATE TABLE bouture.states (                    -- snapshots, partagés entre clones
  id            uuid PRIMARY KEY,
  kind          text NOT NULL,
  state_version int  NOT NULL,
  data          jsonb NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE bouture.entries (
  id            uuid PRIMARY KEY,                -- UUIDv7 (ordre temporel), générée avant écriture
  session_id    uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  parent_id     uuid REFERENCES bouture.entries(id),
  type          text NOT NULL,                   -- root|user|assistant|tool_result|state_edit|child_result|…
  content       jsonb NOT NULL,                  -- message neutre / diff / référence enfant
  state_id      uuid REFERENCES bouture.states(id),  -- NULL si agent sans état
  is_checkpoint boolean NOT NULL,
  interrupted   boolean NOT NULL DEFAULT false,
  turn_id       uuid,
  usage         jsonb,
  created_at    timestamptz NOT NULL DEFAULT now(),
  label         text,                            -- point de sauvegarde nommé (D95)
  hidden_at     timestamptz,                     -- corbeille (D100)
  metadata      jsonb NOT NULL DEFAULT '{}',     -- données de l'application (D101)
  derived_from  uuid                             -- original d'un message édité (D97)
);
CREATE INDEX ON bouture.entries (session_id, parent_id, created_at);

CREATE TABLE bouture.turns (
  id          uuid PRIMARY KEY,
  session_id  uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  trigger     text NOT NULL,     -- message|answer|fork|regenerate|child_result
  status      text NOT NULL,     -- running|completed|awaiting_input|interrupted|error|max_steps|crashed|server_shutdown
  usage       jsonb,
  error       jsonb,
  started_at  timestamptz NOT NULL DEFAULT now(),
  ended_at    timestamptz
);
CREATE INDEX ON bouture.turns (session_id, started_at DESC);

CREATE TABLE bouture.turn_events (               -- tampon d'affichage, purgé (D24)
  id          bigserial PRIMARY KEY,             -- = SSE id / Last-Event-ID
  session_id  uuid NOT NULL,
  turn_id     uuid NOT NULL,
  type        text NOT NULL,
  data        jsonb NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON bouture.turn_events (session_id, id);

CREATE TABLE bouture.publications (              -- registre des versions (D50)
  kind         text NOT NULL,
  object_id    uuid NOT NULL,
  version      int  NOT NULL,
  session_id   uuid NOT NULL REFERENCES bouture.sessions(id),
  entry_id     uuid NOT NULL REFERENCES bouture.entries(id),
  state_id     uuid NOT NULL REFERENCES bouture.states(id),
  owner        text NOT NULL,
  published_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (object_id, version)
);
CREATE INDEX ON bouture.publications (kind, owner);

CREATE TABLE bouture.pending_child_results (     -- file quand le parent est occupé (D59)
  id                uuid PRIMARY KEY,
  parent_session_id uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  child_session_id  uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  object_id         uuid NOT NULL,
  version           int  NOT NULL,
  created_at        timestamptz NOT NULL DEFAULT now(),
  applied_at        timestamptz,
  applied_entry_id  uuid
);
CREATE INDEX ON bouture.pending_child_results (parent_session_id) WHERE applied_at IS NULL;
```

Notes :
- **Une transaction par tour** (D29) : `INSERT entries…`, `INSERT states` (si changé),
  `UPDATE sessions SET leaf_entry_id, pending, running_turn_id = NULL`, `UPDATE turns`,
  application des `pending_child_results`.
- Prise du verrou : `UPDATE sessions SET running_turn_id=$t, lock_expires_at=now()+'30s'
  WHERE id=$s AND (running_turn_id IS NULL OR lock_expires_at < now()) RETURNING …`.
  Un bail expiré repris déclenche d'abord la clôture `crashed` du tour précédent (D28).
- Canaux `NOTIFY` : `bouture_events` (payload `session_id:event_id`) et `bouture_cancel` (payload `turn_id`).
- L'arbre d'une session se charge en une requête (`WHERE session_id = $1`), la branche active se reconstruit en mémoire.
- Migrations : `apply_migrations(connection, target=None)` (synchrone, idempotent) pour une
  révision Alembic, `SCHEMA_VERSION` ; `setup(auto_migrate=False)` vérifie en lecture seule (D103).
- `entries.type = 'input_update'` : `content = {"inputs": {name: {mode, kind, object_id, version, data, data_version}}}` (D99).

---

## 3. Événements de streaming

Enveloppe commune (une ligne SSE `id:` = `id`, `event:` = `type`, `data:` = JSON) :

```json
{ "id": 1042, "session_id": "…", "turn_id": "…", "type": "text_delta", "data": { … }, "ts": "2026-10-11T12:00:00Z" }
```

| `type` | `data` | Quand |
|---|---|---|
| `turn_start` | `{trigger, parent_entry_id}` | début de tour |
| `user_message` | `{entry_id, content}` | message utilisateur enregistré (D81) |
| `message_start` | `{entry_id, role: "assistant"}` | début d'un appel LLM (entry_id pré-attribué) |
| `text_delta` | `{entry_id, text}` | texte (regroupé) |
| `reasoning_delta` | `{entry_id, text}` | raisonnement, si le modèle en produit |
| `tool_call_delta` | `{entry_id, index, tool_call_id, name, args_delta}` | arguments d'outil en cours, si `stream_tool_args` (D105) |
| `tool_call` | `{entry_id, tool_call_id, name, args}` | appel d'outil complet |
| `message_end` | `{entry_id, interrupted, usage}` | fin d'un appel LLM |
| `tool_result` | `{entry_id, tool_call_id, name, is_error, content}` | après exécution d'un outil |
| `state_changed` | `{patch: [...JSON Patch], state_version}` | état modifié (outil, handler, answer) |
| `ask_user` | `{tool_call_id, tool_name, question, choices, payload_preview?}` | pause (suivi de `turn_end{awaiting_input}`) |
| `child_spawned` | `{session_id, kind}` | session enfant créée |
| `child_result_applied` | `{entry_id, child_session_id, object_id, version}` | résultat d'enfant appliqué |
| `custom` | `{name, data}` | `ctx.emit()` d'un outil |
| `entry_metadata` | `{entries: {entry_id: metadata}}` | posé par `on_entries`, juste avant `turn_end` (D101) |
| `turn_end` | `{reason, leaf_entry_id, usage, error?}` | toujours le dernier événement d'un tour |

`reason` ∈ `completed | awaiting_input | interrupted | error | max_steps | crashed | server_shutdown`.
`error` = `{code, message}` assaini (D64).

Un résultat d'enfant appliqué alors que le parent est inactif produit un mini-tour
(`turn_start{trigger:"child_result"}` … `turn_end`), pour que tout passe par le même flux.

Exemple — tour avec outil :
```
turn_start → user_message → message_start(A1) → text_delta… → tool_call(set_name) → message_end(A1)
→ tool_result(T1) → state_changed → message_start(A2) → text_delta… → message_end(A2)
→ turn_end{completed}
```

---

## 4. Module de routes (`bouture.fastapi`, extra `[fastapi]`)

```python
from bouture.fastapi import make_router

app.include_router(make_router(agents, current_user=get_current_user), prefix="/agents")
```

Aucune logique : chaque route = un appel au cœur avec `owner = current_user()`.
Mapping des erreurs : `NotFound`→404, `SessionBusy`→409, `InvalidTarget`/`NotAwaitingInput`→409,
`StateValidationError`/`ResolveError`/`InvalidSettings`→422, `UnknownKind`→404.
`make_router(agents, current_user, expose_prompt=True)` : `False` masque le prompt de `/preview`.

| Méthode | Chemin | Cœur |
|---|---|---|
| `POST` | `/sessions` | `create_session` |
| `GET` | `/sessions?kind=&archived=&cursor=&object_id=&metadata=<json>` | `list_sessions` |
| `GET` / `PATCH` | `/sessions/{id}` | `get_session` / `rename`+`update_metadata`+`set_model`+`set_model_settings` |
| `POST` | `/sessions/{id}/archive` · `/unarchive` · `/clone` (body `{title?, entry_id?}`) | … |
| `POST` | `/sessions/{id}/preview` (body `{content?}`) | `preview` |
| `POST` | `/sessions/{id}/messages` | `send_message` → **SSE** du tour (fermer = détacher) |
| `POST` | `/sessions/{id}/answer` | `answer` → SSE |
| `POST` | `/sessions/{id}/entries/{eid}/fork` | `fork` → SSE |
| `POST` | `/sessions/{id}/entries/{eid}/regenerate` | `regenerate` → SSE |
| `POST` | `/sessions/{id}/stop` | `stop` |
| `GET` | `/sessions/{id}/events` (+ `Last-Event-ID`) | `subscribe` → SSE |
| `GET` | `/sessions/{id}/tree?include_hidden=&labeled=` · `/branch?entry_id=` · `/entries/{eid}/siblings` | … |
| `POST` | `/sessions/{id}/entries` (body `{role, content, parent_entry_id?}`) | `append_message` |
| `PATCH` | `/sessions/{id}/entries/{eid}` (body `{content}`) | `edit_entry` |
| `PUT` | `/sessions/{id}/entries/{eid}/label` (body `{label}`) | `label_entry` |
| `PATCH` | `/sessions/{id}/entries/{eid}/metadata` (body `{metadata}`) | `set_entry_metadata` |
| `POST` | `/sessions/{id}/entries/{eid}/hide` · `/restore` | `hide_branch` / `restore_branch` |
| `GET` / `PATCH` | `/sessions/{id}/inputs?entry_id=` (body `{inputs}`) | `get_inputs` / `update_inputs` |
| `POST` | `/sessions/{id}/switch` · `/rollback` (body `{entry_id}`) | … |
| `GET` / `PUT` | `/sessions/{id}/state?entry_id=` (body `{state, amend?}`) | `get_state` / `edit_state` |
| `POST` | `/sessions/{id}/publish` | `publish` |
| `GET` | `/publications/{kind}/{object_id}` | `list_publications` |

---

## 5. Livrables et tests

- `src/bouture/` (cœur), `src/bouture/fastapi/` (extra), `examples/character_world/`
  (deux kinds ; `read_world` lit un monde publié ; `propose_portrait` montre la pause).
- Tests (faux LLM + mémoire) : tour simple · chat sans outil (1 appel) · outil qui modifie
  l'état · état invalide annulé · pause/reprise (y compris « redémarrage » = nouvelle instance
  `Agents` sur le même stockage) · message pendant une pause · rollback · fork · regenerate ·
  siblings/switch · clone (branche active seule) · édition manuelle · stop (streaming, outil,
  entre étapes) · déconnexion sans stop · bail expiré → `crashed` · publication + idempotence ·
  `on_publish` qui échoue → rollback · enfant + file d'attente · migration d'état ·
  isolation entre propriétaires · routes (TestClient).
- Intégration (`tests/test_integration.py`, D92–D105) : réglages par session et contexte
  d'appel · aperçu sans tour (identique à la requête réelle) · sauvegarde automatique amendée ·
  points nommés · `initial_messages` / `append_message` · édition sans régénération · clone
  depuis une entrée · `update_inputs` et rollback · corbeille · `on_entries` et métadonnées ·
  recherche par objet / métadonnées · import · `tool_call_delta` opt-in. Routes associées,
  `LiteLLMClient` (identifiants par appel, estimation) et migrations (`tests/test_migrations.py` :
  vérification en lecture seule, idempotence).
- Tests de conformité stockage sur mémoire **et** Postgres (sautés sans `BOUTURE_TEST_PG_URL`).

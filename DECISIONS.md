# bouture — Décisions de conception

Journal des décisions prises avant l'implémentation. Chaque ligne est tranchée ;
la changer demande une nouvelle décision explicite.

## Fondations

| # | Décision |
|---|---|
| D1 | Boucle agent maison en Python (pas de Pi / OpenCode en sous-processus). |
| D2 | Format de messages **neutre**, propre à la lib, stocké dans l'arbre. Jamais de format fournisseur en base. |
| D3 | Protocole `ModelClient` ; adaptateur par défaut **LiteLLM** en extra `[litellm]`. Le cœur n'en dépend pas. |
| D4 | Python ≥ 3.12. Cœur : `pydantic>=2`, `sqlalchemy[asyncio]>=2`. Extras : `litellm`, `fastapi`, `asyncpg`, `psycopg`. |
| D5 | Le cœur n'importe jamais FastAPI. |

## Arbre et branches

| # | Décision |
|---|---|
| D10 | Une entrée par message/événement : `root`, `user`, `assistant`, `tool_result`, `state_edit`, `child_result` (+ `summary`, `input_update` possibles plus tard sans migration : `type` texte, contenu JSONB). |
| D11 | Chaque entrée référence un snapshot (`state_id`). Nouveau snapshot **uniquement si l'état change** ; sinon réutilisation. Pas d'état → pas de snapshot. |
| D12 | Checkpoints : `root`, fin de tour, `state_edit`, `child_result`, entrée `interrupted`. La feuille est toujours sur un checkpoint, sauf pendant une pause (sur l'`assistant(tool_call)`). |
| D13 | **rollback(entry)** : déplace la feuille sur un checkpoint. Rien n'est supprimé. |
| D14 | **fork(entry, contenu)** : `entry` est un message `user` ; crée un `user` frère (même parent) et lance un tour depuis l'état du parent. |
| D15 | **regenerate(entry)** : remonte au message `user` du tour ; nouvelle réponse enfant de ce `user` (sœur de l'ancienne), avec l'état au moment de ce `user`. Tout le tour est rejoué. |
| D16 | **siblings(entry)** liste les frères ; **switch_branch(entry)** place la feuille sur le descendant le plus récent (enfant le plus récent à chaque niveau). |
| D17 | **clone(session)** : nouvelle session indépendante. ⚠️ **Copie uniquement la branche active** (root → feuille), renumérotée ; snapshots partagés ; inputs copiés à l'identique. Les branches mortes ne sont pas copiées. Pas de clone entre propriétaires. |

## Exécution des tours

| # | Décision |
|---|---|
| D20 | Agent sans outil = un seul appel LLM, sans surcoût notable. |
| D21 | Crochet `build_context` par définition ; défaut = système + inputs résolus + branche active + bloc d'état en queue. |
| D22 | **Seul un stop explicite arrête un tour.** F5, déconnexion SSE, onglet fermé n'ont aucun effet. |
| D23 | Le tour tourne en tâche de fond (`TurnRunner`, défaut in-process `asyncio`), détaché de la requête. |
| D24 | Événements journalisés dans `turn_events` (deltas regroupés ~100 ms / 200 car.) + `NOTIFY`. Reconnexion via `after` / `Last-Event-ID`. Journal purgé ~1 h après la fin du tour (l'arbre est la vérité). |
| D25 | Verrou = bail en base (`running_turn_id`, `lock_expires_at`), renouvelé à chaque étape. Second message pendant un tour → `SessionBusy` (409). |
| D26 | Stop : registre local `turn_id → Task` + `cancel_requested` + `NOTIFY` inter-processus ; relu entre les étapes. L'annulation ferme le stream HTTP du fournisseur. |
| D27 | Stop pendant le streaming → texte partiel enregistré `interrupted`, tool_calls partiels jetés. Stop pendant un outil → outil annulé, `tool_result` d'erreur « interrupted ». Stop entre étapes → plus d'appel LLM. |
| D28 | Processus mort en plein tour : à expiration du bail, le tour est clos `crashed`, texte partiel reconstruit depuis `turn_events` et enregistré `interrupted`. **Pas de reprise automatique.** SIGTERM : `shutdown()` clôt proprement (`server_shutdown`). |
| D29 | Écriture d'un tour en **une transaction** : entrées + snapshot + feuille (+ résultats d'enfants en attente). |

## Pause / reprise

| # | Décision |
|---|---|
| D30 | Un outil renvoie `AskUser(question, choices, payload)`. La pause **termine** le tour (transaction : entrées, snapshot, feuille, `pending`) et libère le verrou. |
| D31 | Pas de ré-exécution de l'outil : `answer()` lance un nouveau tour qui appelle `on_answer(ctx, answer, payload)` (optionnel ; défaut = réponse brute comme `tool_result`). |
| D32 | Plusieurs questions dans un même message assistant : posées **une par une** (file `pending`). |
| D33 | Message utilisateur ou édition d'état pendant une pause → la question est annulée (`tool_result` « pas de réponse »), puis l'action est traitée. |
| D34 | Rollback / fork / regenerate pendant une pause : autorisés, `pending` abandonné. |

## État

| # | Décision |
|---|---|
| D40 | `ctx.state` = copie de travail mutable. Revalidation après chaque outil ; invalide → annulation des modifs de cet outil + erreur renvoyée au LLM. |
| D41 | Événement `state_changed` avec **JSON Patch** (RFC 6902). |
| D42 | État injecté **en queue de contexte**, éphémère, jamais dans le prompt système (cache de prompt). Rendu via `render_state` (défaut JSON). |
| D43 | `edit_state` crée une entrée `state_edit` (checkpoint, diff stocké). Le LLM en est informé par une note dans le contexte. Pendant un tour → 409. |
| D44 | `state_version` + `state_migrations` par définition. Migration **paresseuse à la lecture**, snapshots jamais réécrits. Version stockée > code → `StateVersionError`. S'applique aux inputs `Data`, pas aux `payload`. |
| D45 | L'état ne contient que des **références** vers les binaires. |

## Publication, inputs, enfants

| # | Décision |
|---|---|
| D50 | La lib tient le registre `publications (kind, object_id, version, …)` ; le projet tient le contenu via `on_publish`. |
| D51 | Un objet par session ; `object_id` attribué à la 1ʳᵉ publication ; un clone = nouvel objet. Republier un état identique est idempotent. |
| D52 | On publie l'état de la feuille (checkpoint, hors tour et hors pause). |
| D53 | `on_publish` s'exécute **dans la transaction** de la lib, avec une `AsyncSession` SQLAlchemy liée. Exception → tout est annulé. |
| D54 | Inputs **nommés** ; `Ref(kind, id, version=None)` est figé sur la dernière version à la création ; `Data(kind, value)` est validé et versionné. Immuables. |
| D55 | Résolveur par kind ; appelé à la création (échec immédiat) ; reçoit l'owner (autorisation = projet) ; résultat en cache LRU par processus. |
| D56 | Inputs placés après le système, avant l'historique. Pas de mise à jour d'input en v1 (`input_update` possible plus tard). |
| D57 | `ctx.spawn_child(kind, inputs, on_result=<nom>, payload, initial_message=None)` ; même owner ; événement `child_spawned`. |
| D58 | `child_handler` nommé sur le kind parent ; reçoit l'état **publié** de l'enfant (jamais sa session). |
| D59 | Chaque publication d'enfant → entrée `child_result` (checkpoint) sur la **feuille courante** du parent, LLM informé. Parent occupé/en pause → file persistée appliquée en fin de tour ou après la pause. |

## Hôte, erreurs, cycle de vie

| # | Décision |
|---|---|
| D60 | `deps` au niveau application, contexte générique `[State, Deps]`. Les outils ouvrent leur propre session DB. ⚠️ Les effets de bord hors état **ne sont pas annulés** par un rollback. |
| D61 | Le cœur prend `owner` explicitement partout ; session d'un autre → `NotFound` (pas `Forbidden`). Les routes injectent l'owner via la fonction « utilisateur courant » du projet. |
| D62 | Modèle (chaîne LiteLLM) sur la définition ; surcharge par session autorisée. |
| D63 | Exception d'outil → `tool_result` d'erreur, le tour continue ; `ToolFatal` arrête. Traces dans les logs uniquement. |
| D64 | Réessais LLM (3, backoff) **uniquement avant le premier token**. Messages d'erreur assainis. |
| D65 | `max_steps=25`, `tool_timeout=120s` par défaut ; pas de timeout de tour. |
| D66 | `usage` stocké par entrée `assistant`, cumulé dans `turn_end`. |
| D67 | Sessions : `title`, `metadata` libre, `archive`/`unarchive`. Crochet `on_turn_end` (après commit). |
| D68 | `purge_session` / `purge_owner` (suppression physique, RGPD) dans le cœur, **non exposés par les routes** ; crochet `on_purge`. |

## Persistance et outillage

| # | Décision |
|---|---|
| D70 | Postgres via SQLAlchemy Core async (dialecte `asyncpg` ou `psycopg` exigé pour LISTEN/NOTIFY), schéma dédié `bouture`. Implémentation mémoire pour les tests. |
| D71 | Migrations SQL embarquées et numérotées, appliquées par `setup()` sous advisory lock ; `auto_migrate=False` → erreur si le schéma est en retard. |
| D72 | Outillage : uv, ruff, pyright strict, pytest + pytest-asyncio. Tests de conformité stockage sur les deux implémentations ; Postgres sauté sans `BOUTURE_TEST_PG_URL`. |

## Écarts et précisions de l'implémentation

| # | Décision |
|---|---|
| D80 | Les inputs résolus sont ajoutés **à la fin du prompt système** (section « Références ») plutôt qu'en message séparé (précise D56). `build_context` permet de changer ce placement. |
| D81 | Nouvel événement `user_message` en début de tour (id de l'entrée créée), pour que les abonnés tardifs affichent le message. |
| D82 | `turns.parent_entry_id` : l'entrée à partir de laquelle le tour a démarré (sert au rattachement après stop/crash). |
| D83 | Réveil des abonnés : signal local (`Hub`) ou NOTIFY, puis **relecture en base** ; le signal ne transporte jamais l'événement. Repli par sondage (`poll_interval`). |
| D84 | L'événement `tool_result` porte `name` ; `ask_user` porte `tool_name`. |
| D85 | Les résultats d'enfants en file sont appliqués au commit du tour parent, ou via un mini-tour `child_result` si le parent est au repos. |
| D86 | Une exception d'outil **restaure l'état** d'avant l'appel (comme un état invalide). |
| D87 | `answer(..., tool_call_id=None)` : cible une question précise quand plusieurs sont en attente ; sinon la plus ancienne. |
| D88 | `list_publications(owner, kind, object_id)` : filtré par propriétaire et par kind. |
| D89 | Un outil qui renvoie `AskUser` en laissant un état invalide : la question **n'est pas posée**, l'erreur de validation revient au LLM comme résultat d'outil. |
| D90 | ruff ignore `E501` (le formateur s'en charge) et `ASYNC109` (`timeout=` fait partie de l'API). Tests psycopg sous Windows : boucle `Selector` (la boucle Proactor n'est pas supportée par psycopg). |
| D91 | L'exemple lit le monde publié via `Agents.get_published_state(ref, owner=)` (la version épinglée de l'input) ; `on_publish` copie aussi le monde dans la table projet `app_worlds` quand le stockage est Postgres. |

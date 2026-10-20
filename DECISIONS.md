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

## Intégration dans une application existante

Besoins remontés par l'intégration de bouture dans une application de chat et d'édition d'entités. Chaque ajout reste générique : aucun concept de l'application n'entre dans la lib.

| # | Décision |
|---|---|
| D92 | **Identifiants et réglages par appel.** `ModelRequest.context` (`CallContext` : owner, session, kind, tour) est rempli à chaque appel. `LiteLLMClient(credentials=async (request) -> dict)` fournit `api_key` / `api_base` / en-têtes **au moment de l'appel** : aucun secret n'est stocké, ni dans la base ni dans les événements, et un échec devient `ModelError(code="credentials_error")`. Les sessions peuvent surcharger des réglages d'échantillonnage (`model_settings`, à la création ou via `set_model_settings`), dans la liste blanche `AgentDefinition(overridable_settings=…)` (par défaut : temperature, top_p, top_k, frequency/presence_penalty, max_tokens, seed, stop). Toute autre clé lève `InvalidSettings` (422) : une session ne peut pas rediriger le modèle vers un autre endpoint. |
| D93 | **Aperçu de requête.** `preview_request` construit la requête du prochain appel exactement comme un tour (même `build_context`, mêmes inputs effectifs), sans verrou ni écriture ; `content` simule un message à venir. `preview` y ajoute l'estimation du client (méthode **optionnelle** `ModelClient.estimate` → `Estimate(input_tokens, cost)` ; LiteLLM : tokenizer + table de prix). La route `POST /sessions/{id}/preview` renvoie le prompt complet, sauf si `make_router(expose_prompt=False)`. |
| D94 | **Sauvegarde automatique.** `edit_state(..., amend=True)` réécrit la feuille au lieu d'empiler une entrée quand c'est un `state_edit` sans enfant, sans nom, non publié, sans question en attente. Le patch est recalculé depuis le parent ; le snapshot orphelin est supprimé, et l'état du parent est réutilisé si l'on y revient. C'est la seule mutation d'une entrée existante : elle ne touche que du contenu que personne d'autre n'a pu référencer. Le débit (debounce) reste côté client. |
| D95 | **Points de sauvegarde nommés** = `label` sur un checkpoint (`label_entry`, `get_tree(labeled=True)`). Pas de table dédiée : le checkpoint est déjà le point stable, et un rollback / clone vers lui suffit. Un `state_edit` nommé n'est plus amendable (D94). |
| D96 | **Messages sans appel LLM.** Crochet `@definition.initial_messages` (messages posés à la création, en checkpoints, ex. un message d'accueil) et `append_message(role, content, parent_entry_id=)` (checkpoint, devient la feuille). Sur la feuille, les questions en attente sont annulées comme pour `send_message`. Un message assistant posé ainsi ne contient que du texte. |
| D97 | **Éditer un message sans régénérer** = copie sœur (`derived_from` → l'original, qui reste dans l'arbre) et recopie de la suite de la branche (états partagés par id, dernier élément forcé checkpoint). On reste fidèle à D3 (entrées immuables) : l'historique de l'édition est un frère, consultable via `siblings`. Côté assistant, seul le texte est remplacé ; raisonnement et appels d'outils sont conservés. |
| D98 | **Clone depuis un point choisi** : `clone(entry_id=)`. Coupe à l'entrée si c'est un checkpoint ou un message utilisateur (pour régénérer dans la copie), sinon au checkpoint au-dessus. |
| D99 | **Changer les références** : `update_inputs` ajoute une entrée `input_update` (checkpoint) qui recouvre les inputs de la création. Les inputs effectifs dépendent donc de la branche : un rollback avant l'entrée rétablit les anciens. Le LLM voit une note « Les références suivantes ont changé ». `get_inputs(entry_id=)` les expose. La table `inputs` reste celle de la création (D54 inchangé). |
| D100 | **Corbeille** : `hide_branch` / `restore_branch` posent / retirent `hidden_at` sur un sous-arbre (même horodatage, donc une restauration ne ressort pas ce qui avait été masqué séparément). Refusé pour la racine et le chemin actif (409). Une branche masquée disparaît de `get_tree`, `siblings` et `latest_descendant` ; fork, regenerate, switch, rollback et edit y sont refusés. La suppression définitive reste `purge_*` (D68). |
| D101 | **Données de l'application sur les entrées** : `metadata` jsonb par entrée, visible dans `EntryView`. Crochet `@definition.on_entries` appelé **dans la transaction du commit** (avec `db`), pour lier les entrées du tour à des objets de l'application ; il émet un événement `entry_metadata` avant `turn_end`, et un échec est journalisé sans casser le tour. `set_entry_metadata` (fusion de premier niveau, `None` supprime) pour les mises à jour ultérieures. |
| D102 | **Retrouver une session** : `list_sessions(object_id=, metadata=)`. `object_id` donne les sessions qui publient cet objet ; `metadata` est un filtre d'inclusion JSON (`@>`, index GIN). C'est ce que le projet utilise pour relier ses objets aux sessions, sans clé étrangère vers lui. |
| D103 | **Migrations pilotées par l'application** : `bouture.storage.postgres.apply_migrations(connection, target=None)` (synchrone, idempotent, sous verrou consultatif) s'appelle depuis une révision Alembic via `op.get_bind()`, et `SCHEMA_VERSION` permet de vérifier. Au démarrage, `setup(auto_migrate=False)` vérifie **en lecture seule** et lève `SchemaOutdated`. Pas de retour arrière. L'`env.py` d'Alembic exclut le schéma `bouture` de l'autogénération. |
| D104 | **Import d'historique** : `import_session(entries=[ImportedEntry…], publications=[ImportedPublication…])`, en une transaction. Les ids et dates sont conservés (l'arbre est ordonné par `(created_at, id)`), les états des `state_edit` sont validés et les patchs calculés. Ni LLM, ni `initial_messages`, ni `on_publish` ne sont appelés. La feuille doit être un checkpoint ; les versions publiées doivent être croissantes, sur un objet encore vierge. |
| D105 | **Streaming des arguments d'outil**, opt-in par définition (`stream_tool_args=True`) : événement `tool_call_delta` (`entry_id`, `index`, `tool_call_id`, `name`, `args_delta`), fusionné comme les autres deltas. Désactivé par défaut, parce qu'il double le volume d'événements pour un usage d'affichage. `ToolCallDelta` fait partie du protocole `ModelClient` ; un client qui ne l'émet pas reste conforme. |
| D106 | **Abonnement Claude** : `ClaudeCodeClient` (extra `claude-code`) passe par le Claude Agent SDK, authentifié par l'abonnement de Claude Code. L'abonnement n'est pas utilisable directement contre l'API, d'où ce détour. Claude Code n'accepte pas d'historique fabriqué : la conversation est sérialisée en transcript dans un seul message (images par référence, raisonnement omis). Les outils sont exposés en serveur MCP ; un hook `PreToolUse` renvoie `defer`, si bien que Claude Code s'arrête au premier appel sans l'exécuter et que la boucle d'outils reste celle de bouture. Outils intégrés, réglages et CLAUDE.md de l'utilisateur sont désactivés (`tools=[]`, `setting_sources=[]`, répertoire de travail vide). `ANTHROPIC_API_KEY` est masquée par défaut pour forcer l'abonnement. Usage personnel et développement uniquement. |

## Stockage SQLite

Pour les applications locales mono-utilisateur (Windows, Linux), souvent synchrones (FastAPI + SQLAlchemy sync + SQLite). Complète D70–D72, qui restent vraies pour Postgres.

| # | Décision |
|---|---|
| D107 | **`SQLiteStorage(path)`** (`bouture.storage.sqlite`, extra `sqlite`, pilote `aiosqlite`) implémente tout `Storage` / `StorageTx` et passe la même suite de conformité que la mémoire et Postgres. Il crée et possède son moteur : l'URL est construite par `URL.create` à partir d'un `pathlib.Path` résolu, jamais analysée comme une chaîne (espaces, `#`, `%`, lettres de lecteur Windows). SQLite n'a pas de schéma : **les tables sont préfixées `bouture_`**, si bien que **l'hôte choisit le fichier**. Recommandé : un fichier dédié (aucune contention avec les écritures de l'application, sauvegarde et suppression séparées). Le fichier de l'application est possible quand `on_publish` / `on_entries` doivent écrire dans les tables de l'hôte **dans la même transaction** (D53, D101) ; `ctx.db` est toujours lié au fichier de bouture. |
| D108 | **Transactions.** Mode WAL (les lecteurs de l'hôte ne bloquent pas bouture), `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout` réglable (30 s). **Toutes** les transactions de bouture s'ouvrent en `BEGIN IMMEDIATE`, pas seulement les écritures : le protocole ne distingue pas les transactions en lecture seule, et une transaction différée qui lit puis écrit échoue (`SQLITE_BUSY`) au lieu d'attendre. La lecture `for_update` du bail suivie de son écriture (D25) reste donc atomique face à un autre connecteur. Dans le processus, un verrou asyncio sérialise les transactions de bouture, pour qu'elles ne se disputent pas le verrou SQLite. |
| D109 | **Types et JSON.** uuid en texte canonique, dates UTC en texte à largeur fixe (l'ordre du texte est celui des dates, ce qui garde `(created_at, id)` et la pagination), JSON en texte. Le filtre `metadata` (D102) est une fonction `bouture_json_contains` enregistrée sur chaque connexion, qui exécute **le même code que la mémoire** (`json_contains`) : `json_extract` ne sait pas exprimer l'inclusion dans des listes imbriquées. La sémantique est celle de `@>` : clés imbriquées incluses récursivement, listes comparées sans ordre ni doublons, `null` égal seulement à `null`, et un booléen n'est jamais égal à un nombre (corrige la mémoire, qui confondait `true` et `1`). Pas d'index : balayage des sessions du propriétaire, acceptable en mono-utilisateur. |
| D110 | **Notifications sans LISTEN/NOTIFY.** Au commit, le stockage prévient les abonnés **du processus** (`Hub`, registre d'annulation) : réveil des flux et stop sont immédiats dans le processus. Un autre connecteur sur le même fichier n'est pas prévenu : ses abonnés relisent par sondage (`poll_interval`, D83), un stop qu'il demande n'est vu qu'au prochain battement du bail (`cancel_requested`, au plus `lock_ttl / 3`), un tour abandonné est récupéré à l'expiration du bail (D28). **Configuration prise en charge : un seul processus par fichier de base** (pas de workers multiples). Le stop inter-processus n'est pas garanti ; le stop dans le processus l'est. |
| D111 | **Migrations SQLite.** Jeu propre au dialecte (`bouture/storage/sqlite_migrations/`, numérotation indépendante de Postgres, table `bouture_schema_version`), appliqué par `setup()` dans une transaction `BEGIN IMMEDIATE`. `setup(auto_migrate=False)` vérifie en lecture seule et lève `SchemaOutdated`, sans créer le fichier s'il est absent. `apply_migrations(connection, target=None)` et `schema_version(connection)` sont synchrones, appelables depuis une révision Alembic (comme D103). SQLite n'a pas de verrou consultatif, et le pilote `sqlite3` peut appliquer le DDL hors transaction : chaque instruction est donc rejouable (`IF NOT EXISTS`), et une future migration devra le rester. L'`env.py` d'Alembic exclut les tables `bouture_*` de l'autogénération. |

## Images et contenu riche

Un outil qui génère une image doit pouvoir la montrer au modèle, sans que l'arbre ni les événements ne stockent d'octets (D45).

| # | Décision |
|---|---|
| D112 | **Résultats d'outils riches.** Un outil renvoie `str` (ou toute valeur, sérialisée comme avant) ou un bloc / une liste non vide de `TextBlock` / `ImageBlock`. `ToolResultMessage.content` devient `str \| list[bloc]` : les entrées déjà stockées (chaîne) restent lisibles telles quelles, `.text` et `.images` donnent une vue uniforme. Une liste de dicts reste du JSON, pas des blocs. Les blocs ne portent que des **références** (D45). L'événement `tool_result` garde `content` en texte (compatibilité des clients) et ajoute `blocks` (dumps des blocs, références seulement) quand le résultat est riche ; la reprise après crash (D28) reconstruit l'entrée depuis `blocks`. |
| D113 | **Chargement par l'hôte.** `@agents.image_loader async (ref, ImageContext) -> (octets, media_type)` (contexte : deps, owner, session, kind). Appelé dès qu'un outil renvoie une image : un échec (exception ou retour invalide) transforme le résultat en **`tool_result` d'erreur** listant les références, et l'état modifié par l'outil est restauré (comme D86) ; le tour continue. Rappelé à chaque construction de requête pour les images de l'historique (messages utilisateur **et** résultats d'outils, même chemin) : une image disparue entre-temps est remplacée, dans la seule copie envoyée, par `[image indisponible : ref]`, l'arbre gardant la référence. Cache par tour (une référence n'est chargée qu'une fois). Les octets ne vivent que dans `ModelRequest.images` (transitoire, jamais stocké ni émis). Sans loader, les images partent par leur référence (URL), comme avant. L'aperçu (D93) ne charge pas d'image. |
| D114 | **LiteLLM : placement des images d'outils**, `LiteLLMClient(tool_images=…)`. `"inline"` : contenu `text` + `image_url` (URL `data:`) dans le message `tool`, que LiteLLM convertit en blocs image du `tool_result` Anthropic. `"user"` : l'API OpenAI n'accepte que du texte dans un message `tool` ; le texte y indique `[image N : ref, jointe au message suivant]` et les images suivent dans un message utilisateur placé après le dernier résultat d'outil, chacune précédée de « Image N renvoyée par l'outil X (id) ». `"auto"` (défaut) : `"inline"` si le fournisseur résolu par LiteLLM est `anthropic`, `"user"` sinon. |
| D115 | **Claude Code : images en entrée streaming** (remplace la partie « images par référence » de D106). Quand la requête porte des images chargées, le prompt n'est plus une chaîne mais un itérable asynchrone d'**un** message utilisateur dont le contenu alterne texte du transcript et blocs `{"type": "image", "source": {"type": "base64", …}}`, insérés juste après leur marqueur `[image : ref]` ; le texte reste identique au transcript sans image. Vérifié sans appel réel : le SDK (`claude_agent_sdk/_internal/query.py`, `stream_input`) transmet le dict tel quel, et la CLI, lancée contre un faux serveur d'API (`ANTHROPIC_BASE_URL`), relaie le bloc image dans la requête. Effets de bord observés : la CLI ajoute un bloc texte `[Image: source: <chemin>]` et **écrit une copie temporaire de l'image** dans son dossier temporaire (`%TEMP%\claude\…\images\`). Sans image chargée, prompt texte inchangé. |

## Outils fournisseur

Recherche et lecture web exécutées par le fournisseur du modèle, pas par bouture : le tour les trace sans les exécuter. Bash, fichiers et exécution de code restent hors périmètre.

| # | Décision |
|---|---|
| D116 | **Déclaration et trace.** `AgentDefinition(provider_tools=["web_search", "web_fetch"])` ; un autre nom lève `ValueError` à la définition. La liste part dans `ModelRequest.provider_tools` (aussi quand `build_context` est personnalisé et la laisse vide). Le client émet `ProviderToolCall(id, name, args)` puis `ProviderToolResult(id, content, is_error)` ; le tour les range dans le message assistant en cours comme `ProviderToolBlock` (bloc à sa place, entre les textes), émet `provider_tool_call` / `provider_tool_result` (`entry_id`, `id`, …) et les rejoue à la reprise après crash (D28). Aucune entrée `tool_result` : ils ne comptent pas dans `max_steps` et ne passent pas par les hooks d'outils. Le résultat conservé est un **résumé texte** borné à `PROVIDER_RESULT_MAX` (20 000) caractères. |
| D117 | **Capacité par client.** Méthode optionnelle `provider_tools(model) -> frozenset[str]` sur le client ; absente = aucun outil (`supported_provider_tools(client, model)`). Vérifiée à `register` (modèle par défaut), `create_session(model=…)`, `import_session(model=…)` et `set_model` : `UnsupportedProviderTool` nomme le client, les outils manquants, le modèle et le kind. `FakeModelClient(provider_tools=…)` choisit ce qu'il déclare. |
| D118 | **Rejeu en texte.** Les fournisseurs ne réacceptent pas leurs blocs serveur d'un appel à l'autre de façon portable (contenu chiffré chez Anthropic, rien chez OpenAI) : un `ProviderToolBlock` passé est rejoué en texte, `<provider_tool id name>{args}</provider_tool>` suivi de `<provider_tool_result>…</provider_tool_result>` (`provider_tool_text`). Perte assumée : le modèle voit le résumé, pas les citations chiffrées. |
| D119 | **ClaudeCodeClient** (complète D106). Les outils demandés deviennent `tools=["WebSearch", "WebFetch"]` du CLI ; le hook `PreToolUse` répond `allow` pour eux et `defer` pour tout le reste (les outils de bouture reviennent toujours à bouture). Chaque appel web consomme un tour du CLI : `max_turns = 1 + provider_tool_turns` (défaut 8) quand le kind en demande, `1` sinon. `error_max_turns` dans ce cas donne `Done("max_turns")` et le tour se termine avec le texte déjà produit, sans erreur. Les `ToolUseBlock` WebSearch/WebFetch (ou `ServerToolUseBlock`) deviennent `ProviderToolCall`, leur `ToolResultBlock` (message utilisateur du CLI) `ProviderToolResult` ; pas de `ToolCallDelta` pour eux. L'usage s'additionne sur les messages d'API successifs d'un même appel. |
| D120 | **LiteLLMClient : outils hébergés Anthropic.** Pour le fournisseur `anthropic` seulement, `web_search` / `web_fetch` partent comme outils `web_search_20250305` / `web_fetch_20250910`, complétés par `LiteLLMClient(provider_tool_options={"web_search": {"max_uses": 3}})`. Dans le flux, LiteLLM les livre comme `tool_calls` d'id `srvtoolu_…` (ni `ToolCallDelta` ni `ToolCallComplete`) et leurs résultats dans `provider_specific_fields["web_search_results"]` (liste cumulative, recherche et lecture) ; `summarize_server_result` en tire le résumé (titres et URL ; URL, titre et texte ; `erreur : code`). Un appel sans résultat (flux coupé) est tracé sans résultat. Autres fournisseurs : non supporté (D117). |

## Outils longs et bail

| # | Décision |
|---|---|
| D121 | **`timeout=None` = sans limite** (remplace D65 pour les délais d'outils). `@d.tool(timeout=None)` n'est plus « hériter du délai de la définition » mais « aucun délai » ; omettre `timeout` hérite (sentinelle `INHERIT`). `AgentDefinition(tool_timeout=None)` supprime la limite par défaut du kind (120 s inchangé sinon). Un délai nul ou négatif lève `ValueError` (avant, `timeout=0` héritait silencieusement). Toujours pas de délai de tour. |
| D122 | **Bail pendant un outil** (précise D25). Le bail est renouvelé par une tâche de battement propre au tour, toutes les `lock_ttl / 3`, indépendante des étapes : un outil qui attend (I/O, `asyncio.to_thread`, sous-processus) garde le bail quelle que soit sa durée — vérifié pour un outil de 10 baux sous horloge simulée, et le même test sans battement finit `crashed`. Pas de battement supplémentaire à ajouter. Limite assumée : un outil qui **bloque la boucle** bloque aussi le battement ; au-delà de `lock_ttl` sans battement, un autre lecteur peut clore le tour `crashed` et, au retour, le battement voit le bail perdu et abandonne le tour (rien n'est écrit). Documenté : passer par `asyncio.to_thread` ou un sous-processus. Le `sleep` du battement est un point d'injection de module (`bouture.turn._sleep`), pour les tests seulement. |

## Hébergement en arrière-plan

| # | Décision |
|---|---|
| D123 | **`bouture.hosting.BackgroundAgents`** : `Agents` tourne sur un thread dédié (démon) avec sa propre boucle, `ProactorEventLoop` sous Windows (`loop_factory` remplaçable). Raison : un hôte synchrone (routes FastAPI `def`, SQLAlchemy synchrone) n'a pas de boucle à prêter, et un hôte qui impose `SelectorEventLoop` sous Windows (psycopg, certains réglages d'uvicorn) ne peut pas lancer de sous-processus ([asyncio, « Platform Support »](https://docs.python.org/3/library/asyncio-platforms.html#windows) : `SelectorEventLoop` ne gère pas les sous-processus sous Windows). Les outils, le stockage et les clients de modèle vivent tous sur cette boucle ; `start()` exécute `agents.setup()` dedans et relève son erreur dans le thread appelant. Enregistrer les définitions avant `start()`. Module séparé, non exporté par `bouture` : `from bouture.hosting import BackgroundAgents`. |
| D124 | **Ponts.** `submit` (→ `concurrent.futures.Future`), `await run(coro)` depuis une autre boucle (annuler l'appelant annule l'appel), `run_sync(coro, timeout)` depuis un thread (au-delà du délai : `TimeoutError` et appel annulé), `stream` / `stream_sync` pour les itérateurs (`handle.events()`, `subscribe`) : fermer l'itérateur côté hôte (`break`, déconnexion SSE) ferme l'itérateur côté bouture, aucune tâche ne reste. `bg.aio` est une vue de `Agents` dont les méthodes async passent par `run`, les générateurs par `stream`, et dont les `TurnHandle` sont rebranchés ; elle est typée `Agents[D]`. `make_router` accepte `Agents` ou `BackgroundAgents` (il prend alors `bg.aio`). Depuis la boucle de bouture elle-même (dans un outil), `run` / `stream` deviennent un simple `await` et `run_sync` / `stream_sync` lèvent `RuntimeError` au lieu de bloquer la boucle. |
| D125 | **Arrêt.** `shutdown(timeout)` (idempotent, bloquant) appelle `agents.shutdown(timeout)` sur la boucle de bouture — les tours en cours finissent `server_shutdown`, appels d'outils ouverts clos « Interrompu », stockage fermé —, puis arrête la boucle, annule ce qui reste (ponts non consommés), ferme les générateurs async et joint le thread. Interdit depuis le thread de bouture. Gestionnaires de contexte synchrone et async (`async with` passe par `asyncio.to_thread`). Après l'arrêt, tout pont lève `RuntimeError`. Le thread est démon : un hôte qui oublie `shutdown()` ne bloque pas la sortie du processus, mais ses tours seront repris `crashed` (D25). |
| D126 | **Appels ouverts dès l'enregistrement du message assistant** (correction, précise D27). Les appels d'outils d'un message assistant sont marqués ouverts dans la même étape synchrone que l'ajout du message. Avant, un arrêt ou un stop tombant pendant l'émission de `message_end` (une écriture, donc un `await`) laissait un `tool_call` sans `tool_result` dans la branche, que le fournisseur refuserait au tour suivant. Trouvé par le test d'arrêt de `BackgroundAgents` sur SQLite ; test de régression déterministe dans `test_turns.py`. |

## Coût et latence de ClaudeCodeClient

Étude menée sans appel réel : la CLI de Claude Code (`claude_agent_sdk` 0.2.168) lancée contre un faux serveur d'API Messages (`ANTHROPIC_BASE_URL`), qui journalise les corps de requête et rejoue des réponses SSE. Les chiffres réels (latence, tokens) restent à mesurer avec `scripts/bench_claude_code.py`, qui n'est lancé qu'avec `--run`.

| # | Décision |
|---|---|
| D127 | **Constat : le mode transcript met peu en cache, et le script de mesure.** La CLI envoie : blocs système (en-tête, « You are a Claude agent… », puis notre prompt système + protocole, avec `cache_control`), les outils `mcp__bouture__*`, puis **un** message utilisateur (le transcript) suivi d'un message « system » d'environnement (répertoire, plateforme, modèle, date du jour) qui porte le point de cache. Le transcript changeant à chaque étape, seuls outils + système peuvent être lus en cache ; tout le transcript est **réécrit en cache** (`cache_write`) à chaque étape, et le processus est relancé à chaque étape (une à deux secondes). `scripts/bench_claude_code.py` mesure, pour une conversation de 15 étapes avec outils (un ou plusieurs tours), la latence totale et du premier token, les tokens d'entrée, `cache_read`, `cache_write` et de sortie, en mode transcript et persistant ; sans `--run` il affiche le plan et sort (code 2), `--fake` déroule le même scénario avec un faux modèle. |
| D128 | **Retenu, désactivé par défaut : processus persistant dans un tour.** `ClaudeCodeClient(persistent=True)` garde un `ClaudeSDKClient` par `turn_id` (D92) entre les étapes d'un même tour. Après un appel différé, l'étape suivante n'envoie au processus qu'**un** message utilisateur fait des blocs `tool_result` natifs (`is_error` conservé, images chargées en base64), suivis des messages utilisateur de fin de requête. La CLI continue alors son historique natif (`tool_use` / `tool_result`, raisonnement signé), et la requête N+1 prolonge la requête N : le préfixe peut être lu en cache. Repli sur le transcript (processus neuf, l'ancien fermé) dès que la requête n'est pas la suite exacte : modèle, système, outils, réglages ou outils fournisseur différents, historique réécrit, appels différents de ceux rendus, message non utilisateur après les résultats. Fin de tour (réponse sans appel), erreur ou annulation : processus fermé. Au plus `max_live` (4) processus gardés (le plus ancien évincé), fermés après `live_ttl` (600 s) d'inactivité au prochain appel, tous par `aclose()`. Écart assumé : un message utilisateur de fin (état en queue de contexte, D42) remplacé d'une étape à l'autre n'est pas retiré de l'historique natif ; le nouveau est ajouté après les résultats, si bien que le modèle voit aussi l'état précédent. D'où l'opt-in, jusqu'à ce que la mesure tranche. Sans `turn_id` (appels hors tour), comportement inchangé. |
| D129 | **Écarté : préfixe de transcript stable.** L'idée était de garder identique le début du transcript d'une étape à l'autre pour qu'il soit lu en cache. Vérifié contre le faux serveur : le point de cache est posé par la CLI sur son message d'environnement, **après** notre message ; aucun point n'est posé à l'intérieur ou avant le transcript,, et le message d'environnement qui le porte est ajouté par la CLI, hors de notre contrôle. Découper le transcript en blocs n'y change rien. Le cache ne lit donc jamais que outils + système, quel que soit l'ordre du transcript. |
| D130 | **Raisonnement : conservé dans un tour seulement par le mode persistant ; écarté entre les tours.** En mode persistant, les blocs `thinking` signés restent dans l'historique natif de la CLI (`context_management` « keep all » observé dans la requête) : le raisonnement est conservé entre les étapes d'un tour. Entre deux tours, il faudrait rejouer un historique fabriqué avec des blocs `thinking` signés, ce que Claude Code n'accepte pas (D106) ; le transcript continue d'omettre le raisonnement. Reprendre la session native de la CLI d'un tour à l'autre (`resume`) est écarté : son historique diverge de celui de bouture dès qu'une branche, une édition ou un autre client intervient. |
| D131 | **`Agents.shutdown` ferme le client de modèle.** Si le client a une méthode `aclose()` (facultative, détectée par `getattr` comme `estimate`), `shutdown` l'appelle après la clôture des tours et avant la fermeture du stockage ; une erreur y est ignorée. Utilisé par `ClaudeCodeClient` pour fermer les processus gardés (D128). |

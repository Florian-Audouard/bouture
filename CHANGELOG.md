# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ; versions
[SemVer](https://semver.org/lang/fr/). Les numéros Dxx renvoient à [DECISIONS.md](DECISIONS.md).

## [0.2.0] — 2026-10-20

### Ajouté

- **Stockage SQLite** (`bouture[sqlite]`, `bouture.storage.sqlite.SQLiteStorage`) pour les
  applications locales mono-processus, Windows et Linux : mode WAL, transactions
  `BEGIN IMMEDIATE`, migrations idempotentes utilisables depuis Alembic (D107–D111).
- **Images dans les résultats d'outils** : un outil renvoie `TextBlock` / `ImageBlock` ; seules
  des références sont stockées, `@agents.image_loader` fournit les octets à l'appel au modèle.
  `LiteLLMClient(tool_images=…)`, images base64 pour `ClaudeCodeClient` (D112–D115).
- **Outils fournisseur** : `AgentDefinition(provider_tools=["web_search", "web_fetch"])`,
  exécutés par le fournisseur, tracés dans l'arbre et les événements (D116–D120).
- **Outils longs** : `@d.tool(timeout=None)` et `AgentDefinition(tool_timeout=None)` = sans
  limite ; le bail reste renouvelé pendant un outil qui attend (D121–D122).
- **`bouture.hosting.BackgroundAgents`** : bouture sur un thread et une boucle à lui
  (`Proactor` sous Windows), ponts synchrones et async, vue `bg.aio`, `make_router(bg, …)`
  (D123–D125).
- **`ClaudeCodeClient(persistent=True)`** (opt-in) : processus Claude Code gardé entre les étapes
  d'un tour, résultats d'outils envoyés en blocs natifs (D128, D130) ;
  `scripts/bench_claude_code.py` mesure latence et cache (D127).
- `Agents.shutdown()` appelle `model_client.aclose()` s'il existe (D131).
- Exemples : `examples/studio.py` (SQLite + `BackgroundAgents` + `ClaudeCodeClient` + outil
  image) ; `examples/character_world` tourne par défaut sur un fichier SQLite.

### Modifié

- `@d.tool(timeout=None)` signifie désormais « sans limite » ; omettre `timeout` hérite du délai
  de la définition. Un délai nul ou négatif lève `ValueError` (D121, remplace D65).
- Les appels d'outils d'un message assistant sont marqués ouverts dès son enregistrement : un
  arrêt pendant `message_end` ne laisse plus d'appel sans résultat (D126).
- `examples/character_world` : `DATABASE_URL` absent → SQLite (`.data/character_world.db`) au
  lieu de la mémoire ; `DATABASE_URL=memory` pour l'ancien comportement.

## [0.1.0]

Première version : sessions versionnées (arbre, fork, rollback), tours avec outils et pause,
publication et inputs, stockage Postgres et mémoire, `LiteLLMClient`, `ClaudeCodeClient`,
routes FastAPI (D1–D106).

[0.2.0]: https://github.com/Florian-Audouard/bouture/releases/tag/v0.2.0

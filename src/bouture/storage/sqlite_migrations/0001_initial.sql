-- Schéma SQLite de bouture (D107). Pas de schéma nommé : tables préfixées ``bouture_``.
-- uuid : texte canonique ; dates : texte UTC à largeur fixe (triable) ; JSON : texte.
-- ``IF NOT EXISTS`` partout : rejouable si un pilote applique le DDL hors transaction.
CREATE TABLE IF NOT EXISTS bouture_sessions (
  id                     text PRIMARY KEY,
  kind                   text NOT NULL,
  owner                  text NOT NULL,
  title                  text,
  metadata               text NOT NULL DEFAULT '{}',
  model_override         text,
  model_settings         text NOT NULL DEFAULT '{}',
  leaf_entry_id          text,
  object_id              text UNIQUE,
  cloned_from_session_id text REFERENCES bouture_sessions(id) ON DELETE SET NULL,
  cloned_from_entry_id   text,
  origin_session_id      text REFERENCES bouture_sessions(id) ON DELETE SET NULL,
  origin_entry_id        text,
  origin_handler         text,
  origin_payload         text,
  pending                text NOT NULL DEFAULT '[]',
  running_turn_id        text,
  lock_expires_at        text,
  cancel_requested       integer NOT NULL DEFAULT 0,
  archived_at            text,
  created_at             text NOT NULL,
  updated_at             text NOT NULL
);

CREATE INDEX IF NOT EXISTS bouture_sessions_owner_idx
  ON bouture_sessions (owner, kind, updated_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS bouture_sessions_origin_idx ON bouture_sessions (origin_session_id);

CREATE TABLE IF NOT EXISTS bouture_session_inputs (
  session_id   text NOT NULL REFERENCES bouture_sessions(id) ON DELETE CASCADE,
  name         text NOT NULL,
  mode         text NOT NULL CHECK (mode IN ('ref', 'data')),
  kind         text NOT NULL,
  object_id    text,
  version      integer,
  data         text,
  data_version integer,
  PRIMARY KEY (session_id, name)
);

CREATE TABLE IF NOT EXISTS bouture_states (
  id            text PRIMARY KEY,
  kind          text NOT NULL,
  state_version integer NOT NULL,
  data          text NOT NULL,
  created_at    text NOT NULL
);

CREATE TABLE IF NOT EXISTS bouture_entries (
  id            text PRIMARY KEY,
  session_id    text NOT NULL REFERENCES bouture_sessions(id) ON DELETE CASCADE,
  parent_id     text REFERENCES bouture_entries(id) ON DELETE CASCADE,
  type          text NOT NULL,
  content       text NOT NULL,
  state_id      text REFERENCES bouture_states(id),
  is_checkpoint integer NOT NULL,
  interrupted   integer NOT NULL DEFAULT 0,
  turn_id       text,
  usage         text,
  created_at    text NOT NULL,
  label         text,
  hidden_at     text,
  metadata      text NOT NULL DEFAULT '{}',
  derived_from  text
);

CREATE INDEX IF NOT EXISTS bouture_entries_session_idx ON bouture_entries (session_id, created_at, id);

CREATE INDEX IF NOT EXISTS bouture_entries_parent_idx ON bouture_entries (parent_id);

CREATE INDEX IF NOT EXISTS bouture_entries_state_idx ON bouture_entries (state_id);

CREATE TABLE IF NOT EXISTS bouture_turns (
  id              text PRIMARY KEY,
  session_id      text NOT NULL REFERENCES bouture_sessions(id) ON DELETE CASCADE,
  trigger         text NOT NULL,
  status          text NOT NULL,
  parent_entry_id text,
  usage           text,
  error           text,
  started_at      text NOT NULL,
  ended_at        text
);

CREATE INDEX IF NOT EXISTS bouture_turns_session_idx ON bouture_turns (session_id, started_at DESC);

-- AUTOINCREMENT : un id d'événement n'est jamais réutilisé après une purge (Last-Event-ID).
CREATE TABLE IF NOT EXISTS bouture_turn_events (
  id          integer PRIMARY KEY AUTOINCREMENT,
  session_id  text NOT NULL,
  turn_id     text NOT NULL,
  type        text NOT NULL,
  data        text NOT NULL,
  created_at  text NOT NULL
);

CREATE INDEX IF NOT EXISTS bouture_turn_events_session_idx ON bouture_turn_events (session_id, id);

CREATE INDEX IF NOT EXISTS bouture_turn_events_turn_idx ON bouture_turn_events (turn_id, id);

CREATE TABLE IF NOT EXISTS bouture_publications (
  kind         text NOT NULL,
  object_id    text NOT NULL,
  version      integer NOT NULL,
  session_id   text NOT NULL REFERENCES bouture_sessions(id) ON DELETE CASCADE,
  entry_id     text NOT NULL,
  state_id     text NOT NULL REFERENCES bouture_states(id),
  owner        text NOT NULL,
  published_at text NOT NULL,
  PRIMARY KEY (object_id, version)
);

CREATE INDEX IF NOT EXISTS bouture_publications_kind_idx ON bouture_publications (kind, owner);

CREATE INDEX IF NOT EXISTS bouture_publications_state_idx ON bouture_publications (state_id);

CREATE TABLE IF NOT EXISTS bouture_pending_child_results (
  id                text PRIMARY KEY,
  parent_session_id text NOT NULL REFERENCES bouture_sessions(id) ON DELETE CASCADE,
  child_session_id  text NOT NULL REFERENCES bouture_sessions(id) ON DELETE CASCADE,
  object_id         text NOT NULL,
  version           integer NOT NULL,
  created_at        text NOT NULL,
  applied_at        text,
  applied_entry_id  text
);

CREATE INDEX IF NOT EXISTS bouture_pending_child_results_idx
  ON bouture_pending_child_results (parent_session_id) WHERE applied_at IS NULL

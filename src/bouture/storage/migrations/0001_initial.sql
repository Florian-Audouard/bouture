CREATE TABLE bouture.sessions (
  id                     uuid PRIMARY KEY,
  kind                   text NOT NULL,
  owner                  text NOT NULL,
  title                  text,
  metadata               jsonb NOT NULL DEFAULT '{}',
  model_override         text,
  leaf_entry_id          uuid,
  object_id              uuid UNIQUE,
  cloned_from_session_id uuid REFERENCES bouture.sessions(id) ON DELETE SET NULL,
  cloned_from_entry_id   uuid,
  origin_session_id      uuid REFERENCES bouture.sessions(id) ON DELETE SET NULL,
  origin_entry_id        uuid,
  origin_handler         text,
  origin_payload         jsonb,
  pending                jsonb NOT NULL DEFAULT '[]',
  running_turn_id        uuid,
  lock_expires_at        timestamptz,
  cancel_requested       boolean NOT NULL DEFAULT false,
  archived_at            timestamptz,
  created_at             timestamptz NOT NULL,
  updated_at             timestamptz NOT NULL
);

CREATE INDEX sessions_owner_idx ON bouture.sessions (owner, kind, updated_at DESC, id DESC);

CREATE INDEX sessions_origin_idx ON bouture.sessions (origin_session_id);

CREATE TABLE bouture.session_inputs (
  session_id   uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  name         text NOT NULL,
  mode         text NOT NULL CHECK (mode IN ('ref', 'data')),
  kind         text NOT NULL,
  object_id    uuid,
  version      int,
  data         jsonb,
  data_version int,
  PRIMARY KEY (session_id, name)
);

CREATE TABLE bouture.states (
  id            uuid PRIMARY KEY,
  kind          text NOT NULL,
  state_version int NOT NULL,
  data          jsonb NOT NULL,
  created_at    timestamptz NOT NULL
);

CREATE TABLE bouture.entries (
  id            uuid PRIMARY KEY,
  session_id    uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  parent_id     uuid REFERENCES bouture.entries(id) ON DELETE CASCADE,
  type          text NOT NULL,
  content       jsonb NOT NULL,
  state_id      uuid REFERENCES bouture.states(id),
  is_checkpoint boolean NOT NULL,
  interrupted   boolean NOT NULL DEFAULT false,
  turn_id       uuid,
  usage         jsonb,
  created_at    timestamptz NOT NULL
);

CREATE INDEX entries_session_idx ON bouture.entries (session_id, id);

CREATE TABLE bouture.turns (
  id              uuid PRIMARY KEY,
  session_id      uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  trigger         text NOT NULL,
  status          text NOT NULL,
  parent_entry_id uuid,
  usage           jsonb,
  error           jsonb,
  started_at      timestamptz NOT NULL,
  ended_at        timestamptz
);

CREATE INDEX turns_session_idx ON bouture.turns (session_id, started_at DESC);

CREATE TABLE bouture.turn_events (
  id          bigserial PRIMARY KEY,
  session_id  uuid NOT NULL,
  turn_id     uuid NOT NULL,
  type        text NOT NULL,
  data        jsonb NOT NULL,
  created_at  timestamptz NOT NULL
);

CREATE INDEX turn_events_session_idx ON bouture.turn_events (session_id, id);

CREATE INDEX turn_events_turn_idx ON bouture.turn_events (turn_id, id);

CREATE TABLE bouture.publications (
  kind         text NOT NULL,
  object_id    uuid NOT NULL,
  version      int NOT NULL,
  session_id   uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  entry_id     uuid NOT NULL,
  state_id     uuid NOT NULL REFERENCES bouture.states(id),
  owner        text NOT NULL,
  published_at timestamptz NOT NULL,
  PRIMARY KEY (object_id, version)
);

CREATE INDEX publications_kind_idx ON bouture.publications (kind, owner);

CREATE TABLE bouture.pending_child_results (
  id                uuid PRIMARY KEY,
  parent_session_id uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  child_session_id  uuid NOT NULL REFERENCES bouture.sessions(id) ON DELETE CASCADE,
  object_id         uuid NOT NULL,
  version           int NOT NULL,
  created_at        timestamptz NOT NULL,
  applied_at        timestamptz,
  applied_entry_id  uuid
);

CREATE INDEX pending_child_results_idx ON bouture.pending_child_results (parent_session_id) WHERE applied_at IS NULL

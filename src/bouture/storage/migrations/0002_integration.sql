-- D92 : réglages d'échantillonnage surchargés par session.
ALTER TABLE bouture.sessions ADD COLUMN model_settings jsonb NOT NULL DEFAULT '{}';

-- D102 : recherche de sessions par métadonnées (containment @>).
CREATE INDEX sessions_metadata_idx ON bouture.sessions USING gin (metadata jsonb_path_ops);

-- D95 : point de sauvegarde nommé.
ALTER TABLE bouture.entries ADD COLUMN label text;

-- D100 : branche mise à la corbeille.
ALTER TABLE bouture.entries ADD COLUMN hidden_at timestamptz;

-- D101 : données applicatives attachées à une entrée.
ALTER TABLE bouture.entries ADD COLUMN metadata jsonb NOT NULL DEFAULT '{}';

-- D97 : entrée dérivée d'une autre (message édité).
ALTER TABLE bouture.entries ADD COLUMN derived_from uuid;

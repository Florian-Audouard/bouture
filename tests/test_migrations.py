"""Migrations Postgres pilotables par l'application (D103)."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from bouture import SchemaOutdated

from .conftest import PG_URL

pytestmark = pytest.mark.skipif(not PG_URL, reason="BOUTURE_TEST_PG_URL non défini")


async def test_check_mode_and_idempotent_migrations() -> None:
    from bouture.storage.postgres import (
        SCHEMA_VERSION,
        PostgresStorage,
        apply_migrations,
        schema_version,
    )

    assert PG_URL
    engine = create_async_engine(PG_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS bouture CASCADE"))
        st = PostgresStorage(engine)
        # Vérification seule : refuse un schéma absent, sans rien créer.
        with pytest.raises(SchemaOutdated):
            await st.setup(auto_migrate=False)
        async with engine.connect() as conn:
            assert await conn.run_sync(schema_version) == 0

        # Comme dans une migration Alembic : connexion synchrone, transaction ouverte.
        async with engine.begin() as conn:
            assert await conn.run_sync(lambda c: apply_migrations(c, target=1)) == 1
        with pytest.raises(SchemaOutdated):
            await st.setup(auto_migrate=False)
        async with engine.begin() as conn:
            assert await conn.run_sync(apply_migrations) == SCHEMA_VERSION
            assert await conn.run_sync(apply_migrations) == SCHEMA_VERSION
        await st.setup(auto_migrate=False)
    finally:
        await engine.dispose()

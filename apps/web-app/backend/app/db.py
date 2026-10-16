"""Async SQLAlchemy engine, session, and Base. Tables are created on startup (create_all).

A production hardening step is to switch to Alembic migrations (see design/05-web-app.md); for the
demo, create_all keeps setup to zero.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# Columns added to existing tables after they first shipped: (table, column, type + default).
# create_all() only creates missing tables, so init_db() adds these to an existing database.
_ADDED_COLUMNS = (
    ("studies", "pipeline", "VARCHAR DEFAULT 'pipeline'"),
    ("studies", "archived", "BOOLEAN DEFAULT FALSE"),
    ("studies", "description", "VARCHAR DEFAULT ''"),
    ("studies", "concurrency", "INTEGER DEFAULT 8"),
    ("rubrics", "preset", "VARCHAR DEFAULT 'reference_qa'"),
    ("rubrics", "system_prompt", "VARCHAR"),
    # dimensions: several judged rubrics per study, one StudyResult row each
    ("studies", "rubric_ids", "JSON DEFAULT '[]'"),
    ("study_results", "dimension", "VARCHAR DEFAULT 'quality'"),
)


def _migrate_sqlite(conn) -> None:
    """The same additive migration for SQLite, which has no ``ADD COLUMN IF NOT EXISTS`` and
    can't drop a constraint: check the schema first, and rebuild ``study_results`` if it still
    has the pre-dimensions ``UNIQUE (study_id)``."""
    from sqlalchemy import inspect, text

    insp = inspect(conn)
    for table, column, ddl in _ADDED_COLUMNS:
        if column not in {c["name"] for c in insp.get_columns(table)}:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
    if any(u["column_names"] == ["study_id"] for u in insp.get_unique_constraints("study_results")):
        # A DML statement first makes the driver open a transaction, so the rebuild below is
        # atomic (pysqlite would otherwise run each DDL statement in autocommit mode).
        conn.execute(text("DELETE FROM study_results WHERE 0"))
        conn.execute(text("ALTER TABLE study_results RENAME TO _study_results_old"))
        Base.metadata.tables["study_results"].create(conn)
        cols = "id, study_id, dimension, payload, created_at"
        conn.execute(
            text(f"INSERT INTO study_results ({cols}) SELECT {cols} FROM _study_results_old"))
        conn.execute(text("DROP TABLE _study_results_old"))


async def init_db() -> None:
    """Create tables if they don't exist (called on app startup)."""
    import app.models  # noqa: F401 — register the models on Base's metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Lightweight additive migration for existing DBs (create_all doesn't ALTER). Keeps prior
        # data (e.g. the eval study).
        if DATABASE_URL.startswith("postgresql"):
            from sqlalchemy import text
            for table, column, ddl in _ADDED_COLUMNS:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl}"))
            await conn.execute(text(
                "ALTER TABLE study_results DROP CONSTRAINT IF EXISTS study_results_study_id_key"))
        elif DATABASE_URL.startswith("sqlite"):
            await conn.run_sync(_migrate_sqlite)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: a session per request."""
    async with SessionLocal() as session:
        yield session

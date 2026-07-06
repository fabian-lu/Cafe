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


async def init_db() -> None:
    """Create tables if they don't exist (called on app startup)."""
    import app.models  # noqa: F401 — register the models on Base's metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Lightweight additive migration for existing DBs (create_all doesn't ALTER). Postgres only;
        # SQLite dev DBs are created fresh with the column. Keeps prior data (e.g. the eval study).
        if DATABASE_URL.startswith("postgresql"):
            from sqlalchemy import text
            for stmt in (
                "ALTER TABLE studies ADD COLUMN IF NOT EXISTS pipeline VARCHAR DEFAULT 'pipeline'",
                "ALTER TABLE studies ADD COLUMN IF NOT EXISTS archived BOOLEAN DEFAULT FALSE",
                "ALTER TABLE studies ADD COLUMN IF NOT EXISTS description VARCHAR DEFAULT ''",
                "ALTER TABLE studies ADD COLUMN IF NOT EXISTS concurrency INTEGER DEFAULT 8",
                "ALTER TABLE rubrics ADD COLUMN IF NOT EXISTS preset VARCHAR DEFAULT 'reference_qa'",
                "ALTER TABLE rubrics ADD COLUMN IF NOT EXISTS system_prompt VARCHAR",
            ):
                await conn.execute(text(stmt))


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: a session per request."""
    async with SessionLocal() as session:
        yield session

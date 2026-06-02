"""
app/db/database.py
==================
Async SQLAlchemy database setup using SQLite (aiosqlite driver).

SQLite is chosen over PostgreSQL for this challenge because:
  - Single-node deployment, no separate database container needed
  - Adequate for the event volumes in this challenge (millions of rows)
  - WAL mode gives concurrent read/write performance
  - Trivially switched to PostgreSQL by changing DATABASE_URL

See CHOICES.md §3 for full rationale.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./data/store_intelligence.db",
)


class Base(DeclarativeBase):
    pass


# Create engine with WAL mode for better concurrent performance
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def init_db() -> None:
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Enable WAL mode for SQLite
        if "sqlite" in DATABASE_URL:
            await conn.execute(__import__("sqlalchemy").text("PRAGMA journal_mode=WAL"))
            await conn.execute(__import__("sqlalchemy").text("PRAGMA synchronous=NORMAL"))


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
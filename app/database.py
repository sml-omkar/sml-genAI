"""
Database Connection
Async SQLAlchemy engine + session factory for PostgreSQL.
"""

from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# --- Async Engine ---
# pool_size: persistent connections in the pool
# max_overflow: extra connections allowed beyond pool_size
# pool_pre_ping: verify connections before using (handles dead connections)
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# --- Session Factory ---
# Each request gets its own session; auto-closes after use
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """
    Base class for all ORM models.
    All models inherit from this to get common features.
    """
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    Yields a session, auto-closes after the request.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Create all tables defined by models.
    Called once at application startup.
    """
    # Import all models to ensure they're registered with SQLAlchemy
    from app.models.user import User
    from app.models.folder import Folder
    from app.models.document import Document
    from app.models.document_memory import DocumentMemory
    from app.models.conversation import Conversation, Message
    from app.models.group import Group, UserGroup, GroupFolder
    from app.models.department import Department
    from app.models.feedback import Feedback
    from app.models.external_api import ExternalApi, ApiFolder
    from app.models.broadcast import TeamsProactiveRef, Broadcast, BroadcastRecipient

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # --- Migrations: each in its own transaction so one failure doesn't abort others ---
    # Using separate engine.begin() per statement prevents InFailedSQLTransactionError
    _migrations = [
        "ALTER TYPE processingstatus ADD VALUE IF NOT EXISTS 'UNDERSTANDING'",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS tokens_in INTEGER DEFAULT 0",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS tokens_out INTEGER DEFAULT 0",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS model_used VARCHAR(100)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS chat_access_enabled BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_token_limit INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS source VARCHAR(20)",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS teams_aad_id VARCHAR(255)",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS teams_email VARCHAR(255)",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS teams_name VARCHAR(255)",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS teams_channel_id VARCHAR(500)",
        # Must add external_api_id BEFORE creating its index
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS external_api_id UUID",
    ]
    for sql in _migrations:
        try:
            async with engine.begin() as c:
                await c.execute(text(sql))
        except Exception as e:
            # IF NOT EXISTS makes most idempotent, but log anyway for visibility
            print(f"[DB] Migration note (likely already applied): {sql[:70]} -> {e}")

    _indexes = [
        "CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source)",
        "CREATE INDEX IF NOT EXISTS idx_conversations_teams_aad ON conversations(teams_aad_id)",
        "CREATE INDEX IF NOT EXISTS idx_conversations_external_api ON conversations(external_api_id)",
    ]
    for sql in _indexes:
        try:
            async with engine.begin() as c:
                await c.execute(text(sql))
        except Exception as e:
            print(f"[DB] Index note: {sql[:70]} -> {e}")

    # Seed default departments if none exist
    from sqlalchemy import select, func
    async with AsyncSessionLocal() as session:
        count = (await session.execute(select(func.count(Department.id)))).scalar() or 0
        if count == 0:
            defaults = [
                Department(name="HR", slug="hr", description="Human Resources"),
                Department(name="IT", slug="it", description="Information Technology"),
                Department(name="Finance", slug="finance", description="Finance & Accounting"),
            ]
            session.add_all(defaults)
            await session.commit()
            print(f"[DB] Seeded {len(defaults)} default departments")


async def close_db():
    """Dispose the engine connection pool on shutdown."""
    await engine.dispose()

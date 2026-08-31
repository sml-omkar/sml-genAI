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

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migration: add 'understanding' to the processingstatus enum
        # (enum labels are stored UPPERCASE; no-op if already present)
        await conn.execute(text(
            "ALTER TYPE processingstatus ADD VALUE IF NOT EXISTS 'UNDERSTANDING'"
        ))

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

"""
Memory Service
Manages conversation history: store, retrieve, cleanup.
Conversations expire after 24 hours.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from uuid import UUID

from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.conversation import Conversation, Message, MessageRole


class MemoryService:
    """Async service for conversation memory management."""

    def __init__(self, ttl_hours: int = 24, max_messages: int = 10):
        self.ttl_hours = ttl_hours
        self.max_messages = max_messages

    async def get_or_create_conversation(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        async with AsyncSessionLocal() as db:
            if conversation_id:
                result = await db.execute(
                    select(Conversation).where(Conversation.id == UUID(conversation_id))
                )
                conv = result.scalar_one_or_none()
                if conv:
                    # Check if expired
                    if self._is_expired(conv):
                        await self._delete_conversation(db, conv)
                        await db.commit()
                        return await self._create_conversation(db, user_id)
                    return conv

            return await self._create_conversation(db, user_id)

    async def _create_conversation(self, db: AsyncSession, user_id: Optional[str] = None) -> Conversation:
        conv = Conversation(
            user_id=UUID(user_id) if user_id else None,
            title=None,
        )
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return conv

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
    ) -> Message:
        """Add a message to a conversation."""
        async with AsyncSessionLocal() as db:
            msg = Message(
                conversation_id=UUID(conversation_id),
                role=MessageRole(role),
                content=content,
                sources=json.dumps(sources) if sources else None,
            )
            db.add(msg)

            # Update conversation timestamp and title
            result = await db.execute(
                select(Conversation).where(Conversation.id == UUID(conversation_id))
            )
            conv = result.scalar_one_or_none()
            if conv:
                conv.updated_at = datetime.utcnow()
                # Auto-generate title from first user message
                if not conv.title and role == "user":
                    conv.title = content[:100]

            await db.commit()
            await db.refresh(msg)
            return msg

    async def get_history(self, conversation_id: str) -> List[Dict]:
        """
        Get recent messages for a conversation.
        Only returns messages from conversations within TTL.
        Returns最多 max_messages most recent messages.
        """
        async with AsyncSessionLocal() as db:
            # Check conversation exists and not expired
            result = await db.execute(
                select(Conversation).where(Conversation.id == UUID(conversation_id))
            )
            conv = result.scalar_one_or_none()
            if not conv or self._is_expired(conv):
                return []

            # Get recent messages
            cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)
            result = await db.execute(
                select(Message)
                .where(
                    and_(
                        Message.conversation_id == UUID(conversation_id),
                        Message.created_at >= cutoff,
                    )
                )
                .order_by(Message.created_at.desc())
                .limit(self.max_messages)
            )
            messages = result.scalars().all()

            # Reverse to chronological order
            messages.reverse()

            return [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "sources": json.loads(msg.sources) if msg.sources else None,
                }
                for msg in messages
            ]

    async def cleanup_expired(self) -> int:
        """Delete conversations older than TTL. Returns count deleted."""
        async with AsyncSessionLocal() as db:
            cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)

            # Find expired conversations (inactive longer than TTL)
            result = await db.execute(
                select(Conversation).where(Conversation.updated_at < cutoff)
            )
            expired = result.scalars().all()

            count = 0
            for conv in expired:
                await self._delete_conversation(db, conv)
                count += 1

            await db.commit()
            return count

    async def _delete_conversation(self, db: AsyncSession, conv: Conversation):
        """Delete a conversation and all its messages."""
        # Messages are cascade-deleted
        await db.delete(conv)

    def _is_expired(self, conv: Conversation) -> bool:
        """Check if conversation has been inactive longer than TTL."""
        last_activity = conv.updated_at or conv.created_at
        age = datetime.utcnow() - last_activity
        return age > timedelta(hours=self.ttl_hours)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Manually delete a conversation."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Conversation).where(Conversation.id == UUID(conversation_id))
            )
            conv = result.scalar_one_or_none()
            if conv:
                await self._delete_conversation(db, conv)
                await db.commit()
                return True
            return False


# Singleton instance
_memory_service: Optional[MemoryService] = None


def get_memory_service(ttl_hours: int = 24, max_messages: int = 10) -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService(ttl_hours=ttl_hours, max_messages=max_messages)
    else:
        _memory_service.ttl_hours = ttl_hours
        _memory_service.max_messages = max_messages
    return _memory_service

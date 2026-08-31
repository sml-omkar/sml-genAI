"""
Per-User Chatbot Access & Daily Token Limits
Shared helpers used by both the public /api/chat endpoint and the Teams bot to
enforce admin-managed access control before forwarding a request to the LLM.

A request is allowed only when the user:
  1. exists and is_active=True (registration gating), AND
  2. has chat_access_enabled=True (admin enable/disable switch), AND
  3. is not already at/over their daily_token_limit (0 = unlimited).
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.conversation import Message, MessageRole, Conversation


async def get_user_daily_usage(db: AsyncSession, user_id) -> int:
    """
    Total tokens (prompt + completion) the given user consumed today,
    across all their conversations. 0 if none.
    """
    start_of_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(
            func.coalesce(func.sum(Message.tokens_in), 0)
            + func.coalesce(func.sum(Message.tokens_out), 0)
        )
        .select_from(Conversation)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(
            Conversation.user_id == user_id,
            Message.role == MessageRole.ASSISTANT.name,
            Message.created_at >= start_of_day,
        )
    )
    return int(result.scalar() or 0)


async def check_user_can_use_chat(
    db: AsyncSession, user: User
) -> Tuple[bool, Optional[str]]:
    """
    Return (allowed, reason). When not allowed, reason is a user-facing message.
    This is the enforcement check called before the LLM.
    """
    if not getattr(user, "is_active", True):
        return False, "Your account has been deactivated. Contact your administrator."
    if not getattr(user, "chat_access_enabled", True):
        return (
            False,
            "Your chatbot access has been disabled. Contact your administrator to re-enable it.",
        )

    limit = getattr(user, "daily_token_limit", 0) or 0
    if limit > 0:
        used = await get_user_daily_usage(db, user.id)
        if used >= limit:
            return (
                False,
                f"You have reached your daily token limit of {limit:,} tokens. "
                "Your usage will reset tomorrow — please contact your administrator to adjust the limit.",
            )
    return True, None

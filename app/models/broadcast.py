"""
Broadcast & Teams Proactive Reference Models
- TeamsProactiveRef: stores ConversationReference per Teams user/channel so bot can send
  proactive messages without user initiating (adapter.continue_conversation).
- Broadcast + BroadcastRecipient: tracks admin-triggered broadcast batches.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, Integer, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.database import Base


class TeamsProactiveRef(Base):
    """
    One row per Teams conversation that bot has seen.
    conversation_reference is the Bot Framework ConversationReference JSON (serviceUrl, conversation, channelId, etc.)
    """
    __tablename__ = "teams_proactive_refs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Who — Teams identity
    aad_object_id = Column(String(255), nullable=True, index=True)
    teams_user_id = Column(String(255), nullable=False, index=True)  # Activity.from.id  e.g. 29:...
    user_name = Column(String(255), nullable=True)
    user_email = Column(String(255), nullable=True)

    # Where — Teams conversation
    conversation_id = Column(String(500), nullable=False, index=True)  # Activity.conversation.id
    service_url = Column(Text, nullable=False)
    channel_id = Column(String(50), nullable=False, default="msteams")
    tenant_id = Column(String(255), nullable=True, index=True)
    team_id = Column(String(255), nullable=True)
    channel_name = Column(String(255), nullable=True)
    conversation_type = Column(String(50), nullable=True)  # personal, channel, groupChat

    # Full Bot Framework ConversationReference JSON (for continue_conversation)
    reference_json = Column(JSONB, nullable=False)

    is_active = Column(Boolean, default=True, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_teams_refs_aad", "aad_object_id"),
        Index("idx_teams_refs_conversation", "conversation_id"),
        Index("idx_teams_refs_tenant", "tenant_id"),
    )


class Broadcast(Base):
    """Admin-triggered broadcast batch."""
    __tablename__ = "broadcasts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    message = Column(Text, nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), nullable=False, default="queued")  # queued, running, done, failed
    total_recipients = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    failed_count = Column(Integer, default=0, nullable=False)
    # Optional filter that was used
    filter_info = Column(JSONB, nullable=True)  # {tenant_id, conversation_type, etc.}
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    recipients = relationship("BroadcastRecipient", back_populates="broadcast", cascade="all, delete-orphan")


class BroadcastRecipient(Base):
    """Per-recipient send status for a broadcast."""
    __tablename__ = "broadcast_recipients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    broadcast_id = Column(UUID(as_uuid=True), ForeignKey("broadcasts.id", ondelete="CASCADE"), nullable=False, index=True)
    ref_id = Column(UUID(as_uuid=True), ForeignKey("teams_proactive_refs.id", ondelete="SET NULL"), nullable=True)
    # Snapshot for display even if ref deleted
    teams_user_id = Column(String(255), nullable=True)
    user_name = Column(String(255), nullable=True)
    conversation_id = Column(String(500), nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, sent, failed
    error = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)

    broadcast = relationship("Broadcast", back_populates="recipients")

    __table_args__ = (
        Index("idx_broadcast_recipients_broadcast", "broadcast_id"),
    )

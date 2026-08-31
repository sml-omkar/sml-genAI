"""
Feedback Model
Stores user feedback on AI answers for quality monitoring.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Feedback(Base):
    """User feedback on AI answers."""
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Which conversation/message this feedback is for
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False, index=True)
    
    # Who gave feedback
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    
    # Feedback type: 1 = thumbs up, -1 = thumbs down
    rating = Column(Integer, nullable=False)
    
    # Optional comment
    comment = Column(Text, nullable=True)
    
    # The question and answer at time of feedback
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    
    # Source documents used
    sources = Column(Text, nullable=True)  # JSON array of document names
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Feedback rating={self.rating} conversation={self.conversation_id}>"

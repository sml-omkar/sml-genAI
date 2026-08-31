"""
Document Memory Model
Stores the LLM-generated "understanding" of each uploaded policy document.

Built during ingestion: after a document is chunked + embedded, the LLM
reads every chunk and distills the policy into:
- summary:   what the document covers
- key_facts: extracted rules/facts (numbers, deadlines, penalties, steps)

This memory is injected into the prompt at query time so answers are more
accurate, and key facts are also embedded into ChromaDB as retrievable notes.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text, String
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.database import Base


class DocumentMemory(Base):
    """LLM-distilled understanding of one document."""
    __tablename__ = "document_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=False,
        index=True,
        unique=True,
    )

    summary = Column(Text, nullable=True)
    key_facts = Column(JSONB, nullable=False, default=list)
    fact_count = Column(Integer, nullable=False, default=0)

    # Which LLM model produced this memory (for auditing/rebuilds)
    model_used = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", backref="memory")

    def __repr__(self):
        return f"<DocumentMemory doc={self.document_id} facts={self.fact_count}>"

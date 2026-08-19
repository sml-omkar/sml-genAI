"""
Document Model
Tracks uploaded PDFs and their RAG processing status.
"""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Integer, Enum, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class ProcessingStatus(str, enum.Enum):
    """RAG pipeline processing stages."""
    UPLOADING = "uploading"       # File received, saving to disk
    EXTRACTING = "extracting"     # Extracting text from PDF
    CHUNKING = "chunking"         # Splitting text into chunks
    EMBEDDING = "embedding"       # Generating vector embeddings
    STORING = "storing"           # Storing embeddings in ChromaDB
    READY = "ready"               # All done, ready for queries
    FAILED = "failed"             # Something went wrong


class Document(Base):
    """
    Document model.
    Represents a single uploaded PDF within a folder.
    Tracks the full RAG processing pipeline status.
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    filename = Column(String(500), nullable=False)          # Original filename
    stored_path = Column(String(1000), nullable=False)      # Path on disk
    file_size = Column(BigInteger, nullable=False, default=0)  # Size in bytes

    # Folder this document belongs to
    folder_id = Column(UUID(as_uuid=True), ForeignKey("folders.id"), nullable=False, index=True)

    # Who uploaded it
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Processing status — tracks where in the RAG pipeline this doc is
    status = Column(
        Enum(ProcessingStatus),
        nullable=False,
        default=ProcessingStatus.UPLOADING,
    )

    # Processing results
    total_pages = Column(Integer, nullable=True)        # Pages in the PDF
    chunk_count = Column(Integer, nullable=True, default=0)  # Chunks created
    error_message = Column(String(2000), nullable=True)  # If status=failed

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)  # When processing completed

    # Relationships
    folder = relationship("Folder", back_populates="documents")
    uploaded_by_user = relationship("User", back_populates="documents")

    def __repr__(self):
        return f"<Document {self.filename} status={self.status.value}>"

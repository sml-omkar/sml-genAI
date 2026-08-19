"""
Folder Model
Folders organize documents by department/category.
Admin creates folders, then uploads PDFs into them.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Folder(Base):
    """
    Folder model.
    Each folder belongs to a department and contains documents.
    Examples: "HR - Leave Policies", "IT - Security", "Finance - Expense Reports"
    """
    __tablename__ = "folders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    department = Column(String(50), nullable=False, index=True)  # "hr", "it", "finance"

    # Track document count for dashboard display
    document_count = Column(Integer, default=0, nullable=False)

    # Who created this folder
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship(
        "Document",
        back_populates="folder",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    group_access = relationship("GroupFolder", back_populates="folder", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Folder {self.name} dept={self.department}>"

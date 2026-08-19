"""
Department Model
Dynamic department management — replaces hardcoded hr/it/finance.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Department(Base):
    """
    Department table.
    Stores available departments. Users, folders, and documents reference
    departments by slug string (e.g. 'hr', 'it').
    """
    __tablename__ = "departments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Department {self.name} ({self.slug})>"

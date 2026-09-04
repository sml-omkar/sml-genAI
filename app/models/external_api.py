"""
External API Model — Folder-scoped chat endpoints
One API belongs to a single department but can attach many folders within that dept.
External apps call POST /api/external/{slug}/chat with X-API-Key and only get
answers from the attached folders' documents.
"""

import re
import secrets
import hashlib
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] or f"api-{uuid4().hex[:8]}"


def generate_api_key() -> tuple[str, str, str]:
    """
    Returns (plain_key, key_hash, key_prefix)
    plain_key is shown once to admin; key_hash stored in DB.
    """
    raw = secrets.token_urlsafe(32)  # ~43 chars
    plain = f"etho_{raw}"
    key_hash = hashlib.sha256(plain.encode()).hexdigest()
    prefix = plain[:12] + "..."
    return plain, key_hash, prefix


class ExternalApi(Base):
    """
    A named, key-protected endpoint scoped to a single department and a set of folders.
    Example: name="Leave Help API", department="hr", folders=[uuid1, uuid2]
    External app POSTs to /api/external/{slug}/chat with header X-API-Key.
    """
    __tablename__ = "external_apis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    department = Column(String(50), nullable=False, index=True)  # must match attached folders' department

    api_key_hash = Column(String(128), nullable=False)
    api_key_prefix = Column(String(32), nullable=True)  # for display: etho_abc...
    is_active = Column(Boolean, default=True, nullable=False)

    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Many-to-many via ApiFolder
    folders = relationship("ApiFolder", back_populates="api", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ExternalApi {self.name} dept={self.department} slug={self.slug}>"


class ApiFolder(Base):
    """Junction: ExternalApi can query a Folder (all folders must be in api.department)."""
    __tablename__ = "api_folders"
    __table_args__ = (
        UniqueConstraint("api_id", "folder_id", name="uq_api_folder"),
        Index("idx_api_folders_api", "api_id"),
        Index("idx_api_folders_folder", "folder_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    api_id = Column(UUID(as_uuid=True), ForeignKey("external_apis.id", ondelete="CASCADE"), nullable=False)
    folder_id = Column(UUID(as_uuid=True), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    api = relationship("ExternalApi", back_populates="folders")
    folder = relationship("Folder")

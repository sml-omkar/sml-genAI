"""
User and Role Models
Defines the RBAC system: departments, roles, and users.
"""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column, String, Boolean, DateTime, Enum, ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Department(str, enum.Enum):
    """Department types for RBAC filtering."""
    HR = "hr"
    IT = "it"
    FINANCE = "finance"


class RoleType(str, enum.Enum):
    """Role types with ascending privilege levels."""
    USER = "user"             # Can only query their department
    DEPT_ADMIN = "dept_admin" # Can upload/manage docs in their department
    SUPER_ADMIN = "super_admin"  # Full access to everything


class User(Base):
    """
    User model.
    Stores user credentials, role, and department assignment.
    Users are identified by email; roles control access.
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    department = Column(Enum(Department), nullable=False)
    role = Column(Enum(RoleType), nullable=False, default=RoleType.USER)
    is_active = Column(Boolean, default=True, nullable=False)

    # Teams integration — AAD Object ID for mapping Teams users to DB users
    aad_object_id = Column(String(255), unique=True, nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships — a user can upload many documents
    documents = relationship("Document", back_populates="uploaded_by_user")
    groups = relationship("UserGroup", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email} role={self.role.value} dept={self.department.value}>"

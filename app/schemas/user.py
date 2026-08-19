"""
User Schemas
Pydantic models for user CRUD request/response payloads.
"""

from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional
from datetime import datetime


class UserResponse(BaseModel):
    """User response (never includes password)."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    department: str
    role: str
    is_active: bool
    created_at: datetime

    @field_validator('id', mode='before')
    @classmethod
    def convert_id(cls, v):
        return str(v)


class UserListResponse(BaseModel):
    """List of users."""
    users: list[UserResponse]
    total: int


class UserCreateRequest(BaseModel):
    """Create a new user."""
    email: str
    password: str
    full_name: str
    department: str
    role: str = "user"


class UserUpdateRequest(BaseModel):
    """Update user fields."""
    full_name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class DashboardStats(BaseModel):
    """Admin dashboard statistics."""
    total_folders: int
    total_documents: int
    total_chunks: int
    total_users: int
    documents_by_status: dict
    documents_by_department: dict

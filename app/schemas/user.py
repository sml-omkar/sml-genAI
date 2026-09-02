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
    chat_access_enabled: bool = True
    daily_token_limit: int = 0
    aad_object_id: Optional[str] = None
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
    chat_access_enabled: bool = True
    daily_token_limit: int = 0
    aad_object_id: Optional[str] = None


class UserUpdateRequest(BaseModel):
    """Update user fields."""
    full_name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None
    chat_access_enabled: Optional[bool] = None
    daily_token_limit: Optional[int] = None
    aad_object_id: Optional[str] = None


class DashboardStats(BaseModel):
    """Admin dashboard statistics."""
    total_folders: int
    total_documents: int
    total_chunks: int
    total_users: int
    total_groups: int = 0
    documents_by_status: dict
    documents_by_department: dict

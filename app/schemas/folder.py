"""
Folder Schemas
Pydantic models for folder create/update/response payloads.
"""

from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional
from datetime import datetime


class FolderCreate(BaseModel):
    """Create a new folder."""
    name: str
    description: Optional[str] = None
    department: str


class FolderUpdate(BaseModel):
    """Update folder fields."""
    name: Optional[str] = None
    description: Optional[str] = None


class FolderResponse(BaseModel):
    """Folder response with metadata."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str]
    department: str
    document_count: int
    created_at: datetime

    @field_validator('id', mode='before')
    @classmethod
    def convert_id(cls, v):
        return str(v)


class FolderListResponse(BaseModel):
    """List of folders."""
    folders: list[FolderResponse]
    total: int

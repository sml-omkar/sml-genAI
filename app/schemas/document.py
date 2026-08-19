"""
Document Schemas
Pydantic models for document upload/response/status payloads.
"""

from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional
from datetime import datetime


class DocumentResponse(BaseModel):
    """Document response with processing status."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    folder_id: str
    file_size: int
    status: str
    total_pages: Optional[int]
    chunk_count: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]

    @field_validator('id', 'folder_id', mode='before')
    @classmethod
    def convert_uuids(cls, v):
        return str(v)


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: list[DocumentResponse]
    total: int


class ProcessingStatusResponse(BaseModel):
    """RAG pipeline processing status for a document."""
    document_id: str
    filename: str
    status: str
    total_pages: Optional[int]
    chunk_count: Optional[int]
    error_message: Optional[str]
    progress_percent: int


class RAGQueryRequest(BaseModel):
    """Test RAG query from admin portal."""
    question: str
    department: Optional[str] = None


class RAGQueryResponse(BaseModel):
    """RAG query result with sources."""
    answer: str
    sources: list[dict]
    chunks_retrieved: int

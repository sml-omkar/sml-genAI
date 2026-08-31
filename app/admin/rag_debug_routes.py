"""
RAG Debug Routes
View the agent's reasoning process step by step.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional

from app.rag.agent import query_rag
from app.models.user import User
from app.auth.dependencies import get_current_user

router = APIRouter()


class DebugRequest(BaseModel):
    question: str
    department: Optional[str] = None


@router.post("/debug")
async def debug_rag(
    request: DebugRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Debug endpoint: shows the full agent reasoning process.
    Returns step-by-step how the agent routes, searches, evaluates, and generates.
    """
    result = await query_rag(
        question=request.question,
        department=request.department,
        debug=True,
    )
    
    return {
        "question": request.question,
        "department": request.department or "all",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "chunks_retrieved": result.get("chunks_retrieved", 0),
        "debug": result.get("debug", {}),
    }

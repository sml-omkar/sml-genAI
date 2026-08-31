"""
Feedback Routes
Submit and retrieve user feedback on AI answers.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.models.feedback import Feedback
from app.models.user import User
from app.auth.dependencies import get_current_user

router = APIRouter()


class FeedbackRequest(BaseModel):
    """Submit feedback on an AI answer."""
    conversation_id: str
    message_id: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    comment: Optional[str] = None
    question: str
    answer: str
    sources: Optional[list] = None


class FeedbackStats(BaseModel):
    """Feedback statistics."""
    total_feedback: int
    positive_count: int
    negative_count: int
    positive_ratio: float
    recent_feedback: list


@router.post("/")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit feedback on an AI answer."""
    # Check if user already gave feedback for this message
    existing = await db.execute(
        select(Feedback).where(
            Feedback.message_id == request.message_id,
            Feedback.user_id == current_user.id,
        )
    )
    if existing.scalar_one_or_none():
        # Update existing feedback
        result = await db.execute(
            select(Feedback).where(
                Feedback.message_id == request.message_id,
                Feedback.user_id == current_user.id,
            )
        )
        feedback = result.scalar_one_or_none()
        feedback.rating = request.rating
        feedback.comment = request.comment
    else:
        # Create new feedback
        feedback = Feedback(
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            user_id=current_user.id,
            rating=request.rating,
            comment=request.comment,
            question=request.question,
            answer=request.answer,
            sources=json.dumps(request.sources) if request.sources else None,
        )
        db.add(feedback)
    
    await db.flush()
    await db.refresh(feedback)
    
    return {"message": "Feedback submitted", "id": str(feedback.id)}


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get feedback statistics (admin only)."""
    # Total feedback
    total = (await db.execute(select(func.count(Feedback.id)))).scalar() or 0
    
    # Positive/negative counts
    positive = (await db.execute(
        select(func.count(Feedback.id)).where(Feedback.rating == 1)
    )).scalar() or 0
    
    negative = (await db.execute(
        select(func.count(Feedback.id)).where(Feedback.rating == -1)
    )).scalar() or 0
    
    # Recent feedback (last 10)
    result = await db.execute(
        select(Feedback)
        .order_by(Feedback.created_at.desc())
        .limit(10)
    )
    recent = result.scalars().all()
    
    return FeedbackStats(
        total_feedback=total,
        positive_count=positive,
        negative_count=negative,
        positive_ratio=positive / total if total > 0 else 0,
        recent_feedback=[
            {
                "id": str(f.id),
                "rating": f.rating,
                "question": f.question[:100],
                "comment": f.comment,
                "created_at": f.created_at.isoformat(),
            }
            for f in recent
        ],
    )

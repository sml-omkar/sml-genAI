"""
Auth Routes
Login, forgot password, and token refresh endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    ForgotPasswordRequest,
)
from app.auth.jwt import (
    verify_password,
    create_access_token,
    hash_password,
)
from app.config import get_settings

router = APIRouter()
settings = get_settings()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Admin login endpoint.
    Validates email/password, returns JWT token with user info.
    """
    # Look up user by email
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    # Verify credentials
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user is active (same 401 to prevent user enumeration)
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Create JWT token
    token = create_access_token(
        user_id=str(user.id),
        email=user.email,
        role=user.role.value,
        department=user.department.value,
    )

    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "department": user.department.value,
        },
    )


@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Forgot password endpoint.
    In production, sends a reset link via email.
    For now, returns a generic message (don't reveal if email exists).
    """
    # Always return success to prevent email enumeration
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user:
        # TODO: Send email with reset token
        # For now, just log it
        print(f"[AUTH] Password reset requested for: {user.email}")

    return {
        "message": "If the email exists, a password reset link has been sent.",
    }

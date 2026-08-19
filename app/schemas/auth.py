"""
Auth Schemas
Pydantic models for authentication request/response payloads.
"""

from pydantic import BaseModel, EmailStr
from typing import Optional


class LoginRequest(BaseModel):
    """Admin login request."""
    email: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response after successful login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: dict  # user info: id, email, role, department, full_name


class UserCreate(BaseModel):
    """Create a new user."""
    email: str
    password: str
    full_name: str
    department: str  # "hr", "it", "finance"
    role: str = "user"  # "user", "dept_admin", "super_admin"


class UserUpdate(BaseModel):
    """Update user fields (all optional)."""
    full_name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    """Request password reset (sends email in future)."""
    email: str


class ResetPasswordRequest(BaseModel):
    """Reset password with token."""
    token: str
    new_password: str

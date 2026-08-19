"""
Auth Dependencies
FastAPI Depends for authentication and authorization.
Supports role-based and group-based access control.
"""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User, RoleType, Department
from app.auth.jwt import verify_token

# --- Bearer Token Extractor ---
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the current user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    return user


def require_role(*allowed_roles: RoleType):
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}",
            )
        return current_user
    return role_checker


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require super_admin or dept_admin."""
    if current_user.role not in (RoleType.SUPER_ADMIN, RoleType.DEPT_ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Admin role required.",
        )
    return current_user


async def require_super_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require super_admin only."""
    if current_user.role != RoleType.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Super admin only.",
        )
    return current_user


def require_department_access(target_department: str):
    async def dept_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role == RoleType.SUPER_ADMIN:
            return current_user
        if current_user.department.value != target_department:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. You can only access {current_user.department.value} department resources.",
            )
        return current_user
    return dept_checker


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Optional authentication — returns None if no token provided or user inactive."""
    if not credentials:
        return None
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    user_id = payload.get("sub")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user and not user.is_active:
        return None
    return user


async def get_current_user_from_cookie(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user from cookie-based JWT (for HTML pages).
    Returns None if not authenticated — pages should redirect to login.
    """
    token = request.cookies.get("access_token")
    if not token:
        return None
    payload = verify_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user and not user.is_active:
        return None
    return user


async def require_page_auth(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Require authentication for HTML page routes.
    Redirects to login if not authenticated.
    """
    user = await get_current_user_from_cookie(request, db)
    if not user:
        from fastapi.responses import RedirectResponse
        raise RedirectResponse(url="/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    return user

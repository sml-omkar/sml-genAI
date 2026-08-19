"""
Admin Routes
Dashboard stats and user management endpoints.
Cyprus admin (super_admin) sees everything.
Dept admins see only their department.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models.user import User, RoleType
from app.models.folder import Folder
from app.models.document import Document, ProcessingStatus
from app.models.group import Group, UserGroup, GroupFolder
from app.schemas.user import (
    UserResponse,
    UserListResponse,
    UserCreateRequest,
    UserUpdateRequest,
    DashboardStats,
)
from app.auth.dependencies import get_current_user, require_role, require_super_admin, require_admin
from app.auth.jwt import hash_password
from app.rbac.permissions import can_manage_users

router = APIRouter()


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get dashboard statistics.
    Super admins see all stats; dept admins see only their department.
    """
    is_super_admin = current_user.role == RoleType.SUPER_ADMIN
    dept_filter = None if is_super_admin else current_user.department.value

    folder_query = select(func.count(Folder.id))
    if dept_filter:
        folder_query = folder_query.where(Folder.department == dept_filter)
    total_folders = (await db.execute(folder_query)).scalar() or 0

    doc_query = select(func.count(Document.id))
    if dept_filter:
        doc_query = doc_query.join(Folder).where(Folder.department == dept_filter)
    total_documents = (await db.execute(doc_query)).scalar() or 0

    chunk_query = select(func.coalesce(func.sum(Document.chunk_count), 0))
    if dept_filter:
        chunk_query = chunk_query.join(Folder).where(Folder.department == dept_filter)
    total_chunks = (await db.execute(chunk_query)).scalar() or 0

    user_query = select(func.count(User.id))
    if dept_filter:
        user_query = user_query.where(User.department == dept_filter)
    total_users = (await db.execute(user_query)).scalar() or 0

    status_query = select(
        Document.status, func.count(Document.id)
    ).group_by(Document.status)
    if dept_filter:
        status_query = status_query.join(Folder).where(Folder.department == dept_filter)
    status_result = await db.execute(status_query)
    docs_by_status = {row[0].value: row[1] for row in status_result.all()}

    dept_query = select(
        Folder.department, func.count(Document.id)
    ).join(Document, isouter=True).group_by(Folder.department)
    dept_result = await db.execute(dept_query)
    docs_by_dept = {row[0]: row[1] or 0 for row in dept_result.all()}

    # Group stats (super admin only)
    total_groups = 0
    if is_super_admin:
        total_groups = (await db.execute(select(func.count(Group.id)))).scalar() or 0

    return DashboardStats(
        total_folders=total_folders,
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_users=total_users,
        documents_by_status=docs_by_status,
        documents_by_department=docs_by_dept,
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List users. Super admins see all; dept admins see their department only."""
    query = select(User)
    if current_user.role != RoleType.SUPER_ADMIN:
        query = query.where(User.department == current_user.department.value)
    query = query.order_by(User.created_at.desc())
    result = await db.execute(query)
    users = result.scalars().all()

    return UserListResponse(
        users=[UserResponse.model_validate(u) for u in users],
        total=len(users),
    )


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new user. Super admins create anyone; dept admins create users only."""
    target_role = RoleType(request.role)
    if not can_manage_users(current_user.role, target_role):
        raise HTTPException(status_code=403, detail="You cannot create users with this role.")

    if current_user.role != RoleType.SUPER_ADMIN:
        if request.department != current_user.department.value:
            raise HTTPException(status_code=403, detail="You can only create users in your department.")

    existing = await db.execute(select(User).where(User.email == request.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="A user with this email already exists.")

    new_user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        full_name=request.full_name,
        department=request.department,
        role=target_role,
    )
    db.add(new_user)
    await db.flush()
    await db.refresh(new_user)

    return UserResponse.model_validate(new_user)


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update user fields."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if current_user.role != RoleType.SUPER_ADMIN:
        if user.department != current_user.department.value:
            raise HTTPException(status_code=403, detail="Access denied.")

    if request.full_name is not None:
        user.full_name = request.full_name
    if request.department is not None:
        user.department = request.department
    if request.role is not None:
        target_role = RoleType(request.role)
        if not can_manage_users(current_user.role, target_role):
            raise HTTPException(status_code=403, detail="Cannot assign this role.")
        user.role = target_role
    if request.is_active is not None:
        user.is_active = request.is_active
    if request.password is not None:
        user.hashed_password = hash_password(request.password)

    await db.flush()
    await db.refresh(user)
    return UserResponse.model_validate(user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_role(RoleType.SUPER_ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user (super admin only)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself.")
    await db.delete(user)
    return {"message": f"User {user.email} deleted."}

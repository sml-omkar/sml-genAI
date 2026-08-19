"""
Group Management API Routes
Super admin manages groups, assigns users and folders.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.user import User, RoleType
from app.models.folder import Folder
from app.models.group import Group, UserGroup, GroupFolder
from app.auth.dependencies import get_current_user, require_super_admin
from app.schemas.folder import FolderResponse

router = APIRouter(prefix="/api/groups", tags=["groups"])


# --- Schemas ---
from pydantic import BaseModel


class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None


class GroupResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    is_active: bool
    user_count: int = 0
    folder_count: int = 0

    class Config:
        from_attributes = True


class GroupDetailResponse(GroupResponse):
    users: list = []
    folders: list = []


class UserGroupAssign(BaseModel):
    user_id: str


class FolderGroupAssign(BaseModel):
    folder_id: str


# =============================================================================
# Group CRUD
# =============================================================================


@router.get("", response_model=List[GroupResponse])
async def list_groups(
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all groups with user/folder counts."""
    from sqlalchemy import distinct

    result = await db.execute(
        select(
            Group,
            func.count(distinct(UserGroup.user_id)).label("user_count"),
            func.count(distinct(GroupFolder.folder_id)).label("folder_count"),
        )
        .outerjoin(UserGroup, UserGroup.group_id == Group.id)
        .outerjoin(GroupFolder, GroupFolder.group_id == Group.id)
        .group_by(Group.id)
        .order_by(Group.name)
    )
    groups = []
    for row in result.all():
        group = row[0]
        groups.append(GroupResponse(
            id=str(group.id),
            name=group.name,
            description=group.description,
            is_active=group.is_active,
            user_count=row[1],
            folder_count=row[2],
        ))
    return groups


@router.post("", response_model=GroupResponse, status_code=201)
async def create_group(
    data: GroupCreate,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new group."""
    existing = await db.execute(select(Group).where(Group.name == data.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Group name already exists")

    group = Group(name=data.name, description=data.description)
    db.add(group)
    await db.flush()
    await db.commit()
    await db.refresh(group)

    return GroupResponse(
        id=str(group.id), name=group.name, description=group.description,
        is_active=group.is_active, user_count=0, folder_count=0,
    )


@router.get("/{group_id}", response_model=GroupDetailResponse)
async def get_group(
    group_id: UUID,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get group details with users and folders."""
    result = await db.execute(
        select(Group).options(
            selectinload(Group.users).selectinload(UserGroup.user),
            selectinload(Group.folders).selectinload(GroupFolder.folder),
        ).where(Group.id == group_id)
    )
    group = result.scalar_one_or_none()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    return GroupDetailResponse(
        id=str(group.id), name=group.name, description=group.description,
        is_active=group.is_active,
        user_count=len(group.users),
        folder_count=len(group.folders),
        users=[{"id": str(ug.user.id), "email": ug.user.email, "name": ug.user.full_name} for ug in group.users],
        folders=[{"id": str(gf.folder.id), "name": gf.folder.name, "department": gf.folder.department} for gf in group.folders],
    )


@router.delete("/{group_id}")
async def delete_group(
    group_id: UUID,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a group and all its assignments."""
    result = await db.execute(select(Group).where(Group.id == group_id))
    group = result.scalar_one_or_none()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    await db.delete(group)
    await db.commit()
    return {"detail": "Group deleted"}


# =============================================================================
# User-Group Assignment
# =============================================================================


@router.post("/{group_id}/users")
async def add_user_to_group(
    group_id: UUID,
    data: UserGroupAssign,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """Add a user to a group."""
    group = await db.get(Group, group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    user = await db.get(User, UUID(data.user_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing = await db.execute(
        select(UserGroup).where(UserGroup.user_id == user.id, UserGroup.group_id == group_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="User already in group")

    ug = UserGroup(user_id=user.id, group_id=group_id)
    db.add(ug)
    await db.commit()
    return {"detail": f"User {user.email} added to group {group.name}"}


@router.delete("/{group_id}/users/{user_id}")
async def remove_user_from_group(
    group_id: UUID,
    user_id: UUID,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """Remove a user from a group."""
    result = await db.execute(
        select(UserGroup).where(UserGroup.group_id == group_id, UserGroup.user_id == user_id)
    )
    ug = result.scalar_one_or_none()
    if not ug:
        raise HTTPException(status_code=404, detail="User not in group")

    await db.delete(ug)
    await db.commit()
    return {"detail": "User removed from group"}


# =============================================================================
# Folder-Group Assignment
# =============================================================================


@router.post("/{group_id}/folders")
async def add_folder_to_group(
    group_id: UUID,
    data: FolderGroupAssign,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """Grant a group access to a folder."""
    group = await db.get(Group, group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    folder = await db.get(Folder, UUID(data.folder_id))
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    existing = await db.execute(
        select(GroupFolder).where(GroupFolder.group_id == group_id, GroupFolder.folder_id == folder.id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Group already has access to this folder")

    gf = GroupFolder(group_id=group_id, folder_id=folder.id)
    db.add(gf)
    await db.commit()
    return {"detail": f"Group {group.name} granted access to folder {folder.name}"}


@router.delete("/{group_id}/folders/{folder_id}")
async def remove_folder_from_group(
    group_id: UUID,
    folder_id: UUID,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
):
    """Revoke a group's access to a folder."""
    result = await db.execute(
        select(GroupFolder).where(GroupFolder.group_id == group_id, GroupFolder.folder_id == folder_id)
    )
    gf = result.scalar_one_or_none()
    if not gf:
        raise HTTPException(status_code=404, detail="Group does not have access to this folder")

    await db.delete(gf)
    await db.commit()
    return {"detail": "Folder access revoked from group"}

"""
RBAC Permissions
Centralized permission definitions for all operations.
Maps (role × action) → allowed/denied.
Group-based access: users access folders through their groups.
"""

from enum import Enum
from typing import Set, List, Optional
from dataclasses import dataclass
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import RoleType, User
from app.models.group import Group, UserGroup, GroupFolder
from app.models.folder import Folder


class Action(str, Enum):
    """Actions that can be performed on resources."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    UPLOAD = "upload"
    QUERY = "query"
    MANAGE_USERS = "manage_users"
    MANAGE_GROUPS = "manage_groups"


@dataclass
class Permission:
    """Defines what a role can do."""
    role: RoleType
    actions: Set[Action]
    scope: str  # "all", "own_department", "groups", "own"


# =============================================================================
# Permission Matrix
# =============================================================================
# super_admin:  ALL actions on ALL departments, manage users & groups
# dept_admin:   CRUD + upload on OWN department, manage users in dept
# user:         query only — via group access to specific folders

PERMISSIONS = {
    RoleType.SUPER_ADMIN: Permission(
        role=RoleType.SUPER_ADMIN,
        actions={Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE,
                 Action.UPLOAD, Action.QUERY, Action.MANAGE_USERS, Action.MANAGE_GROUPS},
        scope="all",
    ),
    RoleType.DEPT_ADMIN: Permission(
        role=RoleType.DEPT_ADMIN,
        actions={Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE,
                 Action.UPLOAD, Action.QUERY},
        scope="own_department",
    ),
    RoleType.USER: Permission(
        role=RoleType.USER,
        actions={Action.READ, Action.QUERY},
        scope="groups",
    ),
}


def has_permission(role: RoleType, action: Action) -> bool:
    perm = PERMISSIONS.get(role)
    if not perm:
        return False
    return action in perm.actions


def get_scope(role: RoleType) -> str:
    perm = PERMISSIONS.get(role)
    if not perm:
        return "none"
    return perm.scope


def can_access_department(user_role: RoleType, user_department: str, target_department: str) -> bool:
    if user_role == RoleType.SUPER_ADMIN:
        return True
    return user_department == target_department


def can_manage_users(actor_role: RoleType, target_role: RoleType) -> bool:
    if actor_role == RoleType.SUPER_ADMIN:
        return True
    if actor_role == RoleType.DEPT_ADMIN and target_role == RoleType.USER:
        return True
    return False


async def get_user_group_ids(user_id, db: AsyncSession) -> List[str]:
    """Get all group IDs a user belongs to."""
    result = await db.execute(
        select(UserGroup.group_id).where(UserGroup.user_id == user_id)
    )
    return [str(row[0]) for row in result.all()]


async def get_user_folder_ids(user_id, db: AsyncSession) -> List[str]:
    """Get all folder IDs a user has access to via their groups."""
    result = await db.execute(
        select(GroupFolder.folder_id)
        .join(UserGroup, UserGroup.group_id == GroupFolder.group_id)
        .where(UserGroup.user_id == user_id)
    )
    return [str(row[0]) for row in result.all()]


async def get_user_accessible_departments(user_id, db: AsyncSession) -> List[str]:
    """Get all departments a user can access via their groups (through folders)."""
    result = await db.execute(
        select(Folder.department)
        .join(GroupFolder, GroupFolder.folder_id == Folder.id)
        .join(UserGroup, UserGroup.group_id == GroupFolder.group_id)
        .where(UserGroup.user_id == user_id)
        .distinct()
    )
    return [row[0] for row in result.all()]


async def can_user_access_folder(user_id, folder_id, db: AsyncSession) -> bool:
    """Check if a user can access a specific folder via group membership."""
    result = await db.execute(
        select(GroupFolder.id)
        .join(UserGroup, UserGroup.group_id == GroupFolder.group_id)
        .where(UserGroup.user_id == user_id, GroupFolder.folder_id == folder_id)
    )
    return result.first() is not None


async def can_user_query_department(user_id: str, department: str, db: AsyncSession) -> bool:
    """Check if a user can query a department — user needs group access to at least one folder in that dept."""
    result = await db.execute(
        select(Folder.id)
        .join(GroupFolder, GroupFolder.folder_id == Folder.id)
        .join(UserGroup, UserGroup.group_id == GroupFolder.group_id)
        .where(UserGroup.user_id == user_id, Folder.department == department)
        .limit(1)
    )
    return result.first() is not None

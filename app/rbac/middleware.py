"""
RBAC Middleware
Enforces group-based and department-based filtering on database queries.
Ensures users can only see/manage resources within their access scope.
"""

from typing import Optional, List
from uuid import UUID

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, RoleType
from app.models.folder import Folder
from app.models.document import Document
from app.rbac.permissions import (
    can_access_department, get_scope,
    get_user_folder_ids, get_user_accessible_departments
)


async def filter_folders_by_access(query: Select, user: User, db: AsyncSession) -> Select:
    """
    Filter folders based on user's RBAC scope.
    - Super admins: all folders
    - Dept admins: their department's folders
    - Users: only folders their groups have access to
    """
    scope = get_scope(user.role)

    if scope == "all":
        return query

    if scope == "own_department":
        return query.where(Folder.department == user.department.value)

    # Group-based access (regular users)
    folder_ids = await get_user_folder_ids(user.id, db)
    if folder_ids:
        return query.where(Folder.id.in_(folder_ids))
    # No group access = no folders
    return query.where(Folder.id == None)


async def filter_documents_by_access(query: Select, user: User, db: AsyncSession) -> Select:
    """
    Filter documents based on user's RBAC scope.
    - Super admins: all documents
    - Dept admins: their department's documents
    - Users: only documents in folders their groups have access to
    """
    scope = get_scope(user.role)

    if scope == "all":
        return query

    if scope == "own_department":
        return query.join(Folder).where(Folder.department == user.department.value)

    # Group-based access (regular users)
    folder_ids = await get_user_folder_ids(user.id, db)
    if folder_ids:
        return query.where(Document.folder_id.in_(folder_ids))
    return query.where(Document.id == None)


async def filter_documents_by_folder_access(
    query: Select,
    user: User,
    db: AsyncSession,
    folder_id: Optional[UUID] = None,
) -> Select:
    scope = get_scope(user.role)

    if scope == "all":
        if folder_id:
            return query.where(Document.folder_id == folder_id)
        return query

    if scope == "own_department":
        if folder_id:
            return query.where(
                Document.folder_id == folder_id,
                Document.folder.has(Folder.department == user.department.value),
            )
        return query.join(Folder).where(Folder.department == user.department.value)

    # Group-based access
    folder_ids = await get_user_folder_ids(user.id, db)
    if folder_id:
        if str(folder_id) in folder_ids:
            return query.where(Document.folder_id == folder_id)
        return query.where(Document.id == None)
    if folder_ids:
        return query.where(Document.folder_id.in_(folder_ids))
    return query.where(Document.id == None)


async def validate_folder_access(user: User, folder: Folder, db: AsyncSession) -> bool:
    """Validate that a user can access a specific folder."""
    if user.role == RoleType.SUPER_ADMIN:
        return True
    if user.role == RoleType.DEPT_ADMIN:
        return user.department.value == folder.department
    # Regular user: check group access
    return await _user_has_folder_access(user.id, folder.id, db)


async def validate_document_access(user: User, document: Document, folder: Folder, db: AsyncSession) -> bool:
    if user.role == RoleType.SUPER_ADMIN:
        return True
    if user.role == RoleType.DEPT_ADMIN:
        return user.department.value == folder.department
    return await _user_has_folder_access(user.id, folder.id, db)


def get_user_department_filter(user: User) -> Optional[str]:
    """Returns department string for ChromaDB filtering, or None for super admins."""
    if user.role == RoleType.SUPER_ADMIN:
        return None
    return user.department.value


async def get_user_department_filter_list(user: User, db: AsyncSession) -> Optional[List[str]]:
    """
    Returns list of departments a user can query via groups.
    For super admins returns None (search all).
    For dept admins returns their department.
    For users returns departments their groups have access to.
    """
    if user.role == RoleType.SUPER_ADMIN:
        return None
    if user.role == RoleType.DEPT_ADMIN:
        return [user.department.value]
    # Regular user: get departments from group folder access
    depts = await get_user_accessible_departments(user.id, db)
    return depts if depts else []


async def _user_has_folder_access(user_id: UUID, folder_id: UUID, db: AsyncSession) -> bool:
    from app.models.group import UserGroup, GroupFolder
    from sqlalchemy import select
    result = await db.execute(
        select(GroupFolder.id)
        .join(UserGroup, UserGroup.group_id == GroupFolder.group_id)
        .where(UserGroup.user_id == user_id, GroupFolder.folder_id == folder_id)
    )
    return result.first() is not None

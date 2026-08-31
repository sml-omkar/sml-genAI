"""
Folder Routes
Create, list, update, and delete department folders.
Supports group-based access for regular users.
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User, RoleType
from app.models.folder import Folder
from app.models.document import Document
from app.schemas.folder import (
    FolderCreate,
    FolderUpdate,
    FolderResponse,
    FolderListResponse,
)
from app.auth.dependencies import get_current_user
from app.rbac.middleware import filter_folders_by_access, validate_folder_access

router = APIRouter()


@router.get("/", response_model=FolderListResponse)
async def list_folders(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List folders. RBAC filtered:
    - Super admins: all folders
    - Dept admins: their department's folders
    - Users: only folders their groups have access to
    """
    query = select(Folder).order_by(Folder.created_at.desc())
    query = await filter_folders_by_access(query, current_user, db)
    result = await db.execute(query)
    folders = result.scalars().all()

    return FolderListResponse(
        folders=[FolderResponse.model_validate(f) for f in folders],
        total=len(folders),
    )


@router.post("/", response_model=FolderResponse, status_code=201)
async def create_folder(
    request: FolderCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new folder. Dept admins can only create in their department."""
    if current_user.role == RoleType.USER:
        raise HTTPException(status_code=403, detail="Users cannot create folders.")

    if current_user.role != RoleType.SUPER_ADMIN:
        if request.department != current_user.department.value:
            raise HTTPException(status_code=403, detail="You can only create folders in your department.")

    existing = await db.execute(
        select(Folder).where(Folder.name == request.name, Folder.department == request.department)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"A folder named '{request.name}' already exists in {request.department}.")

    folder = Folder(
        name=request.name,
        description=request.description,
        department=request.department,
        created_by=current_user.id,
    )
    db.add(folder)
    await db.flush()
    await db.refresh(folder)

    return FolderResponse.model_validate(folder)


@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single folder by ID with RBAC check."""
    result = await db.execute(select(Folder).where(Folder.id == folder_id))
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")

    if not await validate_folder_access(current_user, folder, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    return FolderResponse.model_validate(folder)


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: str,
    request: FolderUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update folder name/description."""
    result = await db.execute(select(Folder).where(Folder.id == folder_id))
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")

    if current_user.role == RoleType.USER:
        raise HTTPException(status_code=403, detail="Users cannot update folders.")

    if not await validate_folder_access(current_user, folder, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    if request.name is not None:
        folder.name = request.name
    if request.description is not None:
        folder.description = request.description

    await db.flush()
    await db.refresh(folder)
    return FolderResponse.model_validate(folder)


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a folder and all its documents."""
    result = await db.execute(select(Folder).where(Folder.id == folder_id))
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")

    if current_user.role == RoleType.USER:
        raise HTTPException(status_code=403, detail="Users cannot delete folders.")

    if not await validate_folder_access(current_user, folder, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    doc_result = await db.execute(select(Document.id, Document.filename, Document.stored_path).where(Document.folder_id == folder_id))
    docs = doc_result.all()
    doc_ids = [str(row[0]) for row in docs]
    doc_names = [row[1] for row in docs]
    stored_paths = [row[2] for row in docs]

    if doc_ids:
        # Delete chunks + EthosAI memory notes from ChromaDB (both share document_id)
        try:
            from app.rag.vectorstore import delete_documents_from_vectordb
            await delete_documents_from_vectordb(doc_ids, doc_names)
        except Exception as e:
            print(f"[WARNING] ChromaDB cleanup failed: {e}")

        # Delete EthosAI global policy memories (DocumentMemory) for all docs in folder
        # This is the EthosAI brain, not per-user conversation memory
        try:
            from app.models.document_memory import DocumentMemory
            from sqlalchemy import delete as sa_delete
            from uuid import UUID as UUUID
            uuid_ids = [UUUID(d) for d in doc_ids]
            await db.execute(sa_delete(DocumentMemory).where(DocumentMemory.document_id.in_(uuid_ids)))
        except Exception as e:
            print(f"[WARNING] DocumentMemory cleanup failed: {e}")

        # Invalidate cache for this department
        try:
            from app.cache.service import cache_invalidate_department
            cache_invalidate_department(folder.department)
        except Exception:
            pass

        # Remove files from disk for each document
        import os as _os
        for p in stored_paths:
            try:
                if p and _os.path.exists(p):
                    _os.remove(p)
            except Exception as e:
                print(f"[WARNING] Failed to remove file {p}: {e}")

    await db.delete(folder)
    return {"message": f"Folder '{folder.name}' and all its documents deleted. Chunks, embeddings and EthosAI memories removed."}

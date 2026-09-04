"""
Folder-scoped External APIs — Admin creates a named API in a single department
and attaches one or more folders within that department.

External apps call POST /api/external/{slug}/chat with X-API-Key header.
RAG is scoped to attached folders' documents only (folder_ids filter in Chroma).
"""

import hashlib
import re
import secrets
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User, RoleType
from app.models.folder import Folder
from app.models.external_api import ExternalApi, ApiFolder, _slugify, generate_api_key
from app.auth.dependencies import get_current_user, require_admin
from app.config import get_settings

settings = get_settings()

# Admin CRUD router (JWT auth)
admin_router = APIRouter(prefix="/api/external-apis", tags=["External APIs"])
# Public chat router (X-API-Key auth, no JWT)
chat_router = APIRouter(prefix="/api/external", tags=["External Chat"])


# --- Schemas ---
class ExternalApiCreate(BaseModel):
    name: str
    department: str
    description: Optional[str] = None
    folder_ids: List[str] = []


class ExternalApiUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    folder_ids: Optional[List[str]] = None  # full replace if provided


class ExternalApiResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str]
    department: str
    api_key_prefix: Optional[str]
    is_active: bool
    folder_count: int = 0
    folders: List[dict] = []
    created_by: str
    created_at: str

    class Config:
        from_attributes = True


class AttachFolderRequest(BaseModel):
    folder_id: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


# --- Helpers ---
def _hash_key(plain: str) -> str:
    return hashlib.sha256(plain.encode()).hexdigest()


async def _validate_department_slug(db: AsyncSession, slug: str) -> str:
    from app.models.department import Department
    slug = slug.lower().strip()
    res = await db.execute(select(Department).where(Department.slug == slug, Department.is_active == True))
    dept = res.scalar_one_or_none()
    if not dept:
        # Also allow raw string if department table empty? but require valid
        raise HTTPException(status_code=400, detail=f"Department '{slug}' does not exist or is inactive. Create it via /api/departments first.")
    return dept.slug


async def _can_admin_manage_dept(current_user: User, target_dept: str):
    if current_user.role == RoleType.SUPER_ADMIN:
        return True
    if current_user.role == RoleType.DEPT_ADMIN and current_user.department.value == target_dept:
        return True
    raise HTTPException(status_code=403, detail=f"You can only manage APIs in your department '{current_user.department.value}'.")


async def _validate_folders_in_dept(db: AsyncSession, folder_ids: List[str], target_dept: str):
    if not folder_ids:
        return []
    # Deduplicate
    uniq = list(dict.fromkeys(folder_ids))
    result = await db.execute(select(Folder).where(Folder.id.in_(uniq)))
    folders = result.scalars().all()
    found = {str(f.id): f for f in folders}
    missing = [fid for fid in uniq if fid not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"Folders not found: {missing}")
    for fid, f in found.items():
        if f.department.lower() != target_dept.lower():
            raise HTTPException(status_code=400, detail=f"Folder '{f.name}' is in department '{f.department}', but API is in '{target_dept}'. All attached folders must be in the API's single department.")
    return [found[fid] for fid in uniq]


# --- Admin CRUD ---

@admin_router.get("", response_model=List[dict])
async def list_apis(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List APIs. Dept admins see only their department."""
    q = select(ExternalApi).order_by(ExternalApi.created_at.desc())
    if current_user.role != RoleType.SUPER_ADMIN:
        q = q.where(ExternalApi.department == current_user.department.value)
    res = await db.execute(q)
    apis = res.scalars().all()
    out = []
    for api in apis:
        cnt = (await db.execute(select(ApiFolder).where(ApiFolder.api_id == api.id))).scalars().all()
        out.append({
            "id": str(api.id),
            "name": api.name,
            "slug": api.slug,
            "description": api.description,
            "department": api.department,
            "api_key_prefix": api.api_key_prefix,
            "is_active": api.is_active,
            "folder_count": len(cnt),
            "created_by": str(api.created_by),
            "created_at": api.created_at.isoformat(),
        })
    return out


@admin_router.post("", status_code=201)
async def create_api(
    data: ExternalApiCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a folder-scoped API. Generates a key shown once."""
    await _can_admin_manage_dept(current_user, data.department.lower())
    dept_slug = await _validate_department_slug(db, data.department)
    folders = await _validate_folders_in_dept(db, data.folder_ids, dept_slug)

    slug = _slugify(data.name)
    # Ensure slug unique
    existing = await db.execute(select(ExternalApi).where(ExternalApi.slug == slug))
    if existing.scalar_one_or_none():
        slug = f"{slug}-{secrets.token_hex(3)}"

    plain, key_hash, prefix = generate_api_key()
    api = ExternalApi(
        name=data.name.strip(),
        slug=slug,
        description=(data.description or "").strip() or None,
        department=dept_slug,
        api_key_hash=key_hash,
        api_key_prefix=prefix,
        is_active=True,
        created_by=current_user.id,
    )
    db.add(api)
    await db.flush()

    for f in folders:
        db.add(ApiFolder(api_id=api.id, folder_id=f.id))
    await db.flush()
    await db.refresh(api)

    return {
        "id": str(api.id),
        "name": api.name,
        "slug": api.slug,
        "description": api.description,
        "department": api.department,
        "api_key": plain,  # shown once
        "api_key_prefix": api.api_key_prefix,
        "is_active": api.is_active,
        "folder_ids": [str(f.id) for f in folders],
        "endpoint": f"/api/external/{api.slug}/chat",
        "curl_example": f"curl -X POST http://HOST/api/external/{api.slug}/chat -H 'X-API-Key: {plain}' -H 'Content-Type: application/json' -d '{{\"message\":\"hello\"}}'",
        "warning": "Copy the api_key now — it is not stored in plaintext and cannot be retrieved again. Rotate if leaked.",
    }


@admin_router.get("/{api_id}")
async def get_api(
    api_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    api = await db.get(ExternalApi, uuid.UUID(api_id))
    if not api:
        raise HTTPException(status_code=404, detail="API not found")
    if current_user.role != RoleType.SUPER_ADMIN and api.department != current_user.department.value:
        raise HTTPException(status_code=403, detail="Access denied for this department")
    folders_res = await db.execute(select(ApiFolder).where(ApiFolder.api_id == api.id))
    api_folders = folders_res.scalars().all()
    folder_ids = [str(af.folder_id) for af in api_folders]
    folder_objs = []
    if folder_ids:
        fres = await db.execute(select(Folder).where(Folder.id.in_(folder_ids)))
        for f in fres.scalars().all():
            folder_objs.append({"id": str(f.id), "name": f.name, "department": f.department, "document_count": f.document_count})
    return {
        "id": str(api.id),
        "name": api.name,
        "slug": api.slug,
        "description": api.description,
        "department": api.department,
        "api_key_prefix": api.api_key_prefix,
        "is_active": api.is_active,
        "folders": folder_objs,
        "folder_ids": folder_ids,
        "endpoint": f"/api/external/{api.slug}/chat",
        "created_by": str(api.created_by),
        "created_at": api.created_at.isoformat(),
    }


@admin_router.put("/{api_id}")
async def update_api(
    api_id: str,
    data: ExternalApiUpdate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    api = await db.get(ExternalApi, uuid.UUID(api_id))
    if not api:
        raise HTTPException(status_code=404, detail="API not found")
    await _can_admin_manage_dept(current_user, api.department)
    if data.name is not None:
        api.name = data.name.strip()
    if data.description is not None:
        api.description = data.description.strip() or None
    if data.is_active is not None:
        api.is_active = data.is_active
    if data.folder_ids is not None:
        # Replace all attachments
        folders = await _validate_folders_in_dept(db, data.folder_ids, api.department)
        # Delete old
        await db.execute(select(ApiFolder).where(ApiFolder.api_id == api.id))
        # Use delete via ORM
        old = (await db.execute(select(ApiFolder).where(ApiFolder.api_id == api.id))).scalars().all()
        for af in old:
            await db.delete(af)
        for f in folders:
            db.add(ApiFolder(api_id=api.id, folder_id=f.id))
    await db.flush()
    await db.refresh(api)
    return {"detail": "Updated", "id": str(api.id), "slug": api.slug}


@admin_router.delete("/{api_id}")
async def delete_api(
    api_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    api = await db.get(ExternalApi, uuid.UUID(api_id))
    if not api:
        raise HTTPException(status_code=404, detail="API not found")
    await _can_admin_manage_dept(current_user, api.department)
    await db.delete(api)
    await db.flush()
    return {"detail": f"API '{api.name}' deleted"}


@admin_router.post("/{api_id}/rotate-key")
async def rotate_key(
    api_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    api = await db.get(ExternalApi, uuid.UUID(api_id))
    if not api:
        raise HTTPException(status_code=404, detail="API not found")
    await _can_admin_manage_dept(current_user, api.department)
    plain, key_hash, prefix = generate_api_key()
    api.api_key_hash = key_hash
    api.api_key_prefix = prefix
    await db.flush()
    return {"api_key": plain, "api_key_prefix": prefix, "warning": "Copy now — not stored plaintext"}


@admin_router.post("/{api_id}/folders")
async def attach_folder(
    api_id: str,
    data: AttachFolderRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    api = await db.get(ExternalApi, uuid.UUID(api_id))
    if not api:
        raise HTTPException(status_code=404, detail="API not found")
    await _can_admin_manage_dept(current_user, api.department)
    folder = await db.get(Folder, uuid.UUID(data.folder_id))
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    if folder.department.lower() != api.department.lower():
        raise HTTPException(status_code=400, detail=f"Folder department '{folder.department}' != API department '{api.department}'")
    existing = await db.execute(select(ApiFolder).where(ApiFolder.api_id == api.id, ApiFolder.folder_id == folder.id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Folder already attached")
    db.add(ApiFolder(api_id=api.id, folder_id=folder.id))
    await db.flush()
    return {"detail": f"Folder '{folder.name}' attached to API '{api.name}'"}


@admin_router.delete("/{api_id}/folders/{folder_id}")
async def detach_folder(
    api_id: str,
    folder_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    api = await db.get(ExternalApi, uuid.UUID(api_id))
    if not api:
        raise HTTPException(status_code=404, detail="API not found")
    await _can_admin_manage_dept(current_user, api.department)
    af = (await db.execute(select(ApiFolder).where(ApiFolder.api_id == api.id, ApiFolder.folder_id == uuid.UUID(folder_id)))).scalar_one_or_none()
    if not af:
        raise HTTPException(status_code=404, detail="Folder not attached to this API")
    await db.delete(af)
    await db.flush()
    return {"detail": "Folder detached"}


# --- Public chat (X-API-Key) ---

@chat_router.post("/{slug}/chat")
async def external_chat(
    slug: str,
    body: ChatRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    External app chat. No JWT. Auth via X-API-Key header.
    RAG is scoped to the API's attached folders only.
    """
    if not x_api_key:
        # Also accept Authorization: Bearer <key> for convenience
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            x_api_key = auth[7:].strip()
        else:
            # Check query param ?api_key= for simple embeds
            x_api_key = request.query_params.get("api_key")

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header (or Authorization: Bearer <key>)")

    # Find API by slug (case-insensitive)
    result = await db.execute(select(ExternalApi).where(ExternalApi.slug == slug.lower()))
    api = result.scalar_one_or_none()
    if not api:
        raise HTTPException(status_code=404, detail=f"API '{slug}' not found")
    if not api.is_active:
        raise HTTPException(status_code=403, detail="This API is disabled")

    # Verify key (constant-time compare via hash)
    provided_hash = _hash_key(x_api_key.strip())
    if provided_hash != api.api_key_hash:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Resolve attached folders (must exist, else 503)
    fres = await db.execute(select(ApiFolder.folder_id).where(ApiFolder.api_id == api.id))
    folder_ids = [str(r[0]) for r in fres.all()]
    if not folder_ids:
        raise HTTPException(status_code=400, detail="This API has no folders attached yet. Attach at least one folder via admin panel.")

    # Question validation
    q = (body.message or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="message is required")
    if len(q) > 4000:
        raise HTTPException(status_code=400, detail="message too long (max 4000 chars)")

    # Memory: per-API conversation (isolated by api_id)
    from app.memory.service import get_memory_service
    from app.rag.agent import query_rag

    settings = get_settings()
    memory = get_memory_service(ttl_hours=settings.CONVERSATION_TTL_HOURS, max_messages=settings.MEMORY_MAX_MESSAGES)

    # Use external_api_id to namespace conversations (different api's chats don't leak)
    # conversation_id client can provide is namespaced per API
    client_cid = body.conversation_id
    conv_id_for_lookup = None
    if client_cid:
        # Namespace client id by api to prevent cross-API hijacking
        import uuid as _uuid
        conv_id_for_lookup = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"ext-{api.slug}-{client_cid}"))
    else:
        # Auto-create ephemeral per-API conversation (no client id)
        import uuid as _uuid
        auto = _uuid.uuid4().hex[:12]
        conv_id_for_lookup = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"ext-{api.slug}-{auto}"))

    conv = await memory.get_or_create_conversation(
        conversation_id=conv_id_for_lookup,
        user_id=None,  # external callers are not User rows
        source="external",
        external_api_id=str(api.id),
    )
    conv_id = str(conv.id)
    history = await memory.get_history(conv_id)
    await memory.add_message(conversation_id=conv_id, role="user", content=q)

    result = await query_rag(
        question=q,
        department=None,  # folder_ids already scopes, ignore dept
        folder_ids=folder_ids,
        chat_history=history,
        include_usage=True,
    )
    usage = result.get("usage", {})
    await memory.add_message(
        conversation_id=conv_id,
        role="assistant",
        content=result.get("answer", ""),
        sources=result.get("sources"),
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
        model_used=settings.OPENAI_MODEL,
    )

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "chunks_retrieved": result.get("chunks_retrieved", 0),
        "conversation_id": conv_id,  # client can reuse this as conversation_id next turn
        "api": {"slug": api.slug, "name": api.name, "department": api.department},
        "usage": usage,
    }

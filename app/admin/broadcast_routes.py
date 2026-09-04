"""
Admin Broadcast API — send proactive Teams messages to all known Teams conversations
"""

import asyncio
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models.user import User, RoleType
from app.models.broadcast import TeamsProactiveRef, Broadcast, BroadcastRecipient
from app.auth.dependencies import require_admin
from app.database import AsyncSessionLocal

router = APIRouter(prefix="/api/admin/broadcast", tags=["Broadcast"])


class BroadcastCreate(BaseModel):
    message: str
    # Optional filter: only send to a tenant or to specific conversation_ids
    tenant_id: Optional[str] = None
    # If provided, only these ref IDs will be targeted (else all active)
    ref_ids: Optional[List[str]] = None
    # Dry run: validate but don't send
    dry_run: bool = False


@router.get("/recipients")
async def list_recipients(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List known Teams conversation refs that broadcast will reach (active only)."""
    from app.bot.proactive import get_active_refs
    # Dept admin sees only their tenant? For now all, but filter by tenant if desired
    refs = await get_active_refs(limit=500)
    return {
        "total": len(refs),
        "recipients": [
            {
                "id": str(r.id),
                "user_name": r.user_name,
                "user_email": r.user_email,
                "aad_object_id": r.aad_object_id,
                "teams_user_id": r.teams_user_id,
                "conversation_id": r.conversation_id,
                "tenant_id": r.tenant_id,
                "service_url": r.service_url,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                "is_active": r.is_active,
            }
            for r in refs
        ],
    }


@router.get("/history")
async def broadcast_history(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(Broadcast).order_by(Broadcast.created_at.desc()).limit(50))
    broadcasts = res.scalars().all()
    out = []
    for b in broadcasts:
        out.append({
            "id": str(b.id),
            "message": b.message,
            "status": b.status,
            "total_recipients": b.total_recipients,
            "success_count": b.success_count,
            "failed_count": b.failed_count,
            "filter_info": b.filter_info,
            "created_by": str(b.created_by),
            "created_at": b.created_at.isoformat() if b.created_at else None,
            "completed_at": b.completed_at.isoformat() if b.completed_at else None,
        })
    return {"broadcasts": out}


@router.get("/{broadcast_id}")
async def broadcast_detail(
    broadcast_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    b = await db.get(Broadcast, UUID(broadcast_id))
    if not b:
        raise HTTPException(status_code=404, detail="Broadcast not found")
    recs = (await db.execute(select(BroadcastRecipient).where(BroadcastRecipient.broadcast_id == b.id).order_by(BroadcastRecipient.status))).scalars().all()
    return {
        "id": str(b.id),
        "message": b.message,
        "status": b.status,
        "total_recipients": b.total_recipients,
        "success_count": b.success_count,
        "failed_count": b.failed_count,
        "filter_info": b.filter_info,
        "created_at": b.created_at.isoformat() if b.created_at else None,
        "completed_at": b.completed_at.isoformat() if b.completed_at else None,
        "recipients": [
            {
                "id": str(r.id),
                "teams_user_id": r.teams_user_id,
                "user_name": r.user_name,
                "conversation_id": r.conversation_id,
                "status": r.status,
                "error": r.error,
                "sent_at": r.sent_at.isoformat() if r.sent_at else None,
            }
            for r in recs
        ],
    }


@router.post("")
async def create_broadcast(
    data: BroadcastCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    msg = (data.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    if len(msg) > 4000:
        raise HTTPException(status_code=400, detail="message too long (max 4000 chars)")

    # Resolve recipients
    if data.ref_ids:
        # Explicit list
        q = select(TeamsProactiveRef).where(TeamsProactiveRef.id.in_([UUID(x) for x in data.ref_ids]), TeamsProactiveRef.is_active == True)
        refs = (await db.execute(q)).scalars().all()
        # Also ensure requested IDs were found
        found_ids = {str(r.id) for r in refs}
        missing = [x for x in data.ref_ids if x not in found_ids]
        if missing:
            raise HTTPException(status_code=400, detail=f"Some ref_ids not found or inactive: {missing}")
    else:
        q = select(TeamsProactiveRef).where(TeamsProactiveRef.is_active == True)
        if data.tenant_id:
            q = q.where(TeamsProactiveRef.tenant_id == data.tenant_id)
        q = q.order_by(TeamsProactiveRef.last_seen.desc()).limit(1000)
        refs = (await db.execute(q)).scalars().all()

    if not refs:
        raise HTTPException(status_code=400, detail="No active Teams recipients found. Users must have chatted with the bot at least once to be reachable for broadcast (proactive refs).")

    if data.dry_run:
        return {
            "dry_run": True,
            "would_send_to": len(refs),
            "recipients_preview": [
                {"id": str(r.id), "user_name": r.user_name, "teams_user_id": r.teams_user_id, "conversation_id": r.conversation_id}
                for r in refs[:20]
            ],
        }

    # Create broadcast record
    b = Broadcast(
        message=msg,
        created_by=current_user.id,
        status="queued",
        total_recipients=len(refs),
        filter_info={"tenant_id": data.tenant_id, "ref_ids": data.ref_ids} if (data.tenant_id or data.ref_ids) else None,
    )
    db.add(b)
    await db.flush()

    # Create recipient rows
    for r in refs:
        db.add(BroadcastRecipient(
            broadcast_id=b.id,
            ref_id=r.id,
            teams_user_id=r.teams_user_id,
            user_name=r.user_name,
            conversation_id=r.conversation_id,
            status="pending",
        ))
    await db.flush()
    bid = str(b.id)
    await db.commit()

    # Fire-and-forget background send
    from app.bot.proactive import send_broadcast
    asyncio.create_task(send_broadcast(bid))

    return {
        "id": bid,
        "status": "queued",
        "total_recipients": len(refs),
        "detail": f"Broadcast queued to {len(refs)} recipients. Poll GET /api/admin/broadcast/{bid} for progress.",
    }


@router.post("/{broadcast_id}/retry-failed")
async def retry_failed(
    broadcast_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    b = await db.get(Broadcast, UUID(broadcast_id))
    if not b:
        raise HTTPException(status_code=404, detail="Broadcast not found")
    # Reset failed recipients to pending and re-queue
    recs = (await db.execute(select(BroadcastRecipient).where(BroadcastRecipient.broadcast_id == b.id, BroadcastRecipient.status == "failed"))).scalars().all()
    if not recs:
        raise HTTPException(status_code=400, detail="No failed recipients to retry")
    for r in recs:
        r.status = "pending"
        r.error = None
    b.status = "queued"
    await db.commit()
    from app.bot.proactive import send_broadcast
    asyncio.create_task(send_broadcast(str(b.id)))
    return {"detail": f"Retrying {len(recs)} failed recipients"}

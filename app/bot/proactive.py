"""
Teams Proactive Messaging — store ConversationReference and broadcast
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict
from uuid import UUID

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.broadcast import TeamsProactiveRef, Broadcast, BroadcastRecipient


async def save_proactive_ref(turn_context) -> Optional[TeamsProactiveRef]:
    """
    Called on every Teams turn. Captures ConversationReference for proactive use.
    Upserts by (teams_user_id + conversation_id). Never throws — broadcast must not block chat.
    """
    try:
        activity = turn_context.activity
        # Build reference via TurnContext helper
        ref = turn_context.get_conversation_reference(activity) if hasattr(turn_context, "get_conversation_reference") else None
        # Fallback manual build if helper missing
        if not ref:
            # minimal reference
            ref = {
                "serviceUrl": getattr(activity, "service_url", "") or getattr(activity, "serviceUrl", ""),
                "channelId": getattr(activity, "channel_id", "msteams"),
                "conversation": {"id": getattr(getattr(activity, "conversation", None), "id", ""), "name": getattr(getattr(activity, "conversation", None), "name", "")},
                "user": {"id": getattr(getattr(activity, "from_property", None), "id", ""), "name": getattr(getattr(activity, "from_property", None), "name", "")},
                "bot": {"id": getattr(getattr(activity, "recipient", None), "id", "")},
                "locale": getattr(activity, "locale", None),
            }
        # Normalize to dict
        if hasattr(ref, "as_dict"):
            ref_dict = ref.as_dict()
        elif isinstance(ref, dict):
            ref_dict = ref
        else:
            # ConversationReference object
            try:
                ref_dict = dict(ref)  # type: ignore
            except Exception:
                ref_dict = {"serviceUrl": getattr(ref, "service_url", ""), "channelId": getattr(ref, "channel_id", "msteams")}

        service_url = ref_dict.get("serviceUrl") or getattr(activity, "service_url", "") or ""
        channel_id = ref_dict.get("channelId") or getattr(activity, "channel_id", "msteams")
        conv_id = (ref_dict.get("conversation") or {}).get("id") or getattr(getattr(activity, "conversation", None), "id", "")
        user_id = (ref_dict.get("user") or {}).get("id") or getattr(getattr(activity, "from_property", None), "id", "")
        tenant_id = None
        team_id = None
        conv_type = None
        try:
            raw = activity.as_dict() if hasattr(activity, "as_dict") else {}
            cd = raw.get("channelData") or {}
            tenant_id = (cd.get("tenant") or {}).get("id")
            team_id = (cd.get("team") or {}).get("id")
            conv_type = (ref_dict.get("conversation") or {}).get("conversationType") or cd.get("conversationType")
        except Exception:
            pass
        # Identity helpers already extracted in bot_handler, but also capture here
        user = getattr(activity, "from_property", None)
        aad = getattr(user, "aad_object_id", None) if user else None
        if not aad and user and getattr(user, "additional_properties", None):
            aad = user.additional_properties.get("aadObjectId") if isinstance(user.additional_properties, dict) else None
        email = getattr(user, "email", None) if user else None
        name = getattr(user, "name", None) if user else None

        if not user_id or not conv_id or not service_url:
            print(f"[PROACTIVE] Skipping ref save — missing ids: user={user_id} conv={conv_id} url={bool(service_url)}")
            return None

        async with AsyncSessionLocal() as db:
            # Upsert: find existing by teams_user_id+conversation_id
            existing = (await db.execute(
                select(TeamsProactiveRef).where(
                    TeamsProactiveRef.teams_user_id == user_id,
                    TeamsProactiveRef.conversation_id == conv_id,
                )
            )).scalar_one_or_none()
            if existing:
                existing.aad_object_id = aad or existing.aad_object_id
                existing.user_name = name or existing.user_name
                existing.user_email = email or existing.user_email
                existing.service_url = service_url
                existing.tenant_id = tenant_id or existing.tenant_id
                existing.team_id = team_id or existing.team_id
                existing.conversation_type = conv_type or existing.conversation_type
                existing.reference_json = ref_dict
                existing.last_seen = datetime.utcnow()
                existing.is_active = True
                await db.commit()
                await db.refresh(existing)
                return existing
            ref_row = TeamsProactiveRef(
                aad_object_id=aad,
                teams_user_id=user_id,
                user_name=name,
                user_email=email,
                conversation_id=conv_id,
                service_url=service_url,
                channel_id=channel_id,
                tenant_id=tenant_id,
                team_id=team_id,
                conversation_type=conv_type,
                reference_json=ref_dict,
                is_active=True,
            )
            db.add(ref_row)
            await db.commit()
            await db.refresh(ref_row)
            print(f"[PROACTIVE] Saved ref for {name} ({user_id}) conv={conv_id[:30]}")
            return ref_row
    except Exception as e:
        print(f"[PROACTIVE] save ref failed (non-fatal): {e}")
        return None


async def get_active_refs(tenant_id: Optional[str] = None, limit: int = 500) -> List[TeamsProactiveRef]:
    async with AsyncSessionLocal() as db:
        q = select(TeamsProactiveRef).where(TeamsProactiveRef.is_active == True).order_by(TeamsProactiveRef.last_seen.desc()).limit(limit)
        if tenant_id:
            q = q.where(TeamsProactiveRef.tenant_id == tenant_id)
        res = await db.execute(q)
        return list(res.scalars().all())


async def deactivate_ref(ref_id: str):
    async with AsyncSessionLocal() as db:
        from sqlalchemy import update
        await db.execute(
            update(TeamsProactiveRef).where(TeamsProactiveRef.id == UUID(ref_id)).values(is_active=False)
        )
        await db.commit()


async def send_broadcast(broadcast_id: str):
    """
    Background worker: iterate BroadcastRecipients with status pending and send via adapter.continue_conversation
    """
    from app.bot.adapter import adapter
    from botbuilder.schema import Activity

    if adapter is None:
        print("[BROADCAST] No adapter — Teams not configured, marking failed")
        async with AsyncSessionLocal() as db:
            b = await db.get(Broadcast, UUID(broadcast_id))
            if b:
                b.status = "failed"
                b.completed_at = datetime.utcnow()
                await db.commit()
        return

    async with AsyncSessionLocal() as db:
        broadcast = await db.get(Broadcast, UUID(broadcast_id))
        if not broadcast:
            print(f"[BROADCAST] {broadcast_id} not found")
            return
        broadcast.status = "running"
        await db.commit()

        # Fetch pending recipients
        q = select(BroadcastRecipient).where(BroadcastRecipient.broadcast_id == broadcast.id, BroadcastRecipient.status == "pending")
        recs = (await db.execute(q)).scalars().all()
        print(f"[BROADCAST] Sending {len(recs)} messages for {broadcast_id}")

        for rec in recs:
            # Load ref
            ref_row = await db.get(TeamsProactiveRef, rec.ref_id) if rec.ref_id else None
            if not ref_row or not ref_row.reference_json:
                rec.status = "failed"
                rec.error = "No conversation reference"
                continue
            ref = ref_row.reference_json
            # Bot Framework expects ConversationReference as dict with serviceUrl etc.
            try:
                # Use adapter.continue_conversation
                def _callback(turn_context):
                    return turn_context.send_activity(broadcast.message)

                # continue_conversation is async and handles trust of serviceUrl
                await adapter.continue_conversation(ref, _callback, bot_id=None)
                rec.status = "sent"
                rec.sent_at = datetime.utcnow()
            except Exception as e:
                # Common: Bot not installed for user, serviceUrl expired, 403
                err = str(e)[:500]
                print(f"[BROADCAST] Failed to {rec.teams_user_id}: {err}")
                rec.status = "failed"
                rec.error = err
                # If 403 BotNotInConversation, deactivate ref
                if "403" in err or "BotNotInConversation" in err or "ConversationNotFound" in err:
                    ref_row.is_active = False

        # Tally
        all_recs = (await db.execute(select(BroadcastRecipient).where(BroadcastRecipient.broadcast_id == broadcast.id))).scalars().all()
        broadcast.success_count = sum(1 for r in all_recs if r.status == "sent")
        broadcast.failed_count = sum(1 for r in all_recs if r.status == "failed")
        broadcast.status = "done"
        broadcast.completed_at = datetime.utcnow()
        await db.commit()
        print(f"[BROADCAST] {broadcast_id} done: {broadcast.success_count} sent, {broadcast.failed_count} failed")

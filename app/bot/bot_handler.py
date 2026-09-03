"""
Teams Bot Handler
Processes messages from Teams, identifies user via Teams identity,
checks group-based access, queries RAG pipeline, responds with Adaptive Cards.
"""

import json
import uuid
from fastapi import APIRouter, Request, Response
from botbuilder.core import TurnContext
from botbuilder.schema import Activity, ActivityTypes
from botbuilder.core.teams import TeamsActivityHandler

from app.bot.adapter import adapter
from app.bot.card_builder import (
    build_welcome_card,
    build_answer_card,
    build_error_card,
    build_no_results_card,
    create_attachment,
)

router = APIRouter()


class PolicyBot(TeamsActivityHandler):
    """
    Teams bot that answers questions from company policy documents.
    Uses group-based RBAC: checks user's groups → folder access → department filter.
    """

    def _extract_teams_identity(self, turn_context: TurnContext):
        """
        Robustly extract Teams sender identity.

        Teams reliably sends `aadObjectId` (mapped to `aad_object_id` in the
        Python SDK) on EVERY message, but does NOT always send `email`.
        We therefore:
          1. Try every known location for the AAD ID (SDK attr, additional
             properties, raw channelData, raw activity JSON).
          2. Try every known location for email / UPN / name.
          3. Log the raw identifiers so EC2 logs can be debugged without
             needing a local Teams repro.
        Returns (aad_id, email, name, channel_id, raw_debug_dict)
        """
        activity = turn_context.activity
        user = getattr(activity, "from_property", None)

        # --- AAD Object ID (most reliable) ---
        aad_id = None
        # SDK canonical snake_case
        if user is not None:
            aad_id = getattr(user, "aad_object_id", None) or getattr(user, "aadObjectId", None) or None
            # additional_properties dict (some SDK versions stash it there)
            addl = getattr(user, "additional_properties", None)
            if not aad_id and isinstance(addl, dict):
                aad_id = addl.get("aadObjectId") or addl.get("aad_object_id")
            # properties dict
            props = getattr(user, "properties", None)
            if not aad_id and isinstance(props, dict):
                aad_id = props.get("aadObjectId") or props.get("aad_object_id")
        # Fallback: raw activity JSON (activity.channel_data / value)
        if not aad_id:
            try:
                raw = activity.as_dict() if hasattr(activity, "as_dict") else {}
                # from.aadObjectId in raw JSON
                from_raw = raw.get("from") or {}
                aad_id = from_raw.get("aadObjectId") or from_raw.get("aad_object_id")
                if not aad_id:
                    # channelData.tenant.id sometimes, but not user AAD
                    pass
            except Exception:
                pass

        # --- Email / UPN ---
        email = ""
        name = ""
        if user is not None:
            # name is always present (display name)
            name = getattr(user, "name", "") or ""
            # Try several email-ish attributes
            for attr in ("email", "user_principal_name", "upn", "userPrincipalName"):
                val = getattr(user, attr, None)
                if val:
                    email = val
                    break
            # additional_properties
            if not email and getattr(user, "additional_properties", None):
                addl = user.additional_properties
                if isinstance(addl, dict):
                    email = addl.get("email") or addl.get("userPrincipalName") or addl.get("upn") or ""
            # properties dict
            if not email and getattr(user, "properties", None):
                props = user.properties
                if isinstance(props, dict):
                    email = props.get("email") or props.get("userPrincipalName") or ""

        # --- Channel / conversation id for debugging ---
        channel_id = ""
        try:
            channel_id = getattr(activity.channel_data, "tenant", {}).get("id") if isinstance(getattr(activity, "channel_data", None), dict) else ""
        except Exception:
            pass

        raw_debug = {
            "from_id": getattr(user, "id", None) if user else None,
            "from_name": name,
            "aad_object_id": aad_id,
            "email": email,
            "channelId": getattr(activity, "channel_id", None),
            "conversation_id": getattr(getattr(activity, "conversation", None), "id", None) if getattr(activity, "conversation", None) else None,
        }
        return aad_id, email, name, channel_id, raw_debug

    async def on_message_activity(self, turn_context: TurnContext):
        # --- Extract user identity from Teams activity ---
        aad_id, user_email, user_name, channel_tenant_id, _dbg = self._extract_teams_identity(turn_context)
        print(f"[BOT] Teams sender identity: { _dbg }")

        user_dept = None
        db_user = None
        teams_aad_id = aad_id
        teams_email = user_email
        teams_name = user_name

        # --- Look up user + group access from database ---
        try:
            from app.database import AsyncSessionLocal
            from app.models.user import User, RoleType
            from app.models.group import UserGroup, GroupFolder
            from app.models.folder import Folder
            from sqlalchemy import select

            async with AsyncSessionLocal() as db:
                # Find user by AAD object ID FIRST (most reliable — Teams always sends this)
                if aad_id:
                    result = await db.execute(select(User).where(User.aad_object_id == aad_id))
                    db_user = result.scalar_one_or_none()
                    if db_user:
                        print(f"[BOT] Matched Teams user by AAD ID {aad_id} -> {db_user.email}")
                    else:
                        print(f"[BOT] No DB user found for AAD ID {aad_id}")

                # Fallback: lookup by email / UPN if AAD did not match
                if not db_user and user_email:
                    result = await db.execute(select(User).where(User.email == user_email))
                    db_user = result.scalar_one_or_none()
                    if db_user:
                        print(f"[BOT] Matched Teams user by email {user_email} -> AAD {db_user.aad_object_id}")
                        # Opportunistically backfill AAD ID if DB row has none
                        if aad_id and not db_user.aad_object_id:
                            try:
                                db_user.aad_object_id = aad_id
                                await db.commit()
                                print(f"[BOT] Backfilled AAD ID {aad_id} for {user_email}")
                            except Exception as e:
                                print(f"[BOT] Failed to backfill AAD ID: {e}")
                                await db.rollback()
                    else:
                        print(f"[BOT] No DB user found for email {user_email}")

                if db_user:
                    # Super admin: can query all departments
                    if db_user.role == RoleType.SUPER_ADMIN:
                        user_dept = None  # None = no department filter = search all
                    # Dept admin: can query their own department
                    elif db_user.role == RoleType.DEPT_ADMIN:
                        user_dept = db_user.department.value
                    # Regular user: get departments from their group folder access
                    else:
                        result = await db.execute(
                            select(Folder.department)
                            .join(GroupFolder, GroupFolder.folder_id == Folder.id)
                            .join(UserGroup, UserGroup.group_id == GroupFolder.group_id)
                            .where(UserGroup.user_id == db_user.id)
                            .distinct()
                        )
                        depts = [row[0] for row in result.all()]
                        if depts:
                            # For now, pass first accessible department
                            # TODO: support multi-department queries
                            user_dept = depts[0]
                        else:
                            # User has no group access to any folder
                            user_dept = "__none__"

        except Exception as e:
            print(f"[BOT] Error looking up user: {e}")

        # --- Handle Adaptive Card submissions ---
        if turn_context.activity.value:
            await turn_context.send_activity(
                "Card action received. Please type your question directly."
            )
            return

        # --- Extract the question text ---
        text = turn_context.activity.text or ""
        if turn_context.activity.entities:
            text = TurnContext.remove_recipient_mention(turn_context.activity).strip()

        if not text:
            await turn_context.send_activity("Please type a question to get started.")
            return

        # --- Greeting / Help ---
        lower_text = text.lower().strip()
        if lower_text in ["hello", "hi", "hey", "help", "?", "start"]:
            card = build_welcome_card()
            attachment = create_attachment(card)
            await turn_context.send_activity(Activity(attachments=[attachment]))
            return

        # --- Server-side user gating ---
        # Trusted sender mode (no gating): any Teams user may query.
        # When the user is registered we apply department filters + admin
        # access control; otherwise they search all departments with no cap.
        user_id_for_memory = None
        if db_user is None:
            user_dept = None  # search all departments
        elif user_dept == "__none__":
            # Registered user without group folder access: in trusted mode
            # let them search all departments instead of blocking.
            user_dept = None
        else:
            # Admin-managed access control (enable/disable) + daily token limit.
            try:
                from app.admin.access import check_user_can_use_chat
                from app.database import AsyncSessionLocal

                async with AsyncSessionLocal() as db:
                    # Reload user within this session so relationships/columns are fresh
                    db_user = await db.get(User, db_user.id)
                    allowed, reason = await check_user_can_use_chat(db, db_user)
                if not allowed:
                    card = build_error_card(reason)
                    attachment = create_attachment(card)
                    await turn_context.send_activity(Activity(attachments=[attachment]))
                    return
                user_id_for_memory = str(db_user.id)
            except Exception as e:
                print(f"[BOT] Access check failed: {e}")
                card = build_error_card(
                    "I could not verify your access right now. Please try again later."
                )
                attachment = create_attachment(card)
                await turn_context.send_activity(Activity(attachments=[attachment]))
                return

        # --- Query the RAG pipeline ---
        try:
            from app.rag.agent import query_rag
            from app.memory.service import get_memory_service
            from app.config import get_settings

            settings = get_settings()
            memory = get_memory_service(
                ttl_hours=settings.CONVERSATION_TTL_HOURS,
                max_messages=settings.MEMORY_MAX_MESSAGES,
            )

            # Use Teams conversation ID for memory (map to stable UUID)
            # Persist Teams identity on the conversation so token usage can be
            # attributed per-Teams-user even when the user is NOT registered
            # in the console's `users` table (user_id = null).
            teams_conv_id = turn_context.activity.conversation.id
            conv_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"teams-{teams_conv_id}"))
            conv = await memory.get_or_create_conversation(
                conversation_id=conv_uuid,
                user_id=user_id_for_memory,
                source="teams",
                teams_aad_id=teams_aad_id,
                teams_email=teams_email,
                teams_name=teams_name,
                teams_channel_id=teams_conv_id,
            )
            conv_id = str(conv.id)

            history = await memory.get_history(conv_id)

            await memory.add_message(
                conversation_id=conv_id,
                role="user",
                content=text,
            )

            # user_dept=None means super_admin (search all)
            # user_dept="hr" means search HR only
            result = await query_rag(
                question=text,
                department=user_dept,
                chat_history=history,
                include_usage=True,
            )

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            chunks_count = result.get("chunks_retrieved", 0)
            usage = result.get("usage", {})

            await memory.add_message(
                conversation_id=conv_id,
                role="assistant",
                content=answer,
                sources=sources,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                model_used=settings.OPENAI_MODEL,
            )

            if chunks_count == 0:
                card = build_no_results_card()
            else:
                card = build_answer_card(answer, sources)

            attachment = create_attachment(card)
            await turn_context.send_activity(Activity(attachments=[attachment]))

        except Exception as e:
            print(f"[BOT] RAG query failed: {e}")
            card = build_error_card(
                "I encountered an error while searching the documents. Please try again later."
            )
            attachment = create_attachment(card)
            await turn_context.send_activity(Activity(attachments=[attachment]))

    async def on_teams_members_added(
        self, teams_members_added: list, team_info, turn_context: TurnContext,
    ):
        for member in teams_members_added:
            if member.id != turn_context.activity.recipient.id:
                card = build_welcome_card()
                attachment = create_attachment(card)
                await turn_context.send_activity(Activity(attachments=[attachment]))


bot = PolicyBot()


@router.post("/api/messages")
async def messages(request: Request):
    """Teams Bot Framework message endpoint."""
    from fastapi.responses import JSONResponse

    body = await request.body()
    auth_header = request.headers.get("Authorization", "")

    if not body:
        return JSONResponse(status_code=400, content={"error": "Empty request body"})

    try:
        activity = Activity().deserialize(json.loads(body))
    except (json.JSONDecodeError, TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "Invalid activity payload"})

    if adapter is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": "Teams bot is not configured (MicrosoftAppId/MicrosoftAppPassword missing)."},
        )

    await adapter.process_activity(auth_header, activity, bot.on_turn)

    return Response(status_code=200)

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
                aad_id = addl.get("aadObjectId") or addl.get("aad_object_id") or addl.get("objectId")
            # properties dict
            props = getattr(user, "properties", None)
            if not aad_id and isinstance(props, dict):
                aad_id = props.get("aadObjectId") or props.get("aad_object_id") or props.get("objectId")
            # Try direct dict access on the object itself (some SDKs store raw in __dict__)
            if not aad_id and hasattr(user, "__dict__"):
                d = user.__dict__
                aad_id = d.get("aad_object_id") or d.get("aadObjectId") or d.get("objectId")
        # Fallback: raw activity JSON (activity.channel_data / value)
        if not aad_id:
            try:
                raw = activity.as_dict() if hasattr(activity, "as_dict") else {}
                # from.aadObjectId in raw JSON
                from_raw = raw.get("from") or {}
                aad_id = from_raw.get("aadObjectId") or from_raw.get("aad_object_id") or from_raw.get("objectId")
                # Teams sometimes nests in channelData or tenant
                if not aad_id and isinstance(raw.get("channelData"), dict):
                    cd = raw["channelData"]
                    # Some payloads put user AAD in channelData.tenant or extra
                    aad_id = cd.get("aadObjectId") or cd.get("aad_object_id")
            except Exception:
                pass

        # Normalize: strip whitespace; keep original case for logging but clean for lookup
        if aad_id and isinstance(aad_id, str):
            aad_id = aad_id.strip()

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
            # Raw fallback
            if not email:
                try:
                    raw = activity.as_dict() if hasattr(activity, "as_dict") else {}
                    from_raw = raw.get("from") or {}
                    email = from_raw.get("email") or from_raw.get("userPrincipalName") or ""
                except Exception:
                    pass
            if email and isinstance(email, str):
                email = email.strip()

        # --- Channel / conversation id for debugging ---
        channel_id = ""
        try:
            cd = getattr(activity, "channel_data", None)
            if isinstance(cd, dict):
                tenant = cd.get("tenant") or {}
                if isinstance(tenant, dict):
                    channel_id = tenant.get("id") or ""
            elif hasattr(activity, "as_dict"):
                raw = activity.as_dict() or {}
                channel_id = (raw.get("channelData") or {}).get("tenant", {}).get("id", "")
        except Exception:
            pass

        # Also capture raw from payload for deep debugging
        raw_from_dump = {}
        try:
            raw = activity.as_dict() if hasattr(activity, "as_dict") else {}
            raw_from_dump = raw.get("from") or {}
        except Exception:
            pass

        raw_debug = {
            "from_id": getattr(user, "id", None) if user else None,
            "from_name": name,
            "aad_object_id": aad_id,
            "aad_object_id_present": bool(aad_id),
            "email": email,
            "email_present": bool(email),
            "channelId": getattr(activity, "channel_id", None),
            "conversation_id": getattr(getattr(activity, "conversation", None), "id", None) if getattr(activity, "conversation", None) else None,
            "raw_from": raw_from_dump,
        }
        return aad_id, email, name, channel_id, raw_debug

    async def on_message_activity(self, turn_context: TurnContext):
        # --- Always save proactive ref so broadcast can reach this user later (fire-and-forget) ---
        try:
            import asyncio as _asyncio
            from app.bot.proactive import save_proactive_ref
            _asyncio.create_task(save_proactive_ref(turn_context))
        except Exception as e:
            print(f"[PROACTIVE] hook failed: {e}")

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
                # --- Diagnostic: log what aad_ids we have in DB (first 5) when lookup fails ---
                # Normalize AAD for comparison: case-insensitive, trimmed
                clean_aad = (aad_id or "").strip().lower() if aad_id else None

                # Find user by AAD object ID FIRST (most reliable — Teams always sends this)
                if clean_aad:
                    from sqlalchemy import func as _func
                    # Case-insensitive exact match; handles GUID case variance
                    result = await db.execute(select(User).where(_func.lower(User.aad_object_id) == clean_aad))
                    db_user = result.scalar_one_or_none()
                    if db_user:
                        print(f"[BOT] Matched Teams user by AAD ID {aad_id} (clean={clean_aad}) -> {db_user.email}")
                    else:
                        print(f"[BOT] No DB user found for AAD ID '{aad_id}' (clean='{clean_aad}')")
                        # Helpful diagnostic: show what AADs ARE in DB
                        try:
                            all_aads = (await db.execute(select(User.email, User.aad_object_id).where(User.aad_object_id.isnot(None)))).all()
                            if all_aads:
                                sample = ", ".join([f"{e}=>{a}" for e, a in all_aads[:5]])
                                print(f"[BOT] DB has {len(all_aads)} users with AAD. Sample: {sample}")
                            else:
                                print("[BOT] DB has 0 users with aad_object_id — did you save the AAD in the user record?")
                            # Also show raw dump to spot field name mismatch
                            print(f"[BOT] Full sender dump for manual compare: aad='{aad_id}' email='{user_email}' name='{user_name}' from_id='{_dbg.get('from_id')}' raw_from={_dbg.get('raw_from')}")
                        except Exception as e:
                            print(f"[BOT] Diagnostic listing failed: {e}")

                # Fallback: lookup by email / UPN if AAD did not match
                if not db_user and user_email:
                    clean_email = user_email.strip().lower()
                    from sqlalchemy import func as _func2
                    result = await db.execute(select(User).where(_func2.lower(User.email) == clean_email))
                    db_user = result.scalar_one_or_none()
                    if db_user:
                        print(f"[BOT] Matched Teams user by email {user_email} (clean={clean_email}) -> AAD {db_user.aad_object_id}")
                        # Opportunistically backfill AAD ID if DB row has none
                        if aad_id and not db_user.aad_object_id:
                            try:
                                db_user.aad_object_id = aad_id.strip()
                                await db.commit()
                                print(f"[BOT] Backfilled AAD ID {aad_id} for {user_email}")
                            except Exception as e:
                                print(f"[BOT] Failed to backfill AAD ID: {e}")
                                await db.rollback()
                    else:
                        print(f"[BOT] No DB user found for email '{user_email}' (clean='{clean_email}')")

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

        # --- Debug: whoami (helps user copy the exact AAD to paste into Admin → Users) ---
        lower_text = text.lower().strip()
        if lower_text in ["whoami", "who am i", "my id", "my aad", "debug", "show my id", "myid", "my aad id"]:
            # Show what Teams actually sent so user can compare with DB value
            matched = f"Matched DB user: {db_user.email} ({db_user.full_name})" if db_user else "Not matched — no user in DB has this AAD. Add it via Admin → Users → Edit."
            info = (
                f"**Your Teams identity (what the bot sees):**\n\n"
                f"- **Name:** {teams_name or '(empty)'}\n"
                f"- **Email/UPN:** {teams_email or '(empty — Teams does not send email without SSO, this is normal)'}\n"
                f"- **AAD Object ID:** `{teams_aad_id or '(empty — bot did not receive an AAD ID!) Please check Teams manifest permissions.'}`\n"
                f"- **Teams From ID:** `{_dbg.get('from_id') or ''}`\n\n"
                f"{matched}\n\n"
                f"**What to do:** Copy the **AAD Object ID** above and paste it into **Admin → Users → Edit → Teams AAD Object ID** for your user. "
                f"Next message will be recognized. If AAD is empty, the Teams app manifest is missing the identity permission — contact admin."
            )
            await turn_context.send_activity(info)
            # Also persist a card for copy-paste
            print(f"[BOT] whoami requested — replied with identity dump for {teams_name}: aad={teams_aad_id} email={teams_email} matched={bool(db_user)}")
            return
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
        # Save ref for newly added members too (so broadcast reaches them even before first message)
        try:
            import asyncio as _asyncio
            from app.bot.proactive import save_proactive_ref
            _asyncio.create_task(save_proactive_ref(turn_context))
        except Exception:
            pass
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

    # --- RAW PAYLOAD LOG for debugging AAD mismatch ---
    # This is the ONLY place we see the exact JSON Teams sent, before SDK deserialization.
    # Helps diagnose why a user's AAD they pasted doesn't match what Teams sends.
    try:
        raw_json = json.loads(body)
        raw_from = raw_json.get("from") or {}
        print(f"[BOT] Raw Teams payload from={raw_from} channelData={raw_json.get('channelData')} text={raw_json.get('text','')[:80]!r}")
    except Exception as e:
        print(f"[BOT] Raw payload logging failed: {e}")

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

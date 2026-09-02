"""
Teams Bot Handler
Processes messages from Teams, identifies user via Teams identity,
checks group-based access, queries RAG pipeline, responds with Adaptive Cards.
"""

import json
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

    async def on_message_activity(self, turn_context: TurnContext):
        # --- Extract user identity from Teams activity ---
        user = turn_context.activity.from_property
        user_email = ""
        user_dept = None
        db_user = None

        if user:
            user_email = getattr(user, "email", "") or ""
            if not user_email:
                user_email = getattr(user, "user_principal_name", "") or ""

        # --- Look up user + group access from database ---
        try:
            from app.database import AsyncSessionLocal
            from app.models.user import User, RoleType
            from app.models.group import UserGroup, GroupFolder
            from app.models.folder import Folder
            from sqlalchemy import select

            async with AsyncSessionLocal() as db:
                # Find user by email or AAD object ID
                if user_email:
                    result = await db.execute(select(User).where(User.email == user_email))
                    db_user = result.scalar_one_or_none()

                if not db_user:
                    aad_id = getattr(user, "aad_object_id", None) if user else None
                    if aad_id:
                        result = await db.execute(select(User).where(User.aad_object_id == aad_id))
                        db_user = result.scalar_one_or_none()

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

            # Use Teams conversation ID for memory
            teams_conv_id = turn_context.activity.conversation.id
            conv = await memory.get_or_create_conversation(
                conversation_id=teams_conv_id,
                user_id=user_id_for_memory,
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

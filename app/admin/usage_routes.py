"""
Admin Usage Analytics Routes
Token-usage tracking showcase for the EthosAI admin portal.

Super admins see usage across all users; dept admins see only their department.
All queries aggregate over the LLM assistant-messages that carry token counts.
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User, RoleType
from app.models.conversation import Message, MessageRole, Conversation
from app.auth.dependencies import require_admin

router = APIRouter()


async def _scoped_user_filter(current_user: User, query):
    """Constrain an aggregate query to the admin's department (unless super admin)."""
    if current_user.role != RoleType.SUPER_ADMIN:
        return query.where(User.department == current_user.department.value)
    return query


def _aggregate_rows(rows):
    """Build the per-user list + totals from raw aggregate rows."""
    users = []
    total_prompt = 0
    total_completion = 0
    total_questions = 0
    for row in rows:
        prompt = int(row.prompt_tokens or 0)
        completion = int(row.completion_tokens or 0)
        total_prompt += prompt
        total_completion += completion
        total_questions += int(row.question_count or 0)
        users.append(
            {
                "user_id": str(row.id),
                "email": row.email,
                "full_name": row.full_name,
                "role": row.role,
                "department": row.department,
                "question_count": int(row.question_count or 0),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        )

    users.sort(key=lambda u: u["total_tokens"], reverse=True)
    return {
        "total_users_with_usage": len(users),
        "total_questions": total_questions,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "top_user": users[0] if users else None,
        "users": users,
    }


@router.get("/usage/per-user")
async def usage_per_user(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate token usage grouped by user, ordered by total tokens (desc).
    For the admin dashboard's per-user usage showcase.
    """
    base = (
        select(
            User.id,
            User.email,
            User.full_name,
            User.role,
            User.department,
            func.count(Message.id).label("question_count"),
            func.coalesce(func.sum(Message.tokens_in), 0).label("prompt_tokens"),
            func.coalesce(func.sum(Message.tokens_out), 0).label("completion_tokens"),
        )
        .select_from(User)
        .join(Conversation, Conversation.user_id == User.id)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Message.role == MessageRole.ASSISTANT.name)
        .group_by(
            User.id,
            User.email,
            User.full_name,
            User.role,
            User.department,
        )
        .order_by(func.sum(Message.tokens_in).desc())
    )
    base = await _scoped_user_filter(current_user, base)
    rows = (await db.execute(base)).all()
    return _aggregate_rows(rows)


@router.get("/usage/timeseries")
async def usage_timeseries(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    days: int = Query(default=14, ge=1, le=90),
    month: str = Query(default=None, description="YYYY-MM to fetch that calendar month (overrides days)"),
):
    """
    Daily token usage over the last N days, for charting.
    Returns one entry per day with prompt/completion/total tokens and request count.
    If month=YYYY-MM is given, returns that calendar month's daily data.
    """
    import calendar as _cal

    if month:
        try:
            y, m = map(int, month.split("-"))
            _, last = _cal.monthrange(y, m)
            since = datetime(y, m, 1)
            days_in_month = last
            # Use month's range instead of days param
            until = datetime(y, m, last, 23, 59, 59, 999999)
        except Exception:
            from fastapi import HTTPException as _HTTP
            raise _HTTP(status_code=400, detail="Invalid month format, use YYYY-MM")
        # Build query for that month range
        daily = (
            select(
                func.date(Message.created_at).label("day"),
                func.count(Message.id).label("requests"),
                func.coalesce(func.sum(Message.tokens_in), 0).label("prompt_tokens"),
                func.coalesce(func.sum(Message.tokens_out), 0).label("completion_tokens"),
            )
            .select_from(Conversation)
            .join(Message, Message.conversation_id == Conversation.id)
            .join(User, User.id == Conversation.user_id)
            .where(
                Message.role == MessageRole.ASSISTANT.name,
                Message.created_at >= since,
                Message.created_at <= until,
            )
            .group_by(func.date(Message.created_at))
            .order_by(func.date(Message.created_at))
        )
        daily = await _scoped_user_filter(current_user, daily)
        rows = (await db.execute(daily)).all()
        by_day = {}
        for row in rows:
            by_day[str(row.day)] = {
                "requests": int(row.requests or 0),
                "prompt_tokens": int(row.prompt_tokens or 0),
                "completion_tokens": int(row.completion_tokens or 0),
                "total_tokens": int((row.prompt_tokens or 0) + (row.completion_tokens or 0)),
            }
        series = []
        for d in range(1, days_in_month + 1):
            day = datetime(y, m, d).date()
            key = str(day)
            entry = by_day.get(key, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            series.append({"date": key, **entry})
        return {"series": series, "month": month, "days_in_month": days_in_month}

    since = datetime.utcnow() - timedelta(days=days - 1)
    since = since.replace(hour=0, minute=0, second=0, microsecond=0)

    daily = (
        select(
            func.date(Message.created_at).label("day"),
            func.count(Message.id).label("requests"),
            func.coalesce(func.sum(Message.tokens_in), 0).label("prompt_tokens"),
            func.coalesce(func.sum(Message.tokens_out), 0).label("completion_tokens"),
        )
        .select_from(Conversation)
        .join(Message, Message.conversation_id == Conversation.id)
        .join(User, User.id == Conversation.user_id)
        .where(
            Message.role == MessageRole.ASSISTANT.name,
            Message.created_at >= since,
        )
        .group_by(func.date(Message.created_at))
        .order_by(func.date(Message.created_at))
    )
    daily = await _scoped_user_filter(current_user, daily)

    rows = (await db.execute(daily)).all()
    by_day = {}
    for row in rows:
        by_day[str(row.day)] = {
            "requests": int(row.requests or 0),
            "prompt_tokens": int(row.prompt_tokens or 0),
            "completion_tokens": int(row.completion_tokens or 0),
            "total_tokens": int((row.prompt_tokens or 0) + (row.completion_tokens or 0)),
        }

    # Fill the full date range so charts are continuous
    series = []
    for i in range(days):
        day = (since + timedelta(days=i)).date()
        key = str(day)
        entry = by_day.get(
            key,
            {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        series.append({"date": key, **entry})

    return {"series": series}


@router.get("/usage/user-access")
async def user_access(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Access-control management view: every user in scope with their chatbot
    access toggle, daily token limit, and today's usage / remaining quota.
    Admins use PATCH /api/admin/users/{id} to change access.
    """
    from datetime import datetime as _dt

    # Today's usage per user
    start_of_day = _dt.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_q = (
        select(
            Conversation.user_id.label("uid"),
            (
                func.coalesce(func.sum(Message.tokens_in), 0)
                + func.coalesce(func.sum(Message.tokens_out), 0)
            ).label("used_today"),
            func.count(Message.id).label("requests_today"),
        )
        .select_from(Conversation)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(
            Message.role == MessageRole.ASSISTANT.name,
            Message.created_at >= start_of_day,
        )
        .group_by(Conversation.user_id)
    )
    today_rows = (await db.execute(today_q)).all()
    today = {str(r.uid): {"used": int(r.used_today or 0), "requests": int(r.requests_today or 0)} for r in today_rows}

    # All users in scope
    user_q = select(User).order_by(User.created_at.desc())
    user_q = await _scoped_user_filter(current_user, user_q)
    users = (await db.execute(user_q)).scalars().all()

    result = []
    for u in users:
        t = today.get(str(u.id), {"used": 0, "requests": 0})
        limit = u.daily_token_limit or 0
        remaining = max(0, limit - t["used"]) if limit > 0 else None
        result.append(
            {
                "user_id": str(u.id),
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role.value if hasattr(u.role, "value") else u.role,
                "department": u.department.value if hasattr(u.department, "value") else u.department,
                "is_active": u.is_active,
                "chat_access_enabled": u.chat_access_enabled,
                "daily_token_limit": limit,
                "used_today": t["used"],
                "requests_today": t["requests"],
                "remaining_today": remaining,
                "limit_enforced": limit > 0,
            }
        )

    # Most recently active / highest usage first
    result.sort(key=lambda x: x["used_today"], reverse=True)
    return {"users": result}

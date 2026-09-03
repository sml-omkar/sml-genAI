"""
Admin Usage Analytics Routes
Token-usage tracking showcase for the EthosAI admin portal.

Super admins see usage across all users; dept admins see only their department.
All queries aggregate over the LLM assistant-messages that carry token counts.

Now also tracks Teams (/api/messages) traffic per-Teams-identity, even when the
Teams sender is NOT yet registered as a `users` row (user_id = NULL). Teams
identity is stored on `conversations.teams_aad_id / teams_email / teams_name`.
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, case
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
    source: str = Query(default="all", description="Filter by source: all|web|teams"),
):
    """
    Daily token usage over the last N days, for charting.
    Returns one entry per day with prompt/completion/total tokens and request count.
    If month=YYYY-MM is given, returns that calendar month's daily data.

    `source` controls which conversations are counted:
      all   = web (/api/chat) + teams (/api/messages)
      web   = only web console traffic
      teams = only Teams bot traffic

    Now counts ALL conversations (including Teams unregistered where user_id IS NULL)
    by NOT inner-joining User. When dept-scoped (dept_admin), only registered-user
    traffic for that department is counted, plus Teams traffic that was linked to
    a user in that department.
    """
    import calendar as _cal

    def _apply_source_filter(q):
        if source == "web":
            return q.where(Conversation.source == "web")
        if source == "teams":
            return q.where(Conversation.source == "teams")
        return q

    def _apply_dept_filter_for_timeseries(q):
        # Dept admin should only see usage from their department.
        # For registered users: Conversation.user_id -> User.department
        # For unregistered Teams: Conversation.teams_* has no department, so it
        # is NOT counted for dept_admin (correct — they'd need to register the user).
        # We achieve this by left-joining User and filtering where user dept matches
        # OR where the row is unregistered but we still want to hide it.
        if current_user.role == RoleType.SUPER_ADMIN:
            return q
        # Only keep rows where the linked user is in the admin's department.
        # Unregistered Teams rows have no user, so they are excluded for dept_admin.
        return q.where(User.department == current_user.department.value)

    if month:
        try:
            y, m = map(int, month.split("-"))
            _, last = _cal.monthrange(y, m)
            since = datetime(y, m, 1)
            days_in_month = last
            until = datetime(y, m, last, 23, 59, 59, 999999)
        except Exception:
            from fastapi import HTTPException as _HTTP
            raise _HTTP(status_code=400, detail="Invalid month format, use YYYY-MM")
        daily = (
            select(
                func.date(Message.created_at).label("day"),
                func.count(Message.id).label("requests"),
                func.coalesce(func.sum(Message.tokens_in), 0).label("prompt_tokens"),
                func.coalesce(func.sum(Message.tokens_out), 0).label("completion_tokens"),
            )
            .select_from(Conversation)
            .join(Message, Message.conversation_id == Conversation.id)
            .outerjoin(User, User.id == Conversation.user_id)
            .where(
                Message.role == MessageRole.ASSISTANT.name,
                Message.created_at >= since,
                Message.created_at <= until,
            )
            .group_by(func.date(Message.created_at))
            .order_by(func.date(Message.created_at))
        )
        daily = _apply_source_filter(daily)
        daily = _apply_dept_filter_for_timeseries(daily)
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
        return {"series": series, "month": month, "days_in_month": days_in_month, "source": source}

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
        .outerjoin(User, User.id == Conversation.user_id)
        .where(
            Message.role == MessageRole.ASSISTANT.name,
            Message.created_at >= since,
        )
        .group_by(func.date(Message.created_at))
        .order_by(func.date(Message.created_at))
    )
    daily = _apply_source_filter(daily)
    daily = _apply_dept_filter_for_timeseries(daily)

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
    for i in range(days):
        day = (since + timedelta(days=i)).date()
        key = str(day)
        entry = by_day.get(
            key,
            {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        series.append({"date": key, **entry})

    return {"series": series, "source": source}


@router.get("/usage/overview")
async def usage_overview(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Totals split by source: web vs teams vs all.
    Useful to see how much Teams is driving vs console.
    """
    base = (
        select(
            Conversation.source.label("source"),
            func.count(Message.id).label("requests"),
            func.coalesce(func.sum(Message.tokens_in), 0).label("prompt_tokens"),
            func.coalesce(func.sum(Message.tokens_out), 0).label("completion_tokens"),
        )
        .select_from(Conversation)
        .join(Message, Message.conversation_id == Conversation.id)
        .outerjoin(User, User.id == Conversation.user_id)
        .where(Message.role == MessageRole.ASSISTANT.name)
        .group_by(Conversation.source)
    )
    # Dept admin: filter to their dept's registered-user traffic only
    if current_user.role != RoleType.SUPER_ADMIN:
        base = base.where(User.department == current_user.department.value)

    rows = (await db.execute(base)).all()
    by_source = {}
    total_requests = 0
    total_prompt = 0
    total_completion = 0
    for r in rows:
        src = r.source or "unknown"
        req = int(r.requests or 0)
        pt = int(r.prompt_tokens or 0)
        ct = int(r.completion_tokens or 0)
        by_source[src] = {"requests": req, "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        total_requests += req
        total_prompt += pt
        total_completion += ct

    # Also count distinct Teams identities vs registered users
    teams_distinct = (await db.execute(
        select(func.count(func.distinct(Conversation.teams_aad_id)))
        .select_from(Conversation)
        .where(Conversation.source == "teams")
        .where(Conversation.teams_aad_id.isnot(None))
    )).scalar() or 0

    registered_with_usage = (await db.execute(
        select(func.count(func.distinct(Conversation.user_id)))
        .select_from(Conversation)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Message.role == MessageRole.ASSISTANT.name)
        .where(Conversation.user_id.isnot(None))
    )).scalar() or 0

    return {
        "by_source": by_source,
        "total": {
            "requests": total_requests,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        },
        "distinct_teams_identities": int(teams_distinct),
        "distinct_registered_users_with_usage": int(registered_with_usage),
    }


@router.get("/usage/teams")
async def usage_teams(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Per-Teams-identity token usage (segregated by AAD ID / email / name).

    This is the Teams counterpart to /usage/per-user. It groups by the
    Teams identity stored on conversations (teams_aad_id), so even
    unregistered Teams senders are visible and their token consumption
    can be audited. If the Teams sender WAS matched to a `users` row,
    the response also includes the linked `user_id` and `email`.

    Super admin sees all Teams identities. Dept admin sees only those
    Teams identities that were linked to a user in their department
    (unlinked Teams traffic is hidden for dept_admin, same as timeseries).
    """
    # Per-Teams-identity aggregates (AAD ID is the key — Teams always sends it)
    base = (
        select(
            Conversation.teams_aad_id.label("teams_aad_id"),
            Conversation.teams_email.label("teams_email"),
            Conversation.teams_name.label("teams_name"),
            func.max(Conversation.user_id).label("linked_user_id"),
            func.count(Message.id).label("requests"),
            func.coalesce(func.sum(Message.tokens_in), 0).label("prompt_tokens"),
            func.coalesce(func.sum(Message.tokens_out), 0).label("completion_tokens"),
        )
        .select_from(Conversation)
        .join(Message, Message.conversation_id == Conversation.id)
        .outerjoin(User, User.id == Conversation.user_id)
        .where(Message.role == MessageRole.ASSISTANT.name)
        .where(Conversation.source == "teams")
        .group_by(Conversation.teams_aad_id, Conversation.teams_email, Conversation.teams_name)
        .order_by(func.sum(Message.tokens_in + Message.tokens_out).desc())
    )
    if current_user.role != RoleType.SUPER_ADMIN:
        # Dept admin: only show Teams conversations that were linked to a user in their dept
        # Unlinked Teams rows (user_id NULL) are excluded.
        base = base.where(User.department == current_user.department.value)

    rows = (await db.execute(base)).all()

    # Resolve linked user details for display
    teams = []
    total_prompt = 0
    total_completion = 0
    total_requests = 0
    for r in rows:
        pt = int(r.prompt_tokens or 0)
        ct = int(r.completion_tokens or 0)
        total_prompt += pt
        total_completion += ct
        total_requests += int(r.requests or 0)
        linked_email = None
        linked_name = None
        linked_dept = None
        if r.linked_user_id:
            # Fetch linked user details (one extra query per distinct aad would be N+1, but
            # Teams distinct count is typically < 1k; acceptable. Optimize later if needed.)
            ures = await db.execute(select(User).where(User.id == r.linked_user_id))
            u = ures.scalar_one_or_none()
            if u:
                linked_email = u.email
                linked_name = u.full_name
                linked_dept = u.department.value if hasattr(u.department, "value") else str(u.department)

        teams.append({
            "teams_aad_id": r.teams_aad_id,
            "teams_email": r.teams_email,
            "teams_name": r.teams_name,
            "linked_user_id": str(r.linked_user_id) if r.linked_user_id else None,
            "linked_email": linked_email,
            "linked_name": linked_name,
            "linked_department": linked_dept,
            "is_registered": r.linked_user_id is not None,
            "requests": int(r.requests or 0),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        })

    # Also include a bucket for Teams conversations where AAD ID was missing (legacy)
    # These will have teams_aad_id IS NULL but source == teams.
    legacy = (await db.execute(
        select(
            func.count(Message.id).label("requests"),
            func.coalesce(func.sum(Message.tokens_in), 0).label("prompt_tokens"),
            func.coalesce(func.sum(Message.tokens_out), 0).label("completion_tokens"),
        )
        .select_from(Conversation)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Message.role == MessageRole.ASSISTANT.name)
        .where(Conversation.source == "teams")
        .where(Conversation.teams_aad_id.is_(None))
    )).first()

    legacy_bucket = None
    if legacy and int(legacy.requests or 0) > 0:
        pt = int(legacy.prompt_tokens or 0)
        ct = int(legacy.completion_tokens or 0)
        legacy_bucket = {
            "requests": int(legacy.requests or 0),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        }

    return {
        "total_teams_identities": len(teams),
        "total_requests": total_requests + (legacy_bucket["requests"] if legacy_bucket else 0),
        "total_prompt_tokens": total_prompt + (legacy_bucket["prompt_tokens"] if legacy_bucket else 0),
        "total_completion_tokens": total_completion + (legacy_bucket["completion_tokens"] if legacy_bucket else 0),
        "total_tokens": total_prompt + total_completion + (legacy_bucket["total_tokens"] if legacy_bucket else 0),
        "teams": teams,
        "legacy_unidentified": legacy_bucket,
    }


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

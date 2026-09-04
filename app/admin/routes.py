"""
Admin Routes
Dashboard stats and user management endpoints.
NXSS AI admin (super_admin) sees everything.
Dept admins see only their department.
"""

import csv
import io
import re
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models.user import User, RoleType
from app.models.folder import Folder
from app.models.document import Document, ProcessingStatus
from app.models.group import Group, UserGroup, GroupFolder
from app.schemas.user import (
    UserResponse,
    UserListResponse,
    UserCreateRequest,
    UserUpdateRequest,
    DashboardStats,
)
from app.auth.dependencies import get_current_user, require_role, require_super_admin, require_admin
from app.auth.jwt import hash_password
from app.rbac.permissions import can_manage_users

router = APIRouter()


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get dashboard statistics.
    Super admins see all stats; dept admins see only their department.
    """
    is_super_admin = current_user.role == RoleType.SUPER_ADMIN
    dept_filter = None if is_super_admin else current_user.department.value

    folder_query = select(func.count(Folder.id))
    if dept_filter:
        folder_query = folder_query.where(Folder.department == dept_filter)
    total_folders = (await db.execute(folder_query)).scalar() or 0

    doc_query = select(func.count(Document.id))
    if dept_filter:
        doc_query = doc_query.join(Folder).where(Folder.department == dept_filter)
    total_documents = (await db.execute(doc_query)).scalar() or 0

    chunk_query = select(func.coalesce(func.sum(Document.chunk_count), 0))
    if dept_filter:
        chunk_query = chunk_query.join(Folder).where(Folder.department == dept_filter)
    total_chunks = (await db.execute(chunk_query)).scalar() or 0

    user_query = select(func.count(User.id))
    if dept_filter:
        user_query = user_query.where(User.department == dept_filter)
    total_users = (await db.execute(user_query)).scalar() or 0

    status_query = select(
        Document.status, func.count(Document.id)
    ).group_by(Document.status)
    if dept_filter:
        status_query = status_query.join(Folder).where(Folder.department == dept_filter)
    status_result = await db.execute(status_query)
    docs_by_status = {row[0].value: row[1] for row in status_result.all()}

    dept_query = select(
        Folder.department, func.count(Document.id)
    ).join(Document, isouter=True).group_by(Folder.department)
    dept_result = await db.execute(dept_query)
    docs_by_dept = {row[0]: row[1] or 0 for row in dept_result.all()}

    # Group stats (super admin only)
    total_groups = 0
    if is_super_admin:
        total_groups = (await db.execute(select(func.count(Group.id)))).scalar() or 0

    return DashboardStats(
        total_folders=total_folders,
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_users=total_users,
        total_groups=total_groups,
        documents_by_status=docs_by_status,
        documents_by_department=docs_by_dept,
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List users. Super admins see all; dept admins see their department only."""
    query = select(User)
    if current_user.role != RoleType.SUPER_ADMIN:
        query = query.where(User.department == current_user.department.value)
    query = query.order_by(User.created_at.desc())
    result = await db.execute(query)
    users = result.scalars().all()

    return UserListResponse(
        users=[UserResponse.model_validate(u) for u in users],
        total=len(users),
    )


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new user. Super admins create anyone; dept admins create users only."""
    target_role = RoleType(request.role)
    if not can_manage_users(current_user.role, target_role):
        raise HTTPException(status_code=403, detail="You cannot create users with this role.")

    if current_user.role != RoleType.SUPER_ADMIN:
        if request.department != current_user.department.value:
            raise HTTPException(status_code=403, detail="You can only create users in your department.")

    existing = await db.execute(select(User).where(User.email == request.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="A user with this email already exists.")

    if request.aad_object_id:
        existing_aad = await db.execute(select(User).where(User.aad_object_id == request.aad_object_id))
        if existing_aad.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="A user with this Teams AAD Object ID already exists.")

    new_user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        full_name=request.full_name,
        department=request.department,
        role=target_role,
        chat_access_enabled=request.chat_access_enabled,
        daily_token_limit=max(0, request.daily_token_limit or 0),
        aad_object_id=request.aad_object_id or None,
    )
    db.add(new_user)
    await db.flush()
    await db.refresh(new_user)

    return UserResponse.model_validate(new_user)


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update user fields."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if current_user.role != RoleType.SUPER_ADMIN:
        if user.department != current_user.department.value:
            raise HTTPException(status_code=403, detail="Access denied.")

    if request.full_name is not None:
        user.full_name = request.full_name
    if request.department is not None:
        user.department = request.department
    if request.role is not None:
        target_role = RoleType(request.role)
        if not can_manage_users(current_user.role, target_role):
            raise HTTPException(status_code=403, detail="Cannot assign this role.")
        user.role = target_role
    if request.is_active is not None:
        user.is_active = request.is_active
    if request.chat_access_enabled is not None:
        user.chat_access_enabled = request.chat_access_enabled
    if request.daily_token_limit is not None:
        if request.daily_token_limit < 0:
            raise HTTPException(status_code=422, detail="Daily token limit cannot be negative.")
        user.daily_token_limit = request.daily_token_limit
    if request.password is not None:
        user.hashed_password = hash_password(request.password)
    if request.aad_object_id is not None:
        user.aad_object_id = request.aad_object_id or None

    await db.flush()
    await db.refresh(user)
    return UserResponse.model_validate(user)


@router.get("/users/bulk-template")
async def download_bulk_template(
    current_user: User = Depends(require_admin),
):
    """Download CSV template for bulk user upload."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["email", "full_name", "password", "department", "role", "daily_token_limit", "chat_access_enabled", "aad_object_id"])
    writer.writerow(["alice@company.com", "Alice Smith", "TempPass123", "it", "user", "5000", "true", "9f8c1d2e-0000-1111-2222-333344445555"])
    writer.writerow(["bob@company.com", "Bob Jones", "TempPass123", "hr", "dept_admin", "0", "true", ""])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nxssai_users_template.csv"},
    )


@router.post("/users/bulk-upload")
async def bulk_upload_users(
    file: UploadFile = File(...),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Bulk create users from a CSV file.
    Expected columns: email, full_name, password, department, role, daily_token_limit, chat_access_enabled, aad_object_id
    - department: hr | it | finance (or slug from departments table)
    - role: user | dept_admin | super_admin
    - daily_token_limit: integer, 0 = unlimited (default 0)
    - chat_access_enabled: true/false (default true)
    - aad_object_id: optional. Teams sends aadObjectId in every message even
      without SSO, while it does NOT send the user's email. Populating this
      column (one-time export from M365 Admin Center → Users) lets the Teams
      bot recognize each user reliably with zero SSO configuration.
    """
    if not file.filename or not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode CSV as UTF-8.")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV is empty or missing header row.")

    # Normalize header names
    headers = [h.strip().lower() for h in reader.fieldnames]
    required = {"email", "full_name", "password", "department", "role"}
    missing = required - set(headers)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(sorted(missing))}. Expected: email, full_name, password, department, role")

    # Build column index map (case-insensitive)
    lower_map = {h.strip().lower(): h for h in reader.fieldnames}

    def get_val(row, key, default=""):
        raw_key = lower_map.get(key.lower())
        if raw_key is None:
            return default
        v = row.get(raw_key, default)
        return (v or default).strip() if isinstance(v, str) else v

    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    valid_departments = {"hr", "it", "finance"}
    # Also fetch dynamic slugs from DB
    try:
        from app.models.department import Department
        dept_rows = (await db.execute(select(Department.slug))).scalars().all()
        valid_departments.update([s.lower() for s in dept_rows])
    except Exception:
        pass
    valid_roles = {"user", "dept_admin", "super_admin"}

    created = 0
    skipped = 0
    errors = []
    seen_in_file = set()
    seen_aad_in_file = set()

    # Preload existing emails for fast duplicate check
    all_emails = set(r[0].lower() for r in (await db.execute(select(User.email))).all())
    all_aad_ids = set(r[0] for r in (await db.execute(select(User.aad_object_id).where(User.aad_object_id.isnot(None)))).all())

    for idx, row in enumerate(reader, start=2):  # row 2 = first data row
        email = get_val(row, "email", "").lower()
        full_name = get_val(row, "full_name", "")
        password = get_val(row, "password", "")
        department = get_val(row, "department", "").lower()
        role = get_val(row, "role", "").lower() or "user"
        daily_limit_raw = get_val(row, "daily_token_limit", "0")
        chat_enabled_raw = get_val(row, "chat_access_enabled", "true").lower()
        aad_id = get_val(row, "aad_object_id", "").strip()

        # Skip completely empty rows
        if not any([email, full_name, password, department, role]):
            continue

        # Validation
        if not email or not email_re.match(email):
            errors.append({"row": idx, "email": email or "(empty)", "error": "Invalid or missing email"})
            continue
        if email in seen_in_file:
            errors.append({"row": idx, "email": email, "error": "Duplicate email inside CSV"})
            continue
        if email in all_emails:
            skipped += 1
            errors.append({"row": idx, "email": email, "error": "Already exists — skipped"})
            continue
        if aad_id:
            if aad_id in seen_aad_in_file:
                errors.append({"row": idx, "email": email, "error": "Duplicate aad_object_id inside CSV"})
                continue
            if aad_id in all_aad_ids:
                skipped += 1
                errors.append({"row": idx, "email": email, "error": "aad_object_id already exists — skipped"})
                continue
        if not full_name:
            errors.append({"row": idx, "email": email, "error": "full_name is required"})
            continue
        if not password or len(password) < 6:
            errors.append({"row": idx, "email": email, "error": "password required (min 6 chars)"})
            continue
        if department not in valid_departments:
            errors.append({"row": idx, "email": email, "error": f"Invalid department '{department}'. Use: {', '.join(sorted(valid_departments))}"})
            continue
        if role not in valid_roles:
            errors.append({"row": idx, "email": email, "error": f"Invalid role '{role}'. Use: user, dept_admin, super_admin"})
            continue
        try:
            target_role = RoleType(role)
        except Exception:
            errors.append({"row": idx, "email": email, "error": f"Invalid role '{role}'"})
            continue
        if not can_manage_users(current_user.role, target_role):
            errors.append({"row": idx, "email": email, "error": f"Your role cannot create '{role}' users"})
            continue
        if current_user.role != RoleType.SUPER_ADMIN and department != current_user.department.value:
            errors.append({"row": idx, "email": email, "error": f"You can only create users in '{current_user.department.value}'"})
            continue
        try:
            daily_limit = int(daily_limit_raw) if str(daily_limit_raw).strip() != "" else 0
            if daily_limit < 0:
                raise ValueError
        except ValueError:
            errors.append({"row": idx, "email": email, "error": "daily_token_limit must be a non-negative integer"})
            continue
        chat_enabled = chat_enabled_raw not in ("false", "0", "no", "off")

        seen_in_file.add(email)
        all_emails.add(email)
        if aad_id:
            seen_aad_in_file.add(aad_id)
            all_aad_ids.add(aad_id)

        try:
            new_user = User(
                email=email,
                hashed_password=hash_password(password),
                full_name=full_name,
                department=department,
                role=target_role,
                chat_access_enabled=chat_enabled,
                daily_token_limit=max(0, daily_limit),
                aad_object_id=aad_id or None,
            )
            db.add(new_user)
            await db.flush()
            created += 1
        except Exception as e:
            errors.append({"row": idx, "email": email, "error": f"DB error: {str(e)[:120]}"})
            await db.rollback()

    return {
        "total_rows": created + skipped + len([e for e in errors if "skipped" not in e.get("error","").lower()]),
        "created": created,
        "skipped_existing": skipped,
        "errors": errors,
        "message": f"Created {created} user(s). Skipped {skipped} existing. {len([e for e in errors if 'skipped' not in e['error'].lower()])} row(s) had errors." if errors else f"Successfully created {created} user(s).",
    }


@router.get("/rbac/overview")
async def rbac_overview(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """RBAC matrix + counts for the admin panel (understandable overview)."""
    from app.models.department import Department
    depts = (await db.execute(select(Department))).scalars().all()
    # Role counts
    role_counts = {}
    for rt in RoleType:
        cnt = (await db.execute(select(func.count(User.id)).where(User.role == rt))).scalar() or 0
        role_counts[rt.value] = cnt
    # Dept counts
    dept_counts = {}
    for d in depts:
        cnt = (await db.execute(select(func.count(User.id)).where(User.department == d.slug))).scalar() or 0
        dept_counts[d.slug] = cnt
    return {
        "roles": [
            {"value": "super_admin", "label": "Super Admin", "scope": "All departments", "can": ["Manage users", "Manage groups", "Upload docs", "Query all"], "count": role_counts.get("super_admin", 0)},
            {"value": "dept_admin", "label": "Dept Admin", "scope": "Own department only", "can": ["Manage users (user role only)", "Upload docs", "Query own dept"], "count": role_counts.get("dept_admin", 0)},
            {"value": "user", "label": "User", "scope": "Via group membership", "can": ["Query assigned folders only"], "count": role_counts.get("user", 0)},
        ],
        "departments": [{"slug": d.slug, "name": d.name, "count": dept_counts.get(d.slug, 0)} for d in depts],
        "permissions": {
            "super_admin": {"create": True, "read": True, "update": True, "delete": True, "upload": True, "manage_users": True, "manage_groups": True},
            "dept_admin": {"create": True, "read": True, "update": True, "delete": True, "upload": True, "manage_users": "users only", "manage_groups": False},
            "user": {"create": False, "read": True, "update": False, "delete": False, "upload": False, "manage_users": False, "manage_groups": False},
        },
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_role(RoleType.SUPER_ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user (super admin only)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself.")
    await db.delete(user)
    return {"message": f"User {user.email} deleted."}

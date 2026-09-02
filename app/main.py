"""
AI-Bot FastAPI Application
Entry point — mounts all routes, initializes database, serves admin portal.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, Request, Depends, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

load_dotenv()

from app.config import get_settings
from app.database import init_db, close_db, get_db
from app.models.user import User, RoleType

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{settings.APP_NAME}] Starting up...")
    print(f"[{settings.APP_NAME}] Database: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
    print(f"[{settings.APP_NAME}] OpenAI Model: {settings.OPENAI_MODEL}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    await init_db()
    print(f"[{settings.APP_NAME}] Database tables created/verified.")

    await _reprocess_stuck_documents()

    # Build policy memory for existing documents uploaded before
    # the memory stage existed (runs in background)
    import asyncio as _asyncio
    _asyncio.create_task(_backfill_policy_memories())

    from app.memory.service import get_memory_service
    memory = get_memory_service(ttl_hours=settings.CONVERSATION_TTL_HOURS)
    cleaned = await memory.cleanup_expired()
    if cleaned:
        print(f"[{settings.APP_NAME}] Cleaned up {cleaned} expired conversation(s).")

    yield

    await close_db()
    print(f"[{settings.APP_NAME}] Shut down cleanly.")


async def _reprocess_stuck_documents():
    import asyncio
    from sqlalchemy import select as sel
    from app.database import AsyncSessionLocal
    from app.models.document import Document, ProcessingStatus
    from app.models.folder import Folder

    stuck_states = [
        ProcessingStatus.UPLOADING, ProcessingStatus.EXTRACTING,
        ProcessingStatus.CHUNKING, ProcessingStatus.EMBEDDING,
        ProcessingStatus.STORING, ProcessingStatus.UNDERSTANDING,
        ProcessingStatus.FAILED,
    ]

    async with AsyncSessionLocal() as db:
        result = await db.execute(sel(Document).where(Document.status.in_(stuck_states)))
        stuck_docs = result.scalars().all()

        if stuck_docs:
            print(f"[{settings.APP_NAME}] Found {len(stuck_docs)} stuck document(s), reprocessing...")
            from app.admin.document_routes import _process_document
            for doc in stuck_docs:
                folder_result = await db.execute(sel(Folder).where(Folder.id == doc.folder_id))
                folder = folder_result.scalar_one_or_none()
                department = folder.department if folder else ""
                print(f"[{settings.APP_NAME}]   Reprocessing: {doc.filename} (dept={department})")
                asyncio.create_task(_process_document(str(doc.id), doc.stored_path, department))


async def _backfill_policy_memories():
    """Background task: build policy memory for existing documents without one."""
    try:
        from app.rag.policy_memory import backfill_missing_memories
        processed = await backfill_missing_memories()
        if processed:
            print(f"[{settings.APP_NAME}] Policy memory backfilled for {processed} document(s).")
    except Exception as e:
        print(f"[{settings.APP_NAME}] Policy memory backfill failed: {e}")


# --- Create FastAPI App ---
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Enterprise Teams Chatbot with RAG Pipeline",
    lifespan=lifespan,
)

static_dir = Path(__file__).parent.parent / "static"
templates_dir = Path(__file__).parent.parent / "templates"
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


# --- Health Check ---
@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    from app.cache.service import get_stats as get_cache_stats
    
    health = {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "services": {}
    }
    
    # Check PostgreSQL
    try:
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            await db.execute(select(1))
        health["services"]["database"] = {"status": "healthy"}
    except Exception as e:
        health["services"]["database"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"
    
    # Check OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        models = client.models.list()
        health["services"]["openai"] = {
            "status": "healthy",
            "model": settings.OPENAI_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
        }
    except Exception as e:
        health["services"]["openai"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"
    
    # Check ChromaDB
    try:
        from app.rag.vectorstore import get_collection_stats
        stats = await get_collection_stats()
        health["services"]["chromadb"] = {
            "status": "healthy",
            "total_chunks": stats["total_chunks"],
        }
    except Exception as e:
        health["services"]["chromadb"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"
    
    # Check Cache
    try:
        cache_stats = get_cache_stats()
        health["services"]["cache"] = {
            "status": "healthy",
            "backend": cache_stats.get("backend", "unknown"),
            "keys": cache_stats.get("keys", 0),
        }
    except Exception as e:
        health["services"]["cache"] = {"status": "unhealthy", "error": str(e)}
    
    return JSONResponse(
        content=health,
        status_code=200 if health["status"] == "healthy" else 503,
    )


@app.get("/health/quick")
async def quick_health_check():
    """Quick health check for load balancers."""
    return JSONResponse(content={"status": "ok"})


# --- Import and Mount API Routes ---
from app.auth.routes import router as auth_router
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])

from app.admin.routes import router as admin_router
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])

from app.admin.usage_routes import router as usage_router
app.include_router(usage_router, prefix="/api/admin", tags=["Admin Usage"])

from app.admin.folder_routes import router as folder_router
app.include_router(folder_router, prefix="/api/folders", tags=["Folders"])

from app.admin.document_routes import router as document_router
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])

from app.admin.group_routes import router as group_router
app.include_router(group_router, tags=["Groups"])

from app.admin.department_routes import router as department_router
app.include_router(department_router, tags=["Departments"])

from app.admin.feedback_routes import router as feedback_router
app.include_router(feedback_router, prefix="/api/feedback", tags=["Feedback"])

from app.admin.rag_debug_routes import router as rag_debug_router
app.include_router(rag_debug_router, prefix="/api/rag", tags=["RAG Debug"])

# Teams Bot endpoint
from app.bot.bot_handler import router as bot_router
app.include_router(bot_router, tags=["Bot"])


# =============================================================================
# PUBLIC ROUTES — EthosAI Chatbot (no auth required)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def ethos_chatbot(request: Request):
    """Public EthosAI chatbot — everyone sees this first."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """Public privacy policy page."""
    return templates.TemplateResponse("privacy.html", {"request": request})


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    """Public terms of use page."""
    return templates.TemplateResponse("terms.html", {"request": request})


# =============================================================================
# ADMIN PORTAL — Requires auth (checked client-side via JS)
# =============================================================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_root(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/admin/folders", response_class=HTMLResponse)
async def admin_folders_page(request: Request):
    return templates.TemplateResponse("folders.html", {"request": request})

@app.get("/admin/documents", response_class=HTMLResponse)
async def admin_documents_page(request: Request):
    return templates.TemplateResponse("documents.html", {"request": request})

@app.get("/admin/departments", response_class=HTMLResponse)
async def admin_departments_page(request: Request):
    return templates.TemplateResponse("departments.html", {"request": request})

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request):
    return templates.TemplateResponse("users.html", {"request": request})

@app.get("/admin/groups", response_class=HTMLResponse)
async def admin_groups_page(request: Request):
    return templates.TemplateResponse("groups.html", {"request": request})

@app.get("/admin/chat", response_class=HTMLResponse)
async def admin_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# =============================================================================
# CHAT API — Used by Teams Bot + can be called by any authenticated client
# =============================================================================

class ChatMessage(BaseModel):
    message: str
    department: Optional[str] = None
    conversation_id: Optional[str] = None


@app.post("/api/chat")
async def chat_endpoint(request: ChatMessage, db: AsyncSession = Depends(get_db), req: Request = None):
    """
    Chat API — query the RAG pipeline.
    Per-user conversation memory with 24h expiry.
    Each user gets their own conversation threads.

    Access is gated server-side: only users registered in the database (and
    marked active) are forwarded to the LLM. Token usage is recorded per user.
    """
    from app.rag.agent import query_rag
    from app.memory.service import get_memory_service
    from app.auth.jwt import verify_token
    from app.models.user import User
    from app.admin.access import check_user_can_use_chat

    user_id = None
    user_dept = None

    # --- Server-side user gating ---
    # The request MUST present a valid JWT issued by this platform. The subject
    # is resolved against the database; only an existing, active user is allowed
    # through to the LLM.
    if not req:
        raise HTTPException(status_code=401, detail="Missing authentication token.")

    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = auth_header[7:]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    user_id = payload.get("sub")
    user_email = payload.get("email")
    user_dept = payload.get("department")

    result = await db.execute(select(User).where(User.id == user_id))
    db_user = result.scalar_one_or_none()
    if not db_user:
        raise HTTPException(
            status_code=403,
            detail="Your account is not registered with EthosAI.",
        )

    # Admin-managed access control + daily token limit enforcement
    allowed, reason = await check_user_can_use_chat(db, db_user)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    memory = get_memory_service(
        ttl_hours=settings.CONVERSATION_TTL_HOURS,
        max_messages=settings.MEMORY_MAX_MESSAGES,
    )

    conv = await memory.get_or_create_conversation(
        conversation_id=request.conversation_id,
        user_id=user_id,
    )
    conv_id = str(conv.id)

    history = await memory.get_history(conv_id)

    await memory.add_message(
        conversation_id=conv_id,
        role="user",
        content=request.message,
    )

    result = await query_rag(
        question=request.message,
        department=None,  # Chat searches ALL documents — RBAC is per-folder, not per-department
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

    result["conversation_id"] = conv_id
    result["usage"] = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return result



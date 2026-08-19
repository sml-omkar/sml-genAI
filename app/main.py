"""
AI-Bot FastAPI Application
Entry point — mounts all routes, initializes database, serves admin portal.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, Request, Depends, Query
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
    print(f"[{settings.APP_NAME}] Ollama: {settings.OLLAMA_BASE_URL}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    await init_db()
    print(f"[{settings.APP_NAME}] Database tables created/verified.")

    await _reprocess_stuck_documents()

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
        ProcessingStatus.STORING, ProcessingStatus.FAILED,
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
    return JSONResponse(content={"status": "healthy", "app": settings.APP_NAME})


# --- Import and Mount API Routes ---
from app.auth.routes import router as auth_router
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])

from app.admin.routes import router as admin_router
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])

from app.admin.folder_routes import router as folder_router
app.include_router(folder_router, prefix="/api/folders", tags=["Folders"])

from app.admin.document_routes import router as document_router
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])

from app.admin.group_routes import router as group_router
app.include_router(group_router, tags=["Groups"])

from app.admin.department_routes import router as department_router
app.include_router(department_router, tags=["Departments"])

# Teams Bot endpoint
from app.bot.bot_handler import router as bot_router
app.include_router(bot_router, tags=["Bot"])


# =============================================================================
# PUBLIC ROUTES — Cyprus AI Chatbot (no auth required)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def cyprus_chatbot(request: Request):
    """Public Cyprus AI chatbot — everyone sees this first."""
    return templates.TemplateResponse("chat.html", {"request": request})


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
async def chat_endpoint(request: ChatMessage, req: Request = None):
    """
    Chat API — query the RAG pipeline.
    Per-user conversation memory with 24h expiry.
    Each user gets their own conversation threads.
    """
    from app.rag.chain import query_rag
    from app.memory.service import get_memory_service
    from app.auth.jwt import verify_token

    user_id = None
    user_dept = None

    # Extract user from JWT for per-user memory
    if req:
        auth_header = req.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = verify_token(token)
            if payload:
                user_id = payload.get("sub")
                user_dept = payload.get("department")

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
    )

    await memory.add_message(
        conversation_id=conv_id,
        role="assistant",
        content=result.get("answer", ""),
        sources=result.get("sources"),
    )

    result["conversation_id"] = conv_id
    return result



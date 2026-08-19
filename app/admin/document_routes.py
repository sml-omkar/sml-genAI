"""
Document Routes
Upload PDFs, check processing status, delete documents.
Supports group-based access for regular users.
"""

import os
import asyncio
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User, RoleType
from app.models.folder import Folder
from app.models.document import Document, ProcessingStatus
from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    ProcessingStatusResponse,
)
from app.auth.dependencies import get_current_user
from app.rbac.middleware import filter_documents_by_access, validate_document_access
from app.config import get_settings

router = APIRouter()
settings = get_settings()

# In-memory pipeline logs — {doc_id: [{"time": "...", "stage": "...", "message": "..."}]}
_pipeline_logs: dict = {}


def _log_pipeline(doc_id: str, stage: str, message: str):
    """Append a timestamped log entry for a document's pipeline."""
    import time
    from datetime import datetime
    if doc_id not in _pipeline_logs:
        _pipeline_logs[doc_id] = []
    _pipeline_logs[doc_id].append({
        "time": datetime.utcnow().strftime("%H:%M:%S"),
        "stage": stage,
        "message": message,
    })


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    folder_id: str = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List documents, optionally filtered by folder. RBAC filtered."""
    query = select(Document).order_by(Document.created_at.desc())

    if folder_id:
        query = query.where(Document.folder_id == folder_id)

    query = await filter_documents_by_access(query, current_user, db)
    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in documents],
        total=len(documents),
    )


@router.post("/upload/{folder_id}", response_model=DocumentResponse, status_code=201)
async def upload_document(
    folder_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF to a specific folder. Users cannot upload."""
    folder_result = await db.execute(select(Folder).where(Folder.id == folder_id))
    folder = folder_result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")

    if current_user.role == RoleType.USER:
        raise HTTPException(status_code=403, detail="Users cannot upload documents.")

    if not await validate_document_access(current_user, Document(), folder, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    content = await file.read()
    file_size = len(content)
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB")

    doc_id = uuid4()
    safe_filename = f"{doc_id}_{file.filename}"
    folder_path = os.path.join(settings.UPLOAD_DIR, folder.department)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, safe_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    document = Document(
        id=doc_id,
        filename=file.filename,
        stored_path=file_path,
        file_size=file_size,
        folder_id=folder_id,
        uploaded_by=current_user.id,
        status=ProcessingStatus.UPLOADING,
    )
    db.add(document)
    await db.flush()
    await db.refresh(document)

    asyncio.create_task(_process_document(str(doc_id), file_path, folder.department))

    return DocumentResponse.model_validate(document)


async def _process_document(doc_id: str, file_path: str, department: str):
    """Background task: run the full RAG pipeline on an uploaded document."""
    from app.database import AsyncSessionLocal
    from app.rag.pdf_extractor import extract_pdf_text
    from app.rag.chunker import chunk_text
    from app.rag.embeddings import embed_texts
    from app.rag.vectorstore import store_chunks

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            document = result.scalar_one_or_none()
            if not document:
                return

            _log_pipeline(doc_id, "start", f"Processing {document.filename}")
            document.status = ProcessingStatus.EXTRACTING
            await db.commit()
            _log_pipeline(doc_id, "extract", "Extracting text from PDF...")
            print(f"[RAG] {document.filename} — Stage 1/4: Extracting text...")

            try:
                extracted = await extract_pdf_text(file_path)
                full_text = extracted["text"]
                total_pages = extracted["pages"]
                _log_pipeline(doc_id, "extract", f"Extracted {len(full_text)} chars from {total_pages} pages")
                print(f"[RAG] {document.filename} — Extracted {len(full_text)} chars from {total_pages} pages")
            except Exception as e:
                document.status = ProcessingStatus.FAILED
                document.error_message = f"Extraction failed: {str(e)}"
                _log_pipeline(doc_id, "error", f"Extraction failed: {str(e)}")
                await db.commit()
                return

            document.status = ProcessingStatus.CHUNKING
            document.total_pages = total_pages
            await db.commit()
            _log_pipeline(doc_id, "chunk", "Splitting text into chunks...")
            print(f"[RAG] {document.filename} — Stage 2/4: Chunking text...")

            try:
                chunks = chunk_text(full_text)
                _log_pipeline(doc_id, "chunk", f"Created {len(chunks)} chunks")
                print(f"[RAG] {document.filename} — Created {len(chunks)} chunks")
            except Exception as e:
                document.status = ProcessingStatus.FAILED
                document.error_message = f"Chunking failed: {str(e)}"
                _log_pipeline(doc_id, "error", f"Chunking failed: {str(e)}")
                await db.commit()
                return

            document.status = ProcessingStatus.EMBEDDING
            await db.commit()
            _log_pipeline(doc_id, "embed", f"Generating embeddings for {len(chunks)} chunks...")
            print(f"[RAG] {document.filename} — Stage 3/4: Generating embeddings...")

            try:
                chunk_texts = [c["text"] for c in chunks]
                embeddings = embed_texts(chunk_texts)
                _log_pipeline(doc_id, "embed", f"Generated {len(embeddings)} embeddings")
                print(f"[RAG] {document.filename} — Generated {len(embeddings)} embeddings")
            except Exception as e:
                document.status = ProcessingStatus.FAILED
                document.error_message = f"Embedding failed: {str(e)}"
                _log_pipeline(doc_id, "error", f"Embedding failed: {str(e)}")
                await db.commit()
                return

            document.status = ProcessingStatus.STORING
            await db.commit()
            _log_pipeline(doc_id, "store", f"Storing {len(chunks)} chunks in vector database...")
            print(f"[RAG] {document.filename} — Stage 4/4: Storing in ChromaDB...")

            try:
                await store_chunks(
                    document_id=doc_id,
                    folder_id=str(document.folder_id),
                    department=department,
                    document_name=document.filename,
                    chunks=chunks,
                    embeddings=embeddings,
                )
                _log_pipeline(doc_id, "done", f"Done! {len(chunks)} chunks stored successfully")
                print(f"[RAG] {document.filename} — Stored {len(chunks)} chunks in ChromaDB")
            except Exception as e:
                document.status = ProcessingStatus.FAILED
                document.error_message = f"Vector store failed: {str(e)}"
                _log_pipeline(doc_id, "error", f"Vector store failed: {str(e)}")
                await db.commit()
                return

            document.status = ProcessingStatus.READY
            document.chunk_count = len(chunks)
            document.processed_at = datetime.utcnow()
            document.error_message = None
            _log_pipeline(doc_id, "done", f"Processing complete — {len(chunks)} chunks ready for search")
            await db.commit()
            print(f"[RAG] {document.filename} — DONE! {len(chunks)} chunks ready.")

        except Exception as e:
            document.status = ProcessingStatus.FAILED
            document.error_message = f"Unexpected error: {str(e)}"
            _log_pipeline(doc_id, "error", f"Unexpected error: {str(e)}")
            await db.commit()
            print(f"[RAG] ERROR processing document {doc_id}: {e}")


@router.get("/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get real-time RAG processing status for a document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    folder_result = await db.execute(select(Folder).where(Folder.id == document.folder_id))
    folder = folder_result.scalar_one_or_none()
    if folder and not await validate_document_access(current_user, document, folder, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    status_progress = {
        ProcessingStatus.UPLOADING: 10,
        ProcessingStatus.EXTRACTING: 30,
        ProcessingStatus.CHUNKING: 50,
        ProcessingStatus.EMBEDDING: 70,
        ProcessingStatus.STORING: 90,
        ProcessingStatus.READY: 100,
        ProcessingStatus.FAILED: 0,
    }

    return ProcessingStatusResponse(
        document_id=str(document.id),
        filename=document.filename,
        status=document.status.value,
        total_pages=document.total_pages,
        chunk_count=document.chunk_count,
        error_message=document.error_message,
        progress_percent=status_progress.get(document.status, 0),
    )


@router.get("/{document_id}/logs")
async def get_pipeline_logs(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get real-time pipeline processing logs for a document."""
    return {"logs": _pipeline_logs.get(document_id, [])}


@router.post("/{document_id}/retry", response_model=DocumentResponse)
async def retry_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retry processing a failed document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    if current_user.role == RoleType.USER:
        raise HTTPException(status_code=403, detail="Users cannot retry documents.")

    folder_result = await db.execute(select(Folder).where(Folder.id == document.folder_id))
    folder = folder_result.scalar_one_or_none()
    department = folder.department if folder else ""

    document.status = ProcessingStatus.UPLOADING
    document.error_message = None
    await db.flush()

    asyncio.create_task(_process_document(str(document.id), document.stored_path, department))

    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and remove its chunks from ChromaDB."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    folder_result = await db.execute(select(Folder).where(Folder.id == document.folder_id))
    folder = folder_result.scalar_one_or_none()
    if folder and not await validate_document_access(current_user, document, folder, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    if current_user.role == RoleType.USER:
        raise HTTPException(status_code=403, detail="Users cannot delete documents.")

    try:
        from app.rag.vectorstore import delete_documents_from_vectordb
        await delete_documents_from_vectordb(
            [str(document.id)],
            [document.filename],
        )
    except Exception as e:
        print(f"[WARNING] ChromaDB cleanup failed: {e}")

    if os.path.exists(document.stored_path):
        os.remove(document.stored_path)

    if folder:
        folder.document_count = max(0, folder.document_count - 1)

    await db.delete(document)
    return {"message": f"Document '{document.filename}' deleted."}

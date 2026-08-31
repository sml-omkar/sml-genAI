"""
Re-index all documents in ChromaDB with the current chunking/embedding settings.

Clears each document's existing chunks, then re-runs the full RAG pipeline
(extract -> chunk -> embed -> store -> policy memory) using the source PDFs
already saved in data/uploads/.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ENVIRONMENT", "production")

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.document import Document
from app.models.folder import Folder
from app.rag.vectorstore import delete_documents_from_vectordb, get_collection
from app.rag.pdf_extractor import extract_pdf_text
from app.rag.chunker import chunk_text, count_tokens_approx
from app.rag.embeddings import embed_texts
from app.rag.vectorstore import store_chunks


async def reindex_all():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document))
        documents = result.scalars().all()

    if not documents:
        print("No documents found in the database.")
        return

    print(f"Found {len(documents)} documents to re-index.\n")

    # Clear all vector chunks for these documents first
    doc_ids = [str(d.id) for d in documents]
    doc_names = [d.filename for d in documents]
    await delete_documents_from_vectordb(doc_ids, doc_names)
    print(f"[REINDEX] Cleared existing chunks for {len(documents)} documents.\n")

    for document in documents:
        doc_id = str(document.id)
        file_path = document.stored_path

        if not file_path or not os.path.exists(file_path):
            print(f"[REINDEX] SKIP {document.filename} — source file not found at: {file_path}")
            continue

        folder_id = str(document.folder_id) if document.folder_id else ""
        department = document.folder.department if document.folder else "it"

        print(f"[REINDEX] Processing: {document.filename} (dept={department})")

        try:
            extracted = await extract_pdf_text(file_path)
            full_text = extracted["text"]
            chunks = chunk_text(full_text)
            print(f"[REINDEX]   -> {len(chunks)} chunks")

            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_texts(chunk_texts)
            print(f"[REINDEX]   -> {len(embeddings)} embeddings generated")

            await store_chunks(
                document_id=doc_id,
                folder_id=folder_id,
                department=department,
                document_name=document.filename,
                chunks=chunks,
                embeddings=embeddings,
            )
            print(f"[REINDEX]   -> Stored {len(chunks)} chunks in vector DB")

            # Rebuild policy memory
            try:
                from app.rag.policy_memory import build_and_store_policy_memory
                memory = await build_and_store_policy_memory(
                    document_id=doc_id,
                    folder_id=folder_id,
                    department=department,
                    document_name=document.filename,
                    chunks=chunks,
                )
                if memory:
                    print(f"[REINDEX]   -> Memory rebuilt ({len(memory['key_facts'])} key facts)")
                else:
                    print("[REINDEX]   -> Memory generation skipped")
            except Exception as e:
                print(f"[REINDEX]   -> Memory skipped ({e})")

        except Exception as e:
            print(f"[REINDEX] ERROR processing {document.filename}: {e}")

    print("\n[REINDEX] Done.")


if __name__ == "__main__":
    asyncio.run(reindex_all())

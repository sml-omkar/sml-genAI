"""
Policy Memory Generator
Ingestion-time "understanding" stage.

After a document is chunked + embedded, the LLM reads every chunk and
distills the policy into a structured memory:
  - summary:   what the document covers
  - key_facts: concrete rules (numbers, deadlines, penalties, steps, names)

Storage:
  1. PostgreSQL  -> document_memories table (full memory, injected at query time)
  2. ChromaDB    -> each key fact embedded as a retrievable "memory note"
                    (is_memory=True) so hybrid search can hit it directly

Memory generation is non-fatal: if the LLM fails, the document still
becomes READY and normal RAG continues to work.
"""

import asyncio
import json
import re
from typing import List, Dict, Optional

from openai import OpenAI

from app.config import get_settings

settings = get_settings()

# Max key facts stored per document (keeps prompts bounded at query time)
MAX_FACTS = 40
# Chars of raw chunk text sent to the LLM per batch (llama3.2 8k ctx safe budget)
BATCH_CHAR_LIMIT = 10000

EXTRACT_SYSTEM = """You are building an internal knowledge base for a company AI assistant.

Read the following excerpt from the company document "{doc_name}" and extract the important facts, rules and policies.

Rules:
- Extract only information that is actually present in the text — never invent anything
- Keep every specific detail: numbers, amounts, days, deadlines, penalties, names, steps, conditions, contact info
- Write each fact as one short, complete, self-contained sentence
- Skip table-of-contents, page headers/footers and repeated boilerplate

Respond with ONLY a JSON object:
{{"facts": ["fact 1", "fact 2", ...]}}"""

SUMMARIZE_SYSTEM = """You are building an internal knowledge base for a company AI assistant.

You are given extracted facts from the company document "{doc_name}". Produce:

1. "summary": 2-4 sentences describing what this document/policy covers and who it applies to.
2. "key_facts": the consolidated list of the most important facts/rules, deduplicated.
   Keep every specific detail: numbers, amounts, days, deadlines, penalties, conditions.

Respond with ONLY a JSON object:
{{"summary": "...", "key_facts": ["fact 1", "fact 2", ...]}}"""


def _parse_json_loose(content: str) -> Optional[dict]:
    """
    Lenient JSON parsing for small LLMs that emit near-JSON
    (single quotes, trailing commas, prose around the object).
    """
    content = content.strip()
    try:
        return json.loads(content)
    except Exception:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None

    candidate = match.group()
    attempts = [
        candidate,
        candidate.replace("'", '"'),
        re.sub(r",\s*([}\]])", r"\1", candidate),
        re.sub(r",\s*([}\]])", r"\1", candidate.replace("'", '"')),
        re.sub(r"[\r\n\t]+", " ", candidate.replace("'", '"')),
    ]
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except Exception:
            continue

    # Last resort: pull out any list of quoted strings in the text
    strings = re.findall(r'"([^"]{10,})"|\'([^\']{10,})\'', candidate)
    flat = [a or b for a, b in strings]
    if flat:
        return {"facts": flat}
    return None


def _llm_json_sync(system: str, user: str, temperature: float, max_tokens: int) -> Optional[dict]:
    """Call the LLM and parse a JSON object out of the response (blocking)."""
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()
        parsed = _parse_json_loose(content)
        if parsed is None:
            print(f"[MEMORY] Unparseable LLM output: {content[:200]}")
        return parsed
    except Exception as e:
        print(f"[MEMORY] LLM error: {e}")
        return None


async def _llm_json(system: str, user: str, temperature: float, max_tokens: int) -> Optional[dict]:
    """Run the blocking LLM call in a thread so the event loop stays responsive."""
    return await asyncio.to_thread(
        _llm_json_sync, system, user, temperature, max_tokens
    )


def _batch_chunks(chunks: List[Dict]) -> List[List[Dict]]:
    """Group chunks into batches that fit the LLM context window."""
    batches: List[List[Dict]] = []
    current: List[Dict] = []
    size = 0
    for c in chunks:
        tlen = len(c["text"])
        if current and size + tlen > BATCH_CHAR_LIMIT:
            batches.append(current)
            current, size = [], 0
        current.append(c)
        size += tlen
    if current:
        batches.append(current)
    return batches


async def generate_policy_memory(chunks: List[Dict], document_name: str) -> Optional[Dict]:
    """
    Read all chunks of a document through the LLM and distill them into
    {"summary": str, "key_facts": [str]}.
    Returns None if understanding failed (caller should continue gracefully).
    """
    if not chunks:
        return None

    batches = _batch_chunks(chunks)

    # Single pass for small documents: extract + summarize together
    if len(batches) == 1:
        text_block = "\n\n".join(c["text"] for c in batches[0])
        result = await _llm_json(
            SUMMARIZE_SYSTEM.replace("{doc_name}", document_name),
            f"Full document text:\n\n{text_block}\n\nProduce the summary and key_facts now.",
            temperature=0.2,
            max_tokens=1600,
        )
        if result and result.get("key_facts"):
            return _clean(result)
        # Fall through to two-pass if single-pass parsing failed

    # Two-pass: extract facts per batch, then merge into summary + final facts
    all_notes: List[str] = []
    for i, batch in enumerate(batches, 1):
        text_block = "\n\n".join(c["text"] for c in batch)
        print(f"[MEMORY] {document_name} — reading batch {i}/{len(batches)}...")
        result = await _llm_json(
            EXTRACT_SYSTEM.replace("{doc_name}", document_name),
            f"Document excerpt:\n\n{text_block}",
            temperature=0.1,
            max_tokens=1200,
        )
        if result and isinstance(result.get("facts"), list):
            all_notes.extend(str(f) for f in result["facts"])

    if not all_notes:
        return None

    # Merge pass (cap notes fed to the merge prompt)
    notes_text = "\n".join(f"- {n}" for n in all_notes[:120])
    merged = await _llm_json(
        SUMMARIZE_SYSTEM.replace("{doc_name}", document_name),
        f"Extracted facts:\n\n{notes_text}\n\nConsolidate them now.",
        temperature=0.2,
        max_tokens=2000,
    )
    if merged and merged.get("key_facts"):
        return _clean(merged)

    # Fallback: use raw extraction notes without LLM merge
    return {
        "summary": "",
        "key_facts": all_notes[:MAX_FACTS],
        "model_used": settings.OPENAI_MODEL,
    }


def _clean(result: dict) -> Dict:
    """Normalize/validate the LLM output."""
    summary = str(result.get("summary") or "").strip()
    facts_raw = result.get("key_facts") or result.get("facts") or []
    seen = set()
    facts = []
    for f in facts_raw:
        f = str(f).strip()
        if len(f) < 5:
            continue
        k = f.lower()
        if k not in seen:
            seen.add(k)
            facts.append(f)
    return {
        "summary": summary[:4000],
        "key_facts": facts[:MAX_FACTS],
        "model_used": settings.OPENAI_MODEL,
    }


# ============================================================================
# Persistence
# ============================================================================

async def build_and_store_policy_memory(
    document_id: str,
    folder_id: str,
    department: str,
    document_name: str,
    chunks: List[Dict],
) -> Optional[Dict]:
    """
    Full memory step: LLM understanding -> Postgres + ChromaDB memory notes.
    Returns the memory dict, or None if generation failed.
    """
    memory = await generate_policy_memory(chunks, document_name)
    if not memory or not memory.get("key_facts"):
        print(f"[MEMORY] {document_name} — no memory generated (skipping)")
        return None

    # 1. Store in PostgreSQL
    await save_memory(document_id, memory)
    print(f"[MEMORY] {document_name} — saved {len(memory['key_facts'])} fact(s) to database")

    # 2. Embed key facts as retrievable memory notes in ChromaDB
    try:
        from app.rag.embeddings import embed_texts
        from app.rag.vectorstore import store_memory_notes
        note_embeddings = embed_texts(memory["key_facts"])
        await store_memory_notes(
            document_id=document_id,
            folder_id=folder_id,
            department=department,
            document_name=document_name,
            notes=memory["key_facts"],
            embeddings=note_embeddings,
        )
        print(f"[MEMORY] {document_name} — stored {len(note_embeddings)} memory note(s) in ChromaDB")
    except Exception as e:
        # Notes are an optimization; DB memory is the source of truth
        print(f"[MEMORY] {document_name} — ChromaDB note storage failed: {e}")

    return memory


async def save_memory(document_id: str, memory: Dict):
    """Upsert the memory row in PostgreSQL."""
    from sqlalchemy import select
    from app.database import AsyncSessionLocal
    from app.models.document_memory import DocumentMemory

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DocumentMemory).where(DocumentMemory.document_id == document_id)
        )
        row = result.scalar_one_or_none()
        if not row:
            row = DocumentMemory(document_id=document_id)
            db.add(row)
        row.summary = memory.get("summary", "")
        row.key_facts = memory.get("key_facts", [])
        row.fact_count = len(memory.get("key_facts", []))
        row.model_used = memory.get("model_used")
        await db.commit()


async def get_memories_for_documents(document_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch memories for a set of document ids.
    Returns {document_id: {"summary", "key_facts", "document_name"}}.
    """
    if not document_ids:
        return {}

    from sqlalchemy import select
    from app.database import AsyncSessionLocal
    from app.models.document_memory import DocumentMemory
    from app.models.document import Document

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DocumentMemory, Document.filename)
            .join(Document, Document.id == DocumentMemory.document_id)
            .where(DocumentMemory.document_id.in_(document_ids))
        )
        rows = result.all()

    memories = {}
    for mem, filename in rows:
        memories[str(mem.document_id)] = {
            "summary": mem.summary or "",
            "key_facts": mem.key_facts or [],
            "document_name": filename,
        }
    return memories


async def delete_memory(document_id: str):
    """Remove the memory row when its document is deleted."""
    from sqlalchemy import delete as sa_delete
    from app.database import AsyncSessionLocal
    from app.models.document_memory import DocumentMemory

    async with AsyncSessionLocal() as db:
        await db.execute(
            sa_delete(DocumentMemory).where(DocumentMemory.document_id == document_id)
        )
        await db.commit()


# ============================================================================
# Backfill — build memory for documents uploaded before this feature existed
# ============================================================================

async def count_docs_missing_memory() -> int:
    """Count READY documents that have no policy memory yet."""
    from sqlalchemy import select, func
    from app.database import AsyncSessionLocal
    from app.models.document import Document, ProcessingStatus
    from app.models.document_memory import DocumentMemory

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(func.count(Document.id))
            .join(
                DocumentMemory,
                DocumentMemory.document_id == Document.id,
                isouter=True,
            )
            .where(Document.status == ProcessingStatus.READY)
            .where(DocumentMemory.id.is_(None))
        )
        return result.scalar() or 0


async def backfill_missing_memories() -> int:
    """
    Find READY documents that have no policy memory and build it for them.
    Chunks are recovered from ChromaDB (they are not stored in Postgres).
    Returns the number of documents processed successfully.
    """
    from sqlalchemy import select
    from app.database import AsyncSessionLocal
    from app.models.document import Document, ProcessingStatus
    from app.models.document_memory import DocumentMemory

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document)
            .join(
                DocumentMemory,
                DocumentMemory.document_id == Document.id,
                isouter=True,
            )
            .where(Document.status == ProcessingStatus.READY)
            .where(DocumentMemory.id.is_(None))
            .order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()

    if not docs:
        return 0

    print(f"[MEMORY] Backfill: {len(docs)} existing document(s) missing policy memory")

    from app.rag.vectorstore import get_chunks_for_document

    processed = 0
    for doc in docs:
        try:
            chunks = await get_chunks_for_document(str(doc.id))
            if not chunks:
                print(f"[MEMORY] Backfill: {doc.filename} has no stored chunks, skipping")
                continue

            # Department is available in chunk metadata
            department = chunks[0].get("metadata", {}).get("department", "")

            memory = await build_and_store_policy_memory(
                document_id=str(doc.id),
                folder_id=str(doc.folder_id),
                department=department,
                document_name=doc.filename,
                chunks=chunks,
            )
            if memory:
                processed += 1
                print(f"[MEMORY] Backfill: {doc.filename} — done ({len(memory['key_facts'])} facts)")
        except Exception as e:
            print(f"[MEMORY] Backfill: {doc.filename} — failed: {e}")

    print(f"[MEMORY] Backfill complete: {processed}/{len(docs)} document(s) now have policy memory")
    return processed

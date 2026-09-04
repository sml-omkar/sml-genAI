"""
Vector Store (ChromaDB)
Store, search, and manage document embeddings.
Hybrid search: vector similarity + keyword matching for precise retrieval.
Cross-encoder reranker for final chunk selection.
"""

from typing import List, Dict, Optional
from functools import lru_cache
import re
import os

# --- Silence ChromaDB telemetry (posthog capture signature mismatch) ---
# Chroma 0.6.x tries to call posthog.capture(event, props) but posthog 3.x
# expects capture(distinct_id, event). This spams "Failed to send telemetry"
# on every get_or_create_collection / query. Disable before importing chromadb.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

settings = get_settings()

# Monkey-patch posthog.capture to no-op so Chroma's background telemetry thread
# never throws "capture() takes 1 positional argument but 3 were given"
try:
    import posthog as _posthog

    def _noop_capture(*args, **kwargs):
        return None

    _posthog.capture = _noop_capture
    # Some Chroma versions import via chromadb.telemetry.posthog
    try:
        import chromadb.telemetry.posthog as _chroma_pg  # type: ignore

        _chroma_pg.capture = _noop_capture
    except Exception:
        pass
except Exception:
    pass

# Cross-encoder reranker (lazy loaded)
_reranker = None


def _get_reranker():
    """Lazy-load cross-encoder reranker (~80MB model, loaded once)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("[RERANKER] Loading cross-encoder model...")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[RERANKER] Cross-encoder loaded")
        except Exception as e:
            print(f"[RERANKER] Failed to load cross-encoder: {e}")
            _reranker = False  # Mark as failed, don't retry
    return _reranker if _reranker is not False else None


class NoOpEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: list) -> list:
        raise NotImplementedError("Embeddings are pre-computed via OpenAI.")


_chroma_client = None

def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        import os
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
        },
    )
    return collection


async def store_chunks(
    document_id: str,
    folder_id: str,
    department: str,
    document_name: str,
    chunks: List[Dict],
    embeddings: List[List[float]],
):
    collection = get_collection()

    ids = [f"{document_id}_chunk_{c['index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "document_id": document_id,
            "folder_id": folder_id,
            "department": department.lower(),
            "document_name": document_name,
            "chunk_index": c["index"],
            "char_start": c.get("char_start", 0),
            "is_table": c.get("is_table", False),
            "page_number": c.get("page_number", 0),
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[VECTORDB] Stored {len(chunks)} chunks for document: {document_name}")


async def store_memory_notes(
    document_id: str,
    folder_id: str,
    department: str,
    document_name: str,
    notes: List[str],
    embeddings: List[List[float]],
):
    """
    Store LLM-generated policy memory notes as retrievable embeddings.
    Same document_id metadata as regular chunks, so they are cleaned up
    automatically when the document is deleted.
    """
    collection = get_collection()

    ids = [f"{document_id}_memory_{i}" for i in range(len(notes))]
    metadatas = [
        {
            "document_id": document_id,
            "folder_id": folder_id,
            "department": department.lower(),
            "document_name": document_name,
            "chunk_index": -1,
            "char_start": 0,
            "is_table": False,
            "page_number": 0,
            "is_memory": True,
        }
        for _ in notes
    ]

    collection.add(
        ids=ids,
        documents=notes,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[VECTORDB] Stored {len(notes)} memory notes for document: {document_name}")


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from query for hybrid search."""
    # Remove common stop words
    stop_words = {
        'what', 'is', 'the', 'are', 'a', 'an', 'in', 'of', 'for', 'to',
        'and', 'or', 'how', 'do', 'does', 'did', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'used', 'using',
        'that', 'this', 'it', 'its', 'on', 'at', 'by', 'with', 'from',
        'as', 'into', 'about', 'me', 'my', 'tell', 'explain', 'give',
        'please', 'company', 'our', 'we', 'i', 'you', 'your',
    }
    words = re.findall(r'[a-zA-Z0-9]+(?:\.[0-9]+)?', query.lower())
    return [w for w in words if w not in stop_words and len(w) > 1]


def _keyword_boost(query: str, doc_text: str, base_score: float) -> float:
    """
    Boost score if chunk contains exact keywords from the query.
    Also boosts for matching numbers and technical terms.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return base_score

    doc_lower = doc_text.lower()
    query_lower = query.lower()

    # Keyword matches — give bigger boost for each keyword found
    kw_matches = [kw for kw in keywords if kw in doc_lower]
    kw_ratio = len(kw_matches) / len(keywords) if keywords else 0

    # Number extraction boost — if query asks about a number and doc has it
    query_numbers = set(re.findall(r'\d+', query))
    doc_numbers = set(re.findall(r'\d+', doc_text))
    number_overlap = len(query_numbers & doc_numbers)
    number_boost = min(number_overlap * 0.12, 0.25) if query_numbers else 0

    # Boost for multi-word exact phrases (e.g. "EBS volume" matching "ebs volume")
    phrase_boost = 0
    words = query_lower.split()
    for window in [2, 3]:
        for i in range(len(words) - window + 1):
            phrase = " ".join(words[i:i+window])
            if len(phrase) > 4 and phrase in doc_lower:
                phrase_boost = max(phrase_boost, 0.12)

    # Partial/stemmed keyword boost — catches "license" matching "licenses", "monitor" matching "monitoring"
    partial_boost = 0
    for kw in keywords:
        if kw not in doc_lower:
            # Check if the keyword appears as a prefix of a word in the doc (stemming)
            if re.search(rf'\b{re.escape(kw[:-1])}\w*', doc_lower):
                partial_boost = max(partial_boost, 0.06)
            else:
                # Check if any doc word starts with the keyword (e.g. "policy" matches "policies")
                if re.search(rf'\b{re.escape(kw)}\w*\b', doc_lower):
                    partial_boost = max(partial_boost, 0.04)

    # If ALL keywords (or nearly all) match, give a strong boost
    all_match_boost = 0.30 if kw_ratio >= 0.8 else (0.15 if kw_ratio >= 0.5 else 0)

    # If NO keywords match at all, penalize slightly to push down irrelevant chunks
    no_match_penalty = -0.05 if kw_ratio == 0 else 0

    boost = (
        kw_ratio * 0.30
        + number_boost
        + phrase_boost
        + partial_boost
        + all_match_boost
        + no_match_penalty
    )
    return base_score + boost


async def search_similar(
    query: str,
    department: Optional[str] = None,
    folder_ids: Optional[List[str]] = None,
    n_results: int = None,
) -> List[Dict]:
    """
    Hybrid search: vector similarity + keyword matching.
    Retrieves more candidates than needed, then re-ranks with keyword boost.
    Supports single-dept + multi-folder scoping for external APIs.
    """
    from app.rag.embeddings import embed_query

    collection = get_collection()

    if n_results is None:
        n_results = settings.TOP_K_RESULTS

    # Retrieve many candidates for hybrid re-ranking
    # The keyword boost needs raw candidates to promote
    collection_count = collection.count()
    search_n = min(max(n_results * 5, 25), 50, collection_count) if collection_count > 0 else 1

    query_embedding = embed_query(query)

    # Build Chroma where filter: folder_ids takes precedence over department
    # (external APIs scope to explicit folders; Teams/web use department)
    where_filter = None
    if folder_ids:
        # Chroma $in requires list of strings
        clean_ids = [str(fid) for fid in folder_ids if fid]
        if len(clean_ids) == 1:
            where_filter = {"folder_id": {"$eq": clean_ids[0]}}
        elif clean_ids:
            where_filter = {"folder_id": {"$in": clean_ids}}
    elif department:
        where_filter = {"department": {"$eq": department.lower()}}

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": search_n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_params["where"] = where_filter

    results = collection.query(**query_params)

    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            base_score = 1 - dist
            # Apply keyword boost for hybrid re-ranking
            boosted_score = _keyword_boost(query, doc, base_score)
            chunks.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "relevance_score": boosted_score,
                "base_score": base_score,
            })

    # Sort by boosted score (highest first)
    chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Step 3: Cross-encoder rerank top candidates
    reranker = _get_reranker()
    if reranker and len(chunks) > 5:
        # Rerank top 15 candidates down to the best 8
        candidates = chunks[:15]
        pairs = [(query, c["text"][:512]) for c in candidates]  # CrossEncoder has 512 token limit
        scores = reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x.get("rerank_score", -999), reverse=True)
        chunks = candidates[:8] + chunks[15:]  # Keep reranked top 8 + remaining

    return chunks[:n_results]


async def get_chunks_for_document(document_id: str) -> List[Dict]:
    """
    Fetch all stored chunks of a document from ChromaDB.
    Used to rebuild policy memory for documents uploaded before
    the memory stage existed (chunks are not kept in Postgres).
    Excludes memory notes; sorted by original chunk order.
    """
    collection = get_collection()
    results = collection.get(
        where={"document_id": {"$eq": document_id}},
        include=["documents", "metadatas"],
    )
    chunks = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("is_memory"):
                continue
            chunks.append({
                "text": doc,
                "index": meta.get("chunk_index", 0),
                "page_number": meta.get("page_number", 0),
                "char_start": meta.get("char_start", 0),
                "is_table": meta.get("is_table", False),
                "metadata": meta,
            })
    chunks.sort(key=lambda c: c["index"])
    return chunks


async def delete_documents_from_vectordb(document_ids: List[str], document_names: List[str] = None):
    """Delete chunks from ChromaDB by document ID or document name."""
    collection = get_collection()

    # Delete by document_id
    for doc_id in document_ids:
        results = collection.get(
            where={"document_id": {"$eq": doc_id}},
            include=[],
        )
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"[VECTORDB] Deleted {len(results['ids'])} chunks for document_id: {doc_id}")

    # Also delete by document_name (handles rebuilt data with different IDs)
    if document_names:
        for name in document_names:
            results = collection.get(
                where={"document_name": {"$eq": name}},
                include=[],
            )
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                print(f"[VECTORDB] Deleted {len(results['ids'])} chunks for document_name: {name}")


async def get_collection_stats() -> Dict:
    collection = get_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": settings.CHROMA_COLLECTION,
    }

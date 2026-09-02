"""
Cache Service
In-memory caching for RAG responses and frequent queries.
Only ever caches real document-sourced answers — the callers skip caching
canned / out-of-scope / error replies so users never see the same hardcoded
answer repeated. Cache is per-process: cleared automatically on restart.
"""

import hashlib
import time
from typing import Optional, Dict, Any

from app.config import get_settings

settings = get_settings()

_memory_cache: Dict[str, Dict] = {}
MEMORY_CACHE_MAX = 500


def cache_get(key: str) -> Optional[Dict]:
    """Get value from in-memory cache."""
    entry = _memory_cache.get(key)
    if entry and entry["expires"] > time.time():
        return entry["value"]
    elif entry:
        del _memory_cache[key]
    return None


def cache_set(key: str, value: Dict, ttl_seconds: int = 3600):
    """Set value in in-memory cache with TTL."""
    if len(_memory_cache) >= MEMORY_CACHE_MAX:
        # Evict oldest entry
        oldest_key = min(_memory_cache, key=lambda k: _memory_cache[k]["expires"])
        del _memory_cache[oldest_key]

    _memory_cache[key] = {
        "value": value,
        "expires": time.time() + ttl_seconds,
    }


def cache_delete(pattern: str):
    """Delete cache entries matching pattern."""
    # Clear memory cache
    keys_to_delete = [k for k in _memory_cache if pattern.replace("*", "") in k]
    for k in keys_to_delete:
        del _memory_cache[k]


def get_rag_cache_key(question: str, department: Optional[str], history_hash: str) -> str:
    """
    Generate cache key for RAG query.
    Department stays in PLAINTEXT so a whole department's answers can be
    purged with a single pattern. Only the volatile part (question +
    history) is hashed.
    """
    dept = (department or "all").lower()
    volatile = hashlib.md5(f"{question}:{history_hash}".encode()).hexdigest()
    return f"rag:{dept}:{volatile}"


def flush_cache():
    """Clear all cache entries."""
    _memory_cache.clear()
    print("[CACHE] In-memory cache cleared")
    return True


def cache_invalidate_department(department: str):
    """Clear all cached entries for a specific department."""
    keys_to_delete = [k for k in _memory_cache if f":{department}:" in k]
    for k in keys_to_delete:
        del _memory_cache[k]


def get_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "backend": "memory",
        "keys": len(_memory_cache),
        "max_keys": MEMORY_CACHE_MAX,
    }
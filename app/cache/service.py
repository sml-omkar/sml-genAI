"""
Cache Service
Redis-backed caching with in-memory fallback.
Reduces LLM calls by caching frequent queries.
"""

import json
import hashlib
import time
from typing import Optional, Dict, Any
from functools import lru_cache

from app.config import get_settings

settings = get_settings()

# Try Redis, fallback to in-memory
_redis_client = None
_memory_cache: Dict[str, Dict] = {}
MEMORY_CACHE_MAX = 500


def _get_redis():
    """Get Redis client, create if needed."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    
    try:
        import redis
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        _redis_client.ping()
        print("[CACHE] Redis connected")
        return _redis_client
    except Exception as e:
        print(f"[CACHE] Redis unavailable ({e}), using in-memory cache")
        _redis_client = False  # Mark as failed
        return None


def _make_key(prefix: str, *args) -> str:
    """Create a cache key from prefix and arguments."""
    raw = f"{prefix}:" + ":".join(str(a) for a in args)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(key: str) -> Optional[Dict]:
    """Get value from cache."""
    # Try Redis first
    redis = _get_redis()
    if redis:
        try:
            data = redis.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
    
    # Fallback to memory
    entry = _memory_cache.get(key)
    if entry and entry["expires"] > time.time():
        return entry["value"]
    elif entry:
        del _memory_cache[key]
    return None


def cache_set(key: str, value: Dict, ttl_seconds: int = 3600):
    """Set value in cache with TTL."""
    # Try Redis first
    redis = _get_redis()
    if redis:
        try:
            redis.setex(key, ttl_seconds, json.dumps(value, default=str))
            return
        except Exception:
            pass
    
    # Fallback to memory
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
    redis = _get_redis()
    if redis:
        try:
            keys = redis.keys(pattern)
            if keys:
                redis.delete(*keys)
        except Exception:
            pass
    
    # Clear memory cache
    keys_to_delete = [k for k in _memory_cache if pattern.replace("*", "") in k]
    for k in keys_to_delete:
        del _memory_cache[k]


def get_rag_cache_key(question: str, department: Optional[str], history_hash: str) -> str:
    """Generate cache key for RAG query."""
    return _make_key("rag", question, department or "all", history_hash)


def flush_cache():
    """Clear all cache entries."""
    redis = _get_redis()
    if redis:
        try:
            redis.flushdb()
            print("[CACHE] Redis cache flushed")
        except Exception:
            pass
    
    _memory_cache.clear()
    print("[CACHE] In-memory cache cleared")
    return True


def cache_invalidate_department(department: str):
    """Clear all cached entries for a specific department."""
    redis = _get_redis()
    if redis:
        try:
            keys = redis.keys(f"*:{department}:*")
            if keys:
                redis.delete(*keys)
                print(f"[CACHE] Invalidated {len(keys)} cache entries for dept: {department}")
        except Exception:
            pass
    
    keys_to_delete = [k for k in _memory_cache if f":{department}:" in k]
    for k in keys_to_delete:
        del _memory_cache[k]


def get_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    redis = _get_redis()
    stats = {"backend": "redis" if redis else "memory"}
    
    if redis:
        try:
            info = redis.info("stats")
            stats["hits"] = info.get("keyspace_hits", 0)
            stats["misses"] = info.get("keyspace_misses", 0)
            stats["keys"] = redis.dbsize()
        except Exception:
            stats["error"] = "Could not get Redis stats"
    else:
        stats["keys"] = len(_memory_cache)
        stats["max_keys"] = MEMORY_CACHE_MAX
    
    return stats

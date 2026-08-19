"""
Embeddings Module
Uses Ollama's nomic-embed-text for generating vector embeddings.
No HuggingFace dependency — runs entirely via Ollama API.
"""

from typing import List
from functools import lru_cache

import ollama as ollama_client

from app.config import get_settings

settings = get_settings()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Ollama.
    Uses nomic-embed-text model (768 dimensions).
    """
    response = ollama_client.embed(
        model="nomic-embed-text",
        input=texts,
    )
    return response["embeddings"]


def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single query using Ollama.
    Uses the same model as indexing for consistency.
    """
    response = ollama_client.embed(
        model="nomic-embed-text",
        input=[query],
    )
    return response["embeddings"][0]

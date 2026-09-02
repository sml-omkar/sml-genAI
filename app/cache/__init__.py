"""
Cache Module
In-memory caching for RAG responses and frequent queries.
(Redis removed — cache lives only in app process memory, so a container
restart clears it and answers always reach the LLM again after deploy.)
"""
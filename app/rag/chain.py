"""
RAG Query Chain — Conversational + Document-aware
Handles greetings naturally, answers from documents when available,
and gives friendly responses when no info is found.
"""

from typing import List, Dict, Optional
import hashlib
import re

from openai import OpenAI

from app.config import get_settings
from app.rag.vectorstore import search_similar
from app.cache.service import cache_get, cache_set, get_rag_cache_key

settings = get_settings()

FALLBACK_RESPONSE = "I don't have specific information about that in the company documents. Could you ask about something else, like our IT policies, HR procedures, or infrastructure setup?"


def _no_context_llm_call(question: str) -> str:
    """Call LLM for no-context fallback. Protected by try/except."""
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are Cyprus AI, a friendly company assistant. Be warm and conversational."},
                {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)},
            ],
            temperature=0.3,
            max_completion_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[RAG] LLM call failed: {e}")
        # Provide more helpful fallback based on error type
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            return "I'm sorry, but I'm currently unable to connect to the AI service. Please try again in a few minutes or contact IT support if the issue persists."
        elif "timeout" in str(e).lower():
            return "I'm sorry, but the AI service is taking longer than expected to respond. Please try a simpler question or try again later."
        else:
            return FALLBACK_RESPONSE


# ---- Greeting / conversational patterns (no RAG needed) ----
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings|"
    r"what'?s\s*up|how\s*(are|r)\s*(you|u)|how'?s\s*it\s*going|"
    r"thanks|thank\s*you|thx|bye|goodbye|see\s*you|"
    r"who\s*are\s*you|what\s*are\s*you|what\s*can\s*you\s*do|"
    r"help|menu|options|commands|capabilities)$",
    re.IGNORECASE,
)

GREETING_RESPONSES = {
    "hi": "Hi there! How can I help you today?",
    "hello": "Hello! I'm Cyprus AI, your company policy assistant. What can I help you with?",
    "hey": "Hey! What can I help you with?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! What can I help you with?",
    "good evening": "Good evening! How can I assist you?",
    "how are you": "I'm doing great, thanks for asking! I'm here to help with any company policy questions you have.",
    "how's it going": "All good! Ready to help you find answers about company policies, IT guidelines, or anything else you need.",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Happy to help anytime.",
    "bye": "Goodbye! Have a great day.",
    "goodbye": "See you later! Feel free to come back anytime.",
    "who are you": "I'm Cyprus AI -- your company policy assistant. I can answer questions about company policies, IT guidelines, HR procedures, and more. Just ask me anything!",
    "what are you": "I'm an AI assistant powered by RAG (Retrieval-Augmented Generation). I search through company documents and give you accurate answers based on actual policy content.",
    "what can you do": "I can help you with:\n- Company policies and procedures\n- IT guidelines and infrastructure info\n- HR policies (leave, benefits, etc.)\n- Any document-based questions\n\nJust type your question and I'll find the answer!",
    "help": "Here's what I can do:\n\n- Answer questions about company policies\n- Help with IT guidelines and procedures\n- Explain HR policies\n- Look up specific document details\n\nJust ask a question -- I'll search through company documents and give you the answer!",
    "what can you help with": "I can help with:\n- Company policies and procedures\n- IT guidelines and infrastructure info\n- HR policies (leave, benefits, etc.)\n- Any document-based questions\n\nJust type your question and I'll find the answer!",
}

DEFAULT_GREETING = "Hey! I'm Cyprus AI. Ask me anything about company policies, IT guidelines, HR procedures, or any other company documents."


SYSTEM_PROMPT = """You are Cyprus AI, a friendly and helpful company assistant. You answer employee questions using company documents.

Personality:
- Be warm, friendly, and conversational
- Give thorough, detailed answers — never just one sentence
- If the user asks to "explain" something, give a comprehensive explanation with all relevant details

Rules:
- Write a clear, detailed answer using ALL the information provided below
- Include every specific detail: numbers, names, dates, rules, sections, requirements, steps, dates, amounts, penalties, deadlines
- If multiple pieces of information are relevant, combine them into a complete, well-structured answer
- Use bullet points or numbered lists when explaining policies with multiple rules
- Never mention "context", "sources", "documents", "sections", or "numbered items"
- Never say "Page X of document Y" — just answer naturally
- Never say "based on the above" or "according to the documents"
- Just answer as if you already know the information
- If the question asks for an explanation, provide a thorough breakdown covering all key points
- If NONE of the information answers the question, say something like:
  "I don't have specific information about that in the company documents. Could you ask about something else, like our IT policies, HR procedures, or infrastructure setup?"
- You remember previous questions in this conversation — use that context

Example answers:

Q: What is the leave policy?
A: The company provides the following leave types: Annual Leave — 20 days per year, credited on January 1st. Sick Leave — 12 days per year with medical certificate required after 2 consecutive days. Maternity Leave — 26 weeks as per Indian law. Paternity Leave — 5 days. Leave must be applied through the HR portal at least 2 days in advance.

Q: What are the IT security rules?
A: The IT security policy requires: 1) All employees must use strong passwords (minimum 12 characters with special characters). 2) Two-factor authentication is mandatory for all external access. 3) VPN must be used when accessing company resources remotely. 4) USB devices are prohibited on company computers. 5) Security incidents must be reported within 2 hours to the IT Security team."""


USER_PROMPT_TEMPLATE = """{history_section}ALL context below is from ONE document: "{best_doc}"

{context}

Question: {question}

Answer using ONLY information from this single document above. Do not mix information from other sources. Provide a thorough, detailed answer with all relevant facts, rules, dates, requirements, and specifics. Use bullet points if explaining multiple items."""


NO_CONTEXT_PROMPT = """You are Cyprus AI, a friendly company assistant.

The user asked: "{question}"

You don't have specific document information about this topic.

Respond in 1-2 sentences:
- Acknowledge their question kindly
- Let them know you don't have that specific info
- Suggest they ask about something you can help with (company policies, IT guidelines, HR procedures, infrastructure)
- Be warm and helpful, not robotic

Example: "I don't have specific information about that in our company documents. I can help with things like IT policies, HR procedures, or infrastructure setup though — want to ask about any of those?" """


def _format_chat_history(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-12:]:  # Last 12 messages for better context
        role = "Employee" if msg["role"] == "user" else "Cyprus AI"
        content = msg["content"][:300]  # Truncate long answers to save context
        lines.append(f"{role}: {content}")
    lines.append("")
    return "\n".join(lines)


def format_context(chunks: List[Dict]) -> str:
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        doc = meta.get("document_name", "Unknown")
        dept = meta.get("department", "").upper()
        score = chunk.get("relevance_score", 0)
        page = meta.get("page_number", 0)
        header = f"[{i}] {doc}"
        if dept:
            header += f" | {dept}"
        if page:
            header += f" | Page {page}"
        header += f" (relevance: {score:.0%})"
        formatted.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(formatted)


def _truncate_repetition(text: str) -> str:
    lines = text.split("\n")
    seen = {}
    result = []
    for line in lines:
        normalized = line.strip().lower()
        if not normalized:
            result.append(line)
            continue
        count = seen.get(normalized, 0) + 1
        seen[normalized] = count
        if count <= 2:
            result.append(line)
        else:
            break
    return "\n".join(result).strip()


def _is_echo(answer: str) -> bool:
    lines = answer.strip().split("\n")
    header_lines = sum(1 for l in lines if any(p in l for p in ["Source:", "Section:", "Dept:", "[1]", "[2]", "[3]"]))
    return header_lines > len(lines) * 0.4 and len(lines) > 3


def _detect_greeting(question: str) -> Optional[str]:
    """Detect if the question is a greeting or conversational, return key."""
    q = question.strip().rstrip("!?.")
    if GREETING_PATTERNS.match(q):
        q_lower = q.lower()
        for key in GREETING_RESPONSES:
            if key in q_lower:
                return key
        return "default"
    return None


async def query_rag(
    question: str,
    department: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    n_results: int = None,
    min_relevance: float = 0.50,
) -> Dict:
    """
    Full RAG pipeline:
    1. Check for greetings/conversational → respond directly
    2. Retrieve chunks via hybrid search
    3. Filter and select best context
    4. Generate answer
    """
    # Check cache first
    history_hash = hashlib.md5(str(chat_history or [])[:500].encode()).hexdigest()[:8]
    cache_key = get_rag_cache_key(question, department, history_hash)
    cached = cache_get(cache_key)
    if cached:
        print(f"[RAG] Cache hit for: {question[:50]}...")
        return cached

    # Step 0: Handle greetings directly (no RAG needed)
    greeting_key = _detect_greeting(question)
    if greeting_key:
        print(f"[RAG] Greeting detected: {greeting_key}")
        if greeting_key == "default":
            answer = DEFAULT_GREETING
        else:
            answer = GREETING_RESPONSES[greeting_key]
        return {
            "answer": answer,
            "sources": [],
            "chunks_retrieved": 0,
        }

    # Step 1: Retrieve chunks — hybrid search (vector similarity + keyword boost)
    print(f"[RAG] Searching for: {question[:60]}...")
    chunks = await search_similar(
        query=question,
        department=department,
        n_results=30,
    )

    # No chunks at all → friendly no-info response
    if not chunks:
        print(f"[RAG] No chunks found for: {question[:50]}")
        return {
            "answer": _no_context_llm_call(question),
            "sources": [],
            "chunks_retrieved": 0,
        }

    # Step 2: Hard gate — best chunk must be relevant enough
    best_score = chunks[0].get("relevance_score", 0)
    print(f"[RAG] Best score: {best_score:.4f} (threshold: {min_relevance})")
    if best_score < min_relevance:
        return {
            "answer": _no_context_llm_call(question),
            "sources": [],
            "chunks_retrieved": len(chunks),
        }

    # Step 3: Filter by minimum relevance — fall back to top chunks if none pass
    relevant = [c for c in chunks if c.get("relevance_score", 0) >= min_relevance]
    if not relevant:
        print(f"[RAG] No chunks above min_relevance={min_relevance}, using top chunks")
        relevant = chunks[:5]

    # Step 4: Group by document — answer from the SINGLE best document only
    doc_groups: Dict[str, List[Dict]] = {}
    for c in relevant:
        doc = c["metadata"].get("document_name", "unknown")
        if doc not in doc_groups:
            doc_groups[doc] = []
        doc_groups[doc].append(c)

    # Rank documents by best chunk score
    doc_ranking = []
    for doc, doc_chunks in doc_groups.items():
        best = max(c.get("relevance_score", 0) for c in doc_chunks)
        avg = sum(c.get("relevance_score", 0) for c in doc_chunks) / len(doc_chunks)
        doc_ranking.append((doc, best, avg, doc_chunks))
    doc_ranking.sort(key=lambda x: x[1], reverse=True)

    if not doc_ranking:
        return {
            "answer": "I don't have specific information about that in the company documents. Could you ask about something else?",
            "sources": [],
            "chunks_retrieved": len(chunks),
        }

    # Pick the best document and take top 5 chunks from it ONLY
    best_doc_name, best_score_val, best_avg, best_chunks = doc_ranking[0]
    selected = best_chunks[:5]

    if not selected:
        return {
            "answer": _no_context_llm_call(question),
            "sources": [],
            "chunks_retrieved": len(chunks),
        }

    context = format_context(selected)
    history_section = _format_chat_history(chat_history or [])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                history_section=history_section,
                best_doc=best_doc_name,
                context=context,
                question=question,
            ),
        },
    ]

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=1024,
            top_p=0.9,
        )

        answer = response.choices[0].message.content.strip()
        answer = _truncate_repetition(answer)

        # Retry if LLM echoed context
        if _is_echo(answer):
            retry_msgs = [
                {"role": "system", "content": "You are Cyprus AI, a friendly assistant. Answer warmly in 1-3 sentences. Do NOT copy text — write your own answer."},
                {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ]
            retry = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=retry_msgs,
                temperature=0.3,
                max_completion_tokens=256,
            )
            retry_answer = retry.choices[0].message.content.strip()
            if not _is_echo(retry_answer) and len(retry_answer) > 10:
                answer = retry_answer

        # Retry if LLM said "don't know" but context has facts
        unsure = ["don't have", "not specified", "cannot determine", "not provided"]
        if any(p in answer.lower() for p in unsure):
            retry_msgs = [
                {"role": "system", "content": "You are Cyprus AI. Answer using ONLY the facts below. Extract the specific answer — do not say you don't know if the facts contain it. Be friendly."},
                {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {question}\n\nAnswer with the specific fact:"},
            ]
            retry = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=retry_msgs,
                temperature=0.0,
                max_completion_tokens=256,
            )
            retry_answer = retry.choices[0].message.content.strip()
            if not any(p in retry_answer.lower() for p in unsure) and len(retry_answer) > 10:
                answer = retry_answer

        # Sources
        seen_docs = {}
        for c in selected:
            doc = c["metadata"].get("document_name", "Unknown")
            score = c.get("relevance_score", 0)
            page = c["metadata"].get("page_number", 0)
            folder_id = c["metadata"].get("folder_id", "")
            doc_id = c["metadata"].get("document_id", "")
            if doc not in seen_docs or score > seen_docs[doc]["relevance_score"]:
                seen_docs[doc] = {
                    "document_name": doc,
                    "department": c["metadata"].get("department", "N/A"),
                    "relevance_score": round(score, 4),
                    "page_number": page,
                    "document_id": doc_id,
                    "folder_id": folder_id,
                }
        sources = list(seen_docs.values())

        result = {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(selected),
        }

        # Cache the result
        cache_set(cache_key, result, ttl_seconds=settings.CACHE_TTL_SECONDS)

        return result

    except Exception as e:
        print(f"[RAG] OpenAI error: {e}")
        return {
            "answer": "I encountered an error processing your question. Please try again.",
            "sources": [],
            "chunks_retrieved": 0,
        }

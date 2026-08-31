"""
Agentic RAG System
Uses LLM as an intelligent agent that:
1. Routes questions (greeting vs document query)
2. Rewrites queries for better retrieval
3. Evaluates search results quality
4. Re-searches if results are poor
5. Generates comprehensive answers
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import json
import re

from openai import OpenAI

from app.config import get_settings
from app.rag.vectorstore import search_similar
from app.cache.service import cache_get, cache_set, get_rag_cache_key

settings = get_settings()

# ============================================================================
# Data classes for agent state
# ============================================================================

@dataclass
class AgentStep:
    """A single step in the agent's reasoning."""
    step_type: str  # "route", "rewrite", "search", "evaluate", "generate"
    input: str
    output: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentState:
    """Tracks the agent's full reasoning process."""
    question: str
    steps: List[AgentStep] = field(default_factory=list)
    intent: str = ""
    search_queries: List[str] = field(default_factory=list)
    chunks_found: int = 0
    chunks_used: int = 0
    confidence: float = 0.0
    answer: str = ""
    sources: List[Dict] = field(default_factory=list)

    def add_step(self, step_type: str, input_text: str, output_text: str, **metadata):
        self.steps.append(AgentStep(
            step_type=step_type,
            input=input_text[:500],
            output=output_text[:500],
            metadata=metadata,
        ))

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "intent": self.intent,
            "steps": [
                {"type": s.step_type, "input": s.input, "output": s.output, **s.metadata}
                for s in self.steps
            ],
            "search_queries": self.search_queries,
            "chunks_found": self.chunks_found,
            "chunks_used": self.chunks_used,
            "confidence": self.confidence,
            "answer": self.answer,
            "sources": self.sources,
        }


# ============================================================================
# LLM Helper
# ============================================================================

_openai_client = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _llm_chat(messages: List[Dict], temperature: float = 0.3, max_tokens: int = 512) -> str:
    """Call LLM with error handling."""
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[AGENT] LLM error: {e}")
        return ""


# ============================================================================
# Step 1: Intent Router
# ============================================================================

ROUTER_SYSTEM = """You are an intent classifier for a company AI assistant called Cyprus AI.

Classify the user's question into ONE of these categories:
- GREETING: Hello, hi, thanks, bye, good morning/afternoon/evening
- CONVERSATIONAL: Jokes, small talk, opinions, general knowledge NOT related to company documents or policies. Also includes requests like "tell me a joke", "what's the weather", "sing a song"
- DOCUMENT_QUERY: Questions that need specific information FROM company documents (policies, procedures, guidelines, rules, infrastructure, setup details)
- FOLLOWUP: Questions that reference previous conversation context (e.g., "tell me more", "what about X", "and the other one")
- CLARIFICATION: Questions asking to explain something mentioned earlier

Important: If the user is asking for entertainment, opinions, or general chat that has nothing to do with company policies or documents, classify as CONVERSATIONAL.

Respond with ONLY the category name, nothing else."""

GREETING_RESPONSES = {
    "hi": "Hi there! How can I help you today?",
    "hello": "Hello! I'm Cyprus AI, your company policy assistant. What can I help you with?",
    "hey": "Hey! What can I help you with?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! What can I help you with?",
    "good evening": "Good evening! How can I assist you?",
    "how are you": "I'm doing great, thanks for asking! I'm here to help with any company policy questions you have.",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Happy to help anytime.",
    "bye": "Goodbye! Have a great day.",
    "who are you": "I'm Cyprus AI -- your company policy assistant. I can answer questions about company policies, IT guidelines, HR procedures, and more.",
    "what can you do": "I can help you with:\n- Company policies and procedures\n- IT guidelines and infrastructure info\n- HR policies (leave, benefits, etc.)\n- Any document-based questions\n\nJust type your question!",
}


async def route_intent(question: str, chat_history: Optional[List[Dict]] = None) -> str:
    """Classify user intent using LLM."""
    q_lower = question.strip().lower().rstrip("!?.")
    
    # Fast path: check common greetings without LLM
    if q_lower in GREETING_RESPONSES or len(q_lower.split()) <= 2:
        for key in GREETING_RESPONSES:
            if key in q_lower:
                return "GREETING"
    
    # Use LLM for classification
    history_context = ""
    if chat_history:
        recent = chat_history[-4:]
        history_context = "\n".join([
            f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:100]}"
            for m in recent
        ])
    
    history_prefix = "Previous conversation:\n" + history_context + "\n\n" if history_context else ""
    
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": f"{history_prefix}User question: {question}"},
    ]
    
    response = _llm_chat(messages, temperature=0.0, max_tokens=20)
    intent = response.upper().strip()
    
    # Validate
    valid_intents = {"GREETING", "CONVERSATIONAL", "DOCUMENT_QUERY", "FOLLOWUP", "CLARIFICATION"}
    if intent not in valid_intents:
        # Default to DOCUMENT_QUERY if unclear
        intent = "DOCUMENT_QUERY"
    
    return intent


# ============================================================================
# Step 2: Query Rewriter
# ============================================================================

QUERY_REWRITER_SYSTEM = """You are a search query optimizer for a company document search system.

Given a user question (and optional conversation context), generate 1-3 optimized search queries that would find the most relevant document chunks.

Rules:
- Extract key terms and concepts from the question
- Remove filler words (what is, how do, can you tell me about)
- Include synonyms or related terms
- For follow-up questions, incorporate context from the conversation
- Keep queries concise (5-15 words each)

Respond with ONLY a JSON array of search queries, no explanation.
Example: ["leave policy annual leave", "vacation days entitlement"]"""


async def rewrite_queries(
    question: str,
    chat_history: Optional[List[Dict]] = None,
) -> List[str]:
    """Generate optimized search queries using LLM."""
    history_context = ""
    if chat_history:
        recent = chat_history[-6:]
        history_context = "\n".join([
            f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:150]}"
            for m in recent
        ])
    
    history_prefix = "Conversation context:\n" + history_context + "\n\n" if history_context else ""
    
    messages = [
        {"role": "system", "content": QUERY_REWRITER_SYSTEM},
        {"role": "user", "content": f"{history_prefix}User question: {question}"},
    ]
    
    response = _llm_chat(messages, temperature=0.2, max_tokens=200)
    
    # Parse JSON array from response
    try:
        # Extract JSON from response (may have extra text)
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group())
            if isinstance(queries, list) and len(queries) > 0:
                return [str(q) for q in queries[:3]]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Fallback: use the original question as the search query
    return [question]


# ============================================================================
# Step 3: Retriever Agent
# ============================================================================

RELEVANCE_EVALUATOR = """You are a relevance evaluator for document search results.

Given a user question and a list of document chunks, evaluate if the chunks contain enough information to answer the question.

Rate the overall relevance:
- HIGH: Chunks directly answer the question with specific facts
- MEDIUM: Chunks are related and contain some useful information
- LOW: Chunks are only tangentially related
- NONE: Chunks don't contain relevant information

Respond with ONLY a JSON object: {"relevance": "HIGH|MEDIUM|LOW|NONE", "reason": "brief explanation"}"""


async def evaluate_chunks(
    question: str,
    chunks: List[Dict],
) -> Tuple[str, str]:
    """Evaluate if chunks are relevant enough to answer the question."""
    if not chunks:
        return "NONE", "No chunks found"
    
    # Build a summary of chunks for evaluation
    chunk_summary = "\n".join([
        f"[{i+1}] {c['text'][:200]}..."
        for i, c in enumerate(chunks[:5])
    ])
    
    messages = [
        {"role": "system", "content": RELEVANCE_EVALUATOR},
        {"role": "user", "content": f"Question: {question}\n\nChunks:\n{chunk_summary}"},
    ]
    
    response = _llm_chat(messages, temperature=0.0, max_tokens=150)
    
    try:
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("relevance", "MEDIUM"), result.get("reason", "")
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Fallback: use score-based evaluation
    best_score = max(c.get("relevance_score", 0) for c in chunks)
    if best_score >= 0.7:
        return "HIGH", f"Best score: {best_score:.2f}"
    elif best_score >= 0.5:
        return "MEDIUM", f"Best score: {best_score:.2f}"
    else:
        return "LOW", f"Best score: {best_score:.2f}"


# ============================================================================
# Step 4: Answer Generator
# ============================================================================

GENERATOR_SYSTEM = """You are Cyprus AI, a friendly and helpful company assistant. You answer employee questions using company documents.

Two-step process — ALWAYS do this in order:
STEP 1 — ANALYZE (think, don't write):
  Before answering, read the retrieved facts and the user's question together.
  Identify exactly what the user is asking and which facts answer it.
  Only use information actually present in the facts.
STEP 2 — ANSWER:
  Then give a direct answer that directly addresses the question, with the depth the user requested.

Rules:
- Default style: SHORT and DIRECT — answer the question in 2-4 sentences
- If the user explicitly asks to "explain", "explain fully", "in detail", "elaborate", "brief me about", "give a brief about", or wants a comprehensive explanation, THEN give a long, thorough, well-structured answer covering all the relevant facts.
- Do NOT dump every fact unless the user asked for a full explanation — answer exactly what was asked
- Be direct and to the point; no fluff, no repetition
- Only use bullet points if a policy genuinely has multiple distinct rules the user asked about
- Include the key specific detail (number, deadline, rule) that answers the question
- NEVER mention "context", "sources", "documents", "sections", or "numbered items"
- NEVER start sentences with "According to...", "As per...", "Based on...", "The policy states..."
- NEVER name any policy, PDF, or file — just state the facts directly as if you know them
- You remember previous questions in this conversation — use that context
- If NONE of the information answers the question, say exactly:
  "I don't have specific information about that in the company documents. Could you ask about something else, like our IT policies, HR procedures, or infrastructure setup?"

Examples of SHORT & DIRECT:
Q: How often must I change my password?
A: Every 45 days. Create a strong, complex password, enable MFA, and do not reuse old passwords.

Q: Who approves AI licenses?
A: IT calculates a combined score from two parts — operational need (max 55) and technical/security (max 45). If the total is 65 or above, the application is approved.

Q: What is the leave policy?
A: You get 20 days annual leave, 12 sick days a year, 26 weeks maternity, and 5 days paternity. Apply at least 2 days in advance through the HR portal.

Example when the user asks to EXPLAIN in detail (note the long, comprehensive answer):
Q: Explain the password policy in detail
A: The password policy requires you to change your password every 45 days. Your password must be strong and complex, combining upper and lower case letters, numbers, and special characters, and should not be reused across systems or from your old passwords. You must also enable Multi-Factor Authentication (MFA) on your account for an extra layer of security. If you forget your password, you can reset it through the IT helpdesk, and your new password must be different from your previous ones to keep your account secure."""


# Strips source-referencing lead-ins the small model tends to produce,
# e.g. "According to the Password Management Policy document, you must..."
def _requests_detail(question: str) -> bool:
    """Detect if the user explicitly asked for a detailed/comprehensive explanation."""
    q = question.lower().strip()
    return bool(re.search(
        r"(explain|elaborate|in detail|full(ly)? (explain|detail|break|detail)|"
        r"brief me|brief about|overview of|summariz|describe|tell me more|"
        r"more about|go into detail|walk me through|what are the (rules|steps|guidelines)|"
        r"how does it work|give me a (full|brief|detailed)|in depth)",
        q,
    ))


GENERATOR_DETAIL_SYSTEM = """You are Cyprus AI, a friendly and helpful company assistant. You answer employee questions using company documents.

The employee asked for a FULL, COMPREHENSIVE explanation, so give a detailed and well-structured answer.

Rules:
- Be thorough: cover ALL relevant facts present in the context that relate to the question
- Use clear structure — short paragraphs and/or bullet points where appropriate
- Explain the reasoning and context behind the rules, not just the raw numbers
- Keep the specific numbers, deadlines, and rules intact and accurate
- NEVER mention "context", "sources", "documents", "sections", or "numbered items"
- NEVER start sentences with "According to...", "As per...", "Based on...", "The policy states..."
- NEVER name any policy, PDF, or file — just state the facts directly as if you know them
- If NONE of the information answers the question, say exactly:
  "I don't have specific information about that in the company documents. Could you ask about something else, like our IT policies, HR procedures, or infrastructure setup?"  """


_SOURCE_LEADIN = re.compile(
    r"^\s*(?:according to|as per|per|based on|from)\s+(?:the\s+)?[\"']?.{0,80}?"
    r"(?:polic(?:y|ies)|document|doc|pdf|guideline[s]?|manual|handbook|file)[\"']?\s*,?\s*",
    re.IGNORECASE,
)


def _clean_answer(answer: str) -> str:
    """Remove 'According to <some> policy/document,' style lead-ins."""
    cleaned = _SOURCE_LEADIN.sub("", answer.strip(), count=1)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def format_chunks_for_llm(chunks: List[Dict]) -> str:
    """Format chunks into context string for the LLM."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        doc = meta.get("document_name", "Unknown")
        dept = meta.get("department", "").upper()
        page = meta.get("page_number", 0)
        score = chunk.get("relevance_score", 0)
        
        header = f"[{i}] {doc}"
        if dept:
            header += f" | {dept}"
        if page:
            header += f" | Page {page}"
        header += f" (relevance: {score:.0%})"
        formatted.append(f"{header}\n{chunk['text']}")
    
    return "\n\n".join(formatted)


def format_history(history: List[Dict]) -> str:
    """Format conversation history for context."""
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-10:]:
        role = "Employee" if msg["role"] == "user" else "Cyprus AI"
        content = msg["content"][:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_memories(memories: Dict[str, Dict]) -> str:
    """
    Format pre-built policy memories into a compact knowledge block.
    memories: {document_id: {"summary", "key_facts", "document_name"}}
    """
    if not memories:
        return ""
    blocks = []
    for mem in memories.values():
        lines = []
        name = mem.get("document_name", "Unknown document")
        summary = (mem.get("summary") or "").strip()
        facts = mem.get("key_facts") or []
        if summary:
            lines.append(f"About '{name}': {summary}")
        if facts:
            lines.append(f"Key rules from '{name}':")
            lines.extend(f"- {f}" for f in facts[:15])
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def generate_answer(
    question: str,
    chunks: List[Dict],
    chat_history: Optional[List[Dict]] = None,
    memories: Optional[Dict[str, Dict]] = None,
) -> str:
    """Generate answer using retrieved chunks + pre-built policy memory."""
    context = format_chunks_for_llm(chunks)
    history_text = format_history(chat_history or [])
    history_prefix = history_text + "\n\n" if history_text else ""

    memory_text = format_memories(memories or {})
    memory_block = f"""Distilled knowledge about these policies (verified notes from reading the full documents):

{memory_text}

""" if memory_text else ""

    user_content = f"""{history_prefix}{memory_block}Context from company documents:

{context}

User question: {question}

First analyze the context and the question together to identify the direct answer, then respond short and direct."""
    
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    if _requests_detail(question):
        messages[0]["content"] = GENERATOR_DETAIL_SYSTEM

    answer = _llm_chat(messages, temperature=0.3, max_tokens=1024)
    
    if not answer:
        return "I encountered an error generating a response. Please try again."
    
    # Check for echo (LLM just copying context)
    lines = answer.strip().split("\n")
    header_lines = sum(1 for l in lines if any(p in l for p in ["Source:", "[1]", "[2]", "[3]"]))
    if header_lines > len(lines) * 0.4 and len(lines) > 3:
        # Retry with simpler prompt
        retry_messages = [
            {"role": "system", "content": "Answer the question using the facts provided. Write naturally, don't copy."},
            {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ]
        retry_answer = _llm_chat(retry_messages, temperature=0.3, max_tokens=512)
        if retry_answer and len(retry_answer) > 20:
            answer = retry_answer
    
    return _clean_answer(answer)


# ============================================================================
# History-based answering (follow-ups like "in short", "what did I ask?")
# ============================================================================

HISTORY_ANSWER_SYSTEM = """You are Cyprus AI, a friendly company assistant.

The employee is referring to your EARLIER REPLY in this conversation — for example asking you to
summarize it, shorten it, repeat it, explain a part of it, or tell them what they asked.

Use ONLY the previous conversation below.
- If the request can be fulfilled from what was already said, do it faithfully (keep all facts)
- Answer naturally and directly — never mention "the conversation" or "as I said above"
- If the earlier replies do NOT contain enough information, respond with exactly:
NO_ANSWER_IN_HISTORY"""


async def _answer_from_history(
    question: str,
    chat_history: List[Dict],
) -> Optional[str]:
    """
    Try to answer purely from conversation history — handles follow-ups that
    reference previous answers ("give it in short", "repeat that", etc.).
    Returns None if the history doesn't contain the answer.
    """
    recent = chat_history[-6:]
    convo = "\n".join(
        f"{'Employee' if m['role'] == 'user' else 'Cyprus AI'}: {m['content'][:1200]}"
        for m in recent
    )
    messages = [
        {"role": "system", "content": HISTORY_ANSWER_SYSTEM},
        {"role": "user", "content": f"Previous conversation:\n{convo}\n\nEmployee now asks: {question}"},
    ]
    answer = _llm_chat(messages, temperature=0.3, max_tokens=700)
    if not answer or answer.strip().startswith("NO_ANSWER_IN_HISTORY"):
        return None
    return _clean_answer(answer)


# ============================================================================
# Main Agent
# ============================================================================

async def query_rag(
    question: str,
    department: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    n_results: int = None,
    min_relevance: float = 0.50,
    debug: bool = False,
) -> Dict:
    """
    Agentic RAG pipeline:
    1. Route intent (greeting vs document query)
    2. Rewrite query for better search
    3. Search and evaluate chunks
    4. Re-search if needed (max 2 rounds)
    5. Generate comprehensive answer
    """
    state = AgentState(question=question)
    
    # ---- Check cache first ----
    history_hash = hashlib.md5(str(chat_history or [])[:500].encode()).hexdigest()[:8]
    cache_key = get_rag_cache_key(question, department, history_hash)
    cached = cache_get(cache_key)
    if cached:
        state.add_step("cache", question, "Cache hit", hit=True)
        state.answer = cached.get("answer", "")
        state.sources = cached.get("sources", [])
        if debug:
            cached["debug"] = state.to_dict()
        return cached
    
    # ---- Step 1: Route Intent ----
    state.add_step("route", question, "Classifying intent...")
    intent = await route_intent(question, chat_history)
    state.intent = intent
    state.add_step("route", question, f"Intent: {intent}", intent=intent)
    print(f"[AGENT] Intent: {intent}")
    
    # ---- Handle Greeting ----
    if intent == "GREETING":
        q_lower = question.strip().lower().rstrip("!?.")
        answer = "Hey! I'm Cyprus AI. Ask me anything about company policies, IT guidelines, HR procedures, or any other company documents."
        for key, response in GREETING_RESPONSES.items():
            if key in q_lower:
                answer = response
                break
        state.answer = answer
        state.add_step("generate", question, answer)
        result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
        cache_set(cache_key, result, ttl_seconds=settings.CACHE_TTL_SECONDS)
        if debug:
            result["debug"] = state.to_dict()
        return result
    
    # ---- Handle Conversational (no docs needed) ----
    if intent == "CONVERSATIONAL":
        messages = [
            {"role": "system", "content": "You are Cyprus AI, a friendly company assistant. Answer conversationally in 1-3 sentences."},
            {"role": "user", "content": question},
        ]
        answer = _llm_chat(messages, temperature=0.7, max_tokens=200)
        state.answer = answer
        state.add_step("generate", question, answer)
        result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
        cache_set(cache_key, result, ttl_seconds=settings.CACHE_TTL_SECONDS)
        if debug:
            result["debug"] = state.to_dict()
        return result
    
    # ---- Step 2: Rewrite Queries ----
    state.add_step("rewrite", question, "Generating search queries...")
    search_queries = await rewrite_queries(question, chat_history)
    state.search_queries = search_queries
    state.add_step("rewrite", question, f"Queries: {search_queries}", queries=search_queries)
    print(f"[AGENT] Search queries: {search_queries}")
    
    # ---- Step 3: Search and Evaluate (max 2 rounds) ----
    all_chunks = []
    best_relevance = "NONE"
    
    for round_num, query in enumerate(search_queries[:2], 1):
        state.add_step("search", query, f"Round {round_num}: Searching...")
        
        chunks = await search_similar(
            query=query,
            department=department,
            n_results=15,
        )
        
        state.chunks_found = len(chunks)
        state.add_step("search", query, f"Found {len(chunks)} chunks", count=len(chunks))
        
        if not chunks:
            continue
        
        # Evaluate relevance
        relevance, reason = await evaluate_chunks(question, chunks)
        state.add_step("evaluate", query, f"Relevance: {relevance} — {reason}", relevance=relevance, reason=reason)
        print(f"[AGENT] Round {round_num} — Relevance: {relevance} — {reason}")
        
        if relevance in ("HIGH", "MEDIUM"):
            all_chunks = chunks
            best_relevance = relevance
            break
        elif relevance == "LOW" and round_num == 1:
            # Try a different query
            continue
        else:
            # Use what we have
            if not all_chunks:
                all_chunks = chunks
                best_relevance = relevance
    
    # ---- Step 4: Select Best Chunks ----
    if all_chunks:
        # Filter by minimum relevance score
        relevant = [c for c in all_chunks if c.get("relevance_score", 0) >= min_relevance]
        if not relevant:
            relevant = all_chunks[:5]

        # Take top 5 chunks
        selected = relevant[:5]
        state.chunks_used = len(selected)
        state.confidence = max(c.get("relevance_score", 0) for c in selected)
    else:
        selected = []
        state.confidence = 0.0

    # ---- Step 4b: Load policy memories for matched documents ----
    memories: Dict[str, Dict] = {}
    doc_ids = list({
        c["metadata"].get("document_id")
        for c in selected
        if c["metadata"].get("document_id")
    })
    if doc_ids:
        try:
            from app.rag.policy_memory import get_memories_for_documents
            memories = await get_memories_for_documents(doc_ids)
            if memories:
                state.add_step("memory", str(doc_ids), f"Loaded {len(memories)} policy memor(ies)")
                print(f"[AGENT] Loaded {len(memories)} policy memor(ies) for context")
        except Exception as e:
            print(f"[AGENT] Memory lookup failed (continuing): {e}")

    # ---- Step 5: Generate Answer ----
    if not selected:
        # No relevant chunks — try answering from conversation history first
        # (handles follow-ups like "give this in short", "what did I ask?")
        history_answer = await _answer_from_history(question, chat_history) if chat_history else None
        if history_answer:
            answer = history_answer
            state.add_step("generate", question, f"Answered from conversation history")
            result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
        else:
            answer = "I don't have specific information about that in the company documents. Could you ask about something else, like our IT policies, HR procedures, or infrastructure setup?"
            state.add_step("generate", question, answer)
            result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
    else:
        state.add_step("generate", question, f"Generating from {len(selected)} chunks...")
        answer = await generate_answer(question, selected, chat_history, memories=memories)
        state.answer = answer
        
        # Extract sources
        seen_docs = {}
        for c in selected:
            doc = c["metadata"].get("document_name", "Unknown")
            score = c.get("relevance_score", 0)
            page = c["metadata"].get("page_number", 0)
            if doc not in seen_docs or score > seen_docs[doc]["relevance_score"]:
                seen_docs[doc] = {
                    "document_name": doc,
                    "department": c["metadata"].get("department", "N/A"),
                    "relevance_score": round(score, 4),
                    "page_number": page,
                    "document_id": c["metadata"].get("document_id", ""),
                    "folder_id": c["metadata"].get("folder_id", ""),
                }
        state.sources = list(seen_docs.values())
        
        result = {
            "answer": answer,
            "sources": state.sources,
            "chunks_retrieved": len(selected),
            "memory_used": bool(memories),
        }
    
    # ---- Cache Result ----
    cache_set(cache_key, result, ttl_seconds=settings.CACHE_TTL_SECONDS)
    
    # ---- Add Debug Info ----
    if debug:
        result["debug"] = state.to_dict()
    
    print(f"[AGENT] Done — Confidence: {state.confidence:.2f}, Sources: {len(state.sources)}")
    return result

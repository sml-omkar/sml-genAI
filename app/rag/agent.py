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

OUT_OF_SCOPE_MESSAGE = (
    "I don't have specific information about that in the company documents. "
    "Could you ask about something else, like our IT policies, HR procedures, or infrastructure setup?"
)

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


# ---------------------------------------------------------------------------
# Token usage tracking
# A pipeline run accumulates usage across all its LLM calls. query_rag resets
# this before starting and reads it at the end so the caller can attribute the
# total token cost to the requesting user.
# ---------------------------------------------------------------------------
_token_usage = {"prompt": 0, "completion": 0, "calls": 0}


def _reset_usage():
    _token_usage["prompt"] = 0
    _token_usage["completion"] = 0
    _token_usage["calls"] = 0


def get_usage() -> dict:
    """Return accumulated token usage for the current pipeline run."""
    return {
        "prompt_tokens": _token_usage["prompt"],
        "completion_tokens": _token_usage["completion"],
        "total_tokens": _token_usage["prompt"] + _token_usage["completion"],
        "llm_calls": _token_usage["calls"],
    }


def _llm_chat(messages: List[Dict], temperature: float = 0.3, max_tokens: int = 512) -> str:
    """Call LLM with error handling. Accumulates token usage for this run."""
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        # Record usage if reported by the provider
        if getattr(response, "usage", None):
            _token_usage["prompt"] += response.usage.prompt_tokens or 0
            _token_usage["completion"] += response.usage.completion_tokens or 0
        _token_usage["calls"] += 1
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[AGENT] LLM error: {e}")
        return ""


# ============================================================================
# Step 1: Intent Router
# ============================================================================

ROUTER_SYSTEM = """You are an intent classifier for a company AI assistant called EthosAI.

Classify the user's question into ONE of these categories:
- GREETING: Hello, hi, thanks, bye, good morning/afternoon/evening
- CONVERSATIONAL: ONLY pure social chatter with no factual content — greetings-like pleasantries, jokes, small talk, "how are you", "sing a song". Use ONLY when the user is just being friendly.
- DOCUMENT_QUERY: ANY question that asks for information, facts, or procedures — whether or not it is about company material. This includes general-knowledge questions like "who won the world cup", "what is the capital of France", and anything about policies, rules, or how to do something.
- FOLLOWUP: Questions that reference previous conversation context (e.g., "tell me more", "what about X", "and the other one")
- CLARIFICATION: Questions asking to explain something mentioned earlier

Important rules:
- If the user is asking for any factual information or general knowledge (news, sports, geography, science, etc.), classify as DOCUMENT_QUERY, NEVER as CONVERSATIONAL.
- Only use CONVERSATIONAL when there is NO request for information at all — pure social pleasantry or humour.
- Only use GREETING for a plain greeting with nothing else asked.

Respond with ONLY the category name, nothing else."""

GREETING_RESPONSES = {
    "hi": "Hi there! How can I help you today?",
    "hello": "Hello! I'm EthosAI, your company policy assistant. What can I help you with?",
    "hey": "Hey! What can I help you with?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! What can I help you with?",
    "good evening": "Good evening! How can I assist you?",
    "how are you": "I'm doing great, thanks for asking! I'm here to help with any company policy questions you have.",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Happy to help anytime.",
    "bye": "Goodbye! Have a great day.",
    "who are you": "I'm EthosAI -- your company policy assistant. I can answer questions about company policies, IT guidelines, HR procedures, and more.",
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

    # Keyword fast path: anything clearly about company documents/policies is a
    # document query WITHOUT calling the LLM. The LLM router was misclassifying
    # policy questions like "tell me something about ai policy" as
    # CONVERSATIONAL, so the bot answered with a generic redirect instead of
    # searching the documents. False positives here are harmless: the document
    # pipeline only ever answers from company docs and refuses otherwise.
    _doc_keywords = {
        "policy", "policies", "guideline", "guidelines", "procedure",
        "procedures", "rule", "rules", "regulation", "regulations",
        "leave", "benefit", "benefits", "holiday", "vacation", "allowance",
        "insurance", "safety", "security", "performance", "appraisal",
        "attendance", "overtime", "nda", "manual", "handbook", "compliance",
        "onboarding", "payroll", "salary", "discipline", "grievance",
        "hr", "human", "ai", "kb", "process", "document", "documents",
        "maternity", "paternity", "sick", "half", "termination",
    }
    if any(w in _doc_keywords for w in re.findall(r'[a-z0-9]+', q_lower)):
        return "DOCUMENT_QUERY"

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
    """
    Decide if retrieved chunks actually answer the question.
    Cost-aware: uses the cheap numeric relevance score as the primary signal and
    only asks the LLM when the score is borderline, so we save a full LLM call in
    the common (clearly relevant / clearly irrelevant) cases.
    """
    if not chunks:
        return "NONE", "No chunks found"

    best_score = max(c.get("relevance_score", 0) for c in chunks)

    # Clear-cut cases decided by the free numeric score (no LLM call needed).
    if best_score >= 0.75:
        return "HIGH", f"Best score: {best_score:.2f}"
    if best_score < 0.45:
        return "LOW", f"Best score: {best_score:.2f}"

    # Borderline (0.45 - 0.75): ask the LLM to confirm, keeping input small.
    chunk_summary = "\n".join([
        f"[{i+1}] {c['text'][:120]}..."
        for i, c in enumerate(chunks[:4])
    ])

    messages = [
        {"role": "system", "content": RELEVANCE_EVALUATOR},
        {"role": "user", "content": f"Question: {question}\n\nChunks:\n{chunk_summary}"},
    ]

    response = _llm_chat(messages, temperature=0.0, max_tokens=40)

    try:
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("relevance", "MEDIUM"), result.get("reason", "")[:120]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to numeric score if the LLM response is unparseable.
    if best_score >= 0.6:
        return "MEDIUM", f"Score fallback: {best_score:.2f}"
    return "LOW", f"Score fallback: {best_score:.2f}"


# ============================================================================
# Step 4: Answer Generator
# ============================================================================

GENERATOR_SYSTEM = """You are EthosAI, an employee-help assistant for SML. You answer questions using ONLY the company-document context given below — nothing else. Everything you say must be grounded in that context.

DECIDE, then ANSWER:
Step A — Are the provided facts enough to answer?
  - Facts answer the question directly  -> proceed to Step B.
  - Facts partly answer, but something is unclear (e.g. which policy, which rule, which person) -> reply asking ONE concise clarifying question to pin it down. Start with "[NEED_CLARIFICATION]" then your brief question.
  - Facts are unrelated or absent      -> reply that you don't have that information and ask what they'd like to know about.
Step B — What depth does the user want?
  Choose naturally from the question's wording: a direct factual question -> short & direct; a request for steps/procedure/how -> numbered steps; a request to explain/elaborate/brief -> thorough, structured detail; a list question -> bullets. Match, don't over-explain, don't under-explain.
Step C — Write the answer in your own words using only the facts.

Constraints:
- Never invent details that are not in the context (no outside knowledge, no guessing, no assumptions).
- Never fabricate numbers, dates, or rules.
- Keep the exact figures/deadlines from the context when they answer the question.
- Do not mention "context", "sources", "documents", "sections", or "retrieved".
- Do not open with "According to...", "As per...", "Based on...", "The policy states...".
- Do not name any policy file or PDF — state facts as if you know them.
- Use earlier turns in the conversation where relevant.

Examples:
Q: How often must I change my password?
A: Every 45 days. Create a strong, complex password, enable MFA, and do not reuse old passwords.

Q: How do I request leave?
A: To request leave:
1. Log in to the HR portal.
2. Click 'Leave Request' and choose the type of leave.
3. Select the dates and add a reason.
4. Submit at least 2 days in advance; your manager approves it in the portal.

Q: Do I need approval for a work-from-home day?
(If the context only covers leave, not WFH)
A: [NEED_CLARIFICATION] The documents cover the leave request process but I don't see a work-from-home policy. Did you mean requesting a leave day, or is there a separate WFH guideline you're referring to?"""


# Strips source-referencing lead-ins the small model tends to produce,
# e.g. "According to the Password Management Policy document, you must..."
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


# Flag we use when the generator tells us it cannot answer from the context.
# Kept as a sentinel so downstream code can detect "please clarify" replies.
CLARITY_SENTINEL = "[NEED_CLARIFICATION]"


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
        role = "Employee" if msg["role"] == "user" else "EthosAI"
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

Follow the instructions in your system prompt. Decide the appropriate response (direct answer, numbered steps, detailed explanation, or a clarifying question)."""

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    answer = _llm_chat(messages, temperature=0.3, max_tokens=700)

    if not answer:
        return "I encountered an error generating a response. Please try again."

    # If the model asked for clarification, keep the sentinel marker so the
    # caller (query_rag) can distinguish it from a normal answer / refusal.
    if CLARITY_SENTINEL in answer:
        return answer.strip()

    # Check for echo (LLM just copying context)
    lines = answer.strip().split("\n")
    header_lines = sum(1 for l in lines if any(p in l for p in ["Source:", "[1]", "[2]", "[3]"]))
    if header_lines > len(lines) * 0.4 and len(lines) > 3:
        # Retry with simpler prompt
        retry_messages = [
            {"role": "system", "content": "Answer the question using ONLY the facts provided. Write naturally, don't copy."},
            {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ]
        retry_answer = _llm_chat(retry_messages, temperature=0.3, max_tokens=512)
        if retry_answer and len(retry_answer) > 20:
            answer = retry_answer

    return _clean_answer(answer)


# ============================================================================
# History-based answering (follow-ups like "in short", "what did I ask?")
# ============================================================================

HISTORY_ANSWER_SYSTEM = """You are EthosAI, a friendly company assistant.

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
        f"{'Employee' if m['role'] == 'user' else 'EthosAI'}: {m['content'][:1200]}"
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
    folder_ids: Optional[List[str]] = None,
    chat_history: Optional[List[Dict]] = None,
    n_results: int = None,
    min_relevance: float = 0.50,
    debug: bool = False,
    include_usage: bool = False,
) -> Dict:
    """
    Agentic RAG pipeline. Public entry point; tracks token usage across the run.
    When ``include_usage`` is True, the returned dict includes a "usage" key with
    prompt/completion/total tokens consumed for this request.
    folder_ids, when provided, scopes retrieval to explicit folders (external APIs).
    """
    _reset_usage()
    result = await _query_rag_impl(
        question=question,
        department=department,
        folder_ids=folder_ids,
        chat_history=chat_history,
        n_results=n_results,
        min_relevance=min_relevance,
        debug=debug,
    )
    if include_usage:
        result["usage"] = get_usage()
    return result


async def _query_rag_impl(
    question: str,
    department: Optional[str] = None,
    folder_ids: Optional[List[str]] = None,
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
    # A repeated question means the user wasn't satisfied with the previous
    # answer — never serve a cached reply for it, always regenerate from the LLM.
    is_repeat_question = False
    if chat_history:
        prev_user_msgs = [m for m in chat_history if m.get("role") == "user"]
        if prev_user_msgs:
            prev = prev_user_msgs[-1].get("content", "").strip().lower()
            is_repeat_question = prev == question.strip().lower()
    if is_repeat_question:
        print(f"[AGENT] Repeat question detected — bypassing cache, regenerating answer")

    history_hash = hashlib.md5(str(chat_history or [])[:500].encode()).hexdigest()[:8]
    cache_key = get_rag_cache_key(question, department, history_hash)
    cached = cache_get(cache_key)
    if cached and not is_repeat_question:
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
        answer = "Hey! I'm EthosAI. Ask me anything about company policies, IT guidelines, HR procedures, or any other company documents."
        for key, response in GREETING_RESPONSES.items():
            if key in q_lower:
                answer = response
                break
        state.answer = answer
        state.add_step("generate", question, answer)
        result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
        if debug:
            result["debug"] = state.to_dict()
        return result
    
    # ---- Handle Conversational (no docs needed) ----
    # IMPORTANT: EthosAI must only answer from company documents. For general
    # chit-chat we stay friendly but NEVER provide factual/world-knowledge
    # answers (that leaks information outside our documents). We either keep it
    # to neutral pleasantries or politely redirect to company topics.
    if intent == "CONVERSATIONAL":
        messages = [
            {"role": "system", "content": (
                "You are EthosAI, SML's company assistant. You ONLY answer from company "
                "documents, so you must NEVER provide general knowledge or factual answers about "
                "the outside world (news, sports, geography, celebrities, etc.).\n"
                "When someone says a simple greeting or pleasantry (hi, thanks, how are you), "
                "respond briefly and warmly and offer to help with company policies.\n"
                "When someone asks a factual or general-knowledge question that is NOT about "
                "company documents, do NOT answer it. Instead say you're here to help with "
                "company policies, and ask what they'd like to look up.\n"
                "Keep replies to 1-2 sentences."
            )},
            {"role": "user", "content": question},
        ]
        answer = _llm_chat(messages, temperature=0.4, max_tokens=200)
        state.answer = answer
        state.add_step("generate", question, answer)
        result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
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
            folder_ids=folder_ids,
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
    # Strict scope enforcement: only retrieve chunks that genuinely meet the
    # relevance threshold. We do NOT force irrelevant chunks through to the
    # generator — that is what let the bot answer questions outside our documents.
    if all_chunks:
        relevant = [c for c in all_chunks if c.get("relevance_score", 0) >= min_relevance]

        # Take top 5 chunks
        selected = relevant[:5]
        state.chunks_used = len(selected)
        state.confidence = max(c.get("relevance_score", 0) for c in selected) if selected else 0.0
    else:
        selected = []
        state.confidence = 0.0

    # ---- Step 4a: Handle evaluator LOW/NONE verdict ----
    # If retrieval pulled back something but the evaluator judged it not actually
    # relevant (LOW/NONE), discard it so we don't answer from unrelated content.
    if selected and best_relevance in ("LOW", "NONE"):
        print(f"[AGENT] Relevance judged '{best_relevance}' — discarding chunks, treating as out of scope")
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
        # No in-scope document chunks. First try answering purely from previous
        # conversation (only for legitimate follow-ups like "explain that" /
        # "in short" / "what did I ask?"). _answer_from_history itself refuses
        # (returns None) when the earlier replies do not contain the answer.
        if chat_history:
            history_answer = await _answer_from_history(question, chat_history)
            if history_answer:
                answer = history_answer
                state.add_step("generate", question, "Answered from conversation history")
                result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
                if debug:
                    result["debug"] = state.to_dict()
                return result

        # Otherwise the question is outside our documents — refuse rather than
        # answer from general knowledge.
        answer = OUT_OF_SCOPE_MESSAGE
        state.add_step("generate", question, answer)
        result = {"answer": answer, "sources": [], "chunks_retrieved": 0}
    else:
        state.add_step("generate", question, f"Generating from {len(selected)} chunks...")
        answer = await generate_answer(question, selected, chat_history, memories=memories)

        # Strip the internal clarification marker before showing the user.
        if answer.startswith(CLARITY_SENTINEL):
            answer = answer[len(CLARITY_SENTINEL):].strip(" :\n")

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
    # Only cache genuine document-sourced answers. Never cache the canned
    # out-of-scope/fallback reply, greetings, or errors — otherwise users see
    # the same hardcoded answer repeated until the TTL expires.
    if state.sources:
        cache_set(cache_key, result, ttl_seconds=settings.CACHE_TTL_SECONDS)
    
    # ---- Add Debug Info ----
    if debug:
        result["debug"] = state.to_dict()
    
    print(f"[AGENT] Done — Confidence: {state.confidence:.2f}, Sources: {len(state.sources)}")
    return result

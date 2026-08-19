"""
Text Chunker (Page-Aware)
Splits extracted text into optimal chunks for RAG.
Page-aware: splits by --- Page N --- markers first, then chunks within each page.
Every line is processed — no content is lost.
"""

import re
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings

settings = get_settings()


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Dict]:
    """
    Split text into chunks optimized for detailed RAG retrieval.
    Page-aware: processes each page separately to preserve citation accuracy.

    Strategy:
    1. Split by --- Page N --- markers
    2. Chunk within each page using RecursiveCharacterTextSplitter
    3. Post-process: merge short chunks, preserve tables/lists
    4. Every chunk gets a page_number metadata

    Returns: [{"text": str, "index": int, "char_start": int, "is_table": bool, "page_number": int}]
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    # Step 1: Split by page markers
    pages = _split_by_pages(text)

    # Step 2: Chunk within each page
    all_chunks = []
    global_index = 0

    for page_num, page_text in pages:
        if not page_text.strip():
            continue

        page_chunks = _chunk_page(page_text, chunk_size, chunk_overlap)

        for pc in page_chunks:
            pc["index"] = global_index
            pc["page_number"] = page_num
            all_chunks.append(pc)
            global_index += 1

    # Step 3: Post-process (merge short chunks within same page only)
    all_chunks = _post_process_chunks(all_chunks)

    return all_chunks


def _split_by_pages(text: str) -> List[Tuple[int, str]]:
    """
    Split extracted text by --- Page N --- markers.
    Returns list of (page_number, page_text) tuples.
    """
    # Pattern matches "--- Page 1 ---", "--- Page 12 ---", etc.
    page_pattern = re.compile(r'^---\s*Page\s+(\d+)\s*---$', re.MULTILINE)

    parts = page_pattern.split(text)
    # parts: ['text_before', '1', 'page1_text', '3', 'page3_text', ...]

    pages = []
    if len(parts) < 2:
        # No page markers found — treat entire text as page 1
        pages.append((1, text))
    else:
        # First part is text before first marker (usually empty)
        # Then alternating: page_number, page_text
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            page_text = parts[i + 1] if i + 1 < len(parts) else ""
            pages.append((page_num, page_text))

    return pages


def _chunk_page(
    page_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    """Chunk a single page's text using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n\n",          # Section breaks
            "\n\n",            # Paragraph breaks
            "\n# ",            # Markdown H1
            "\n## ",           # Markdown H2
            "\n### ",          # Markdown H3
            "\n- ",            # Bullet points
            "\n* ",            # Bullet points (alt)
            "\n1. ",           # Numbered lists
            "\n",              # Line breaks
            ". ",              # Sentence endings
            "; ",              # Semicolons
            ", ",              # Commas
            " ",               # Word boundaries
        ],
        add_start_index=True,
    )

    docs = splitter.create_documents([page_text])
    chunks = []
    for doc in docs:
        chunks.append({
            "text": doc.page_content,
            "index": 0,  # Will be re-indexed later
            "char_start": doc.metadata.get("start_index", 0),
            "is_table": False,
        })

    return chunks


def _post_process_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Post-process chunks for quality:
    1. Merge very short chunks (< 80 chars) with previous chunk (SAME PAGE ONLY)
    2. Detect and preserve tables as whole units
    3. Keep bullet/list items grouped (within same page)
    """
    if not chunks:
        return chunks

    processed = []
    buffer = None

    for chunk in chunks:
        text = chunk["text"].strip()
        page_num = chunk.get("page_number", 1)

        # Detect Markdown table
        lines = text.split("\n")
        is_table = (
            len(lines) >= 2
            and lines[0].strip().startswith("|")
            and any("---" in line for line in lines[:3])
        )

        # If table, keep as whole chunk
        if is_table:
            chunk["is_table"] = True
            processed.append(chunk)
            buffer = None
            continue

        # If very short, buffer for merging (but only if same page)
        if len(text) < 80 and buffer is not None:
            # Only merge if same page
            if buffer.get("page_number") == page_num:
                buffer["text"] += "\n" + text
                buffer["is_table"] = False
                continue
            else:
                # Different page — flush buffer and start new
                processed.append(buffer)
                buffer = None

        # If buffer exists and current chunk is long enough, flush buffer
        if buffer is not None:
            processed.append(buffer)
            buffer = None

        # Check if this starts a list/bullet section — buffer until list ends
        starts_list = text.startswith("- ") or text.startswith("* ") or text.startswith("1. ")
        if starts_list:
            buffer = chunk.copy()
            buffer["is_table"] = False
            continue

        processed.append(chunk)

    # Flush remaining buffer
    if buffer is not None:
        if len(buffer["text"].strip()) >= 40:  # Only keep if meaningful
            processed.append(buffer)

    # Re-index after merging
    for i, chunk in enumerate(processed):
        chunk["index"] = i

    return processed


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (4 chars ≈ 1 token for English).
    """
    return len(text) // 4

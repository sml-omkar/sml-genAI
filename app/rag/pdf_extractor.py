"""
PDF Text Extractor
Extracts text from PDFs using:
1. PyMuPDF4LLM (primary) for text-based PDFs
2. pdfplumber (fallback) for complex tables
3. Tesseract OCR (fallback) for scanned/image-based PDFs

Output: Markdown-formatted text with page markers for precise citation.
"""

import asyncio
from typing import Optional
from pathlib import Path


async def extract_pdf_text(file_path: str) -> dict:
    """
    Extract text from a PDF file.
    Returns: {"text": str, "pages": int, "method": str}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _extract_sync, file_path)
    return result


def _extract_sync(file_path: str) -> dict:
    """
    Synchronous extraction — runs in thread pool.
    Tries text extraction first, falls back to OCR for scanned pages.
    """
    import pymupdf

    doc = pymupdf.open(file_path)
    page_count = doc.page_count

    # Check if PDF has extractable text
    total_text = 0
    page_texts = []
    for page in doc:
        text = page.get_text()
        page_texts.append(text)
        total_text += len(text.strip())
    doc.close()

    avg_text_per_page = total_text / max(page_count, 1)

    # If most pages have text, use text extraction
    if avg_text_per_page > 50:
        # Try PyMuPDF4LLM first (clean Markdown output)
        try:
            result = _extract_with_pymupdf(file_path)
            if result["text"] and len(result["text"].strip()) > 100:
                return result
        except Exception as e:
            print(f"[PDF] PyMuPDF4LLM failed: {e}")

        # Fallback to pdfplumber (better for tables)
        try:
            result = _extract_with_pdfplumber(file_path)
            if result["text"] and len(result["text"].strip()) > 50:
                return result
        except Exception as e:
            print(f"[PDF] pdfplumber failed: {e}")

    # Scanned PDF detected — use OCR
    print(f"[PDF] Scanned PDF detected (avg {avg_text_per_page:.0f} chars/page). Using OCR...")
    try:
        result = _extract_with_ocr(file_path)
        if result["text"] and len(result["text"].strip()) > 50:
            return result
    except Exception as e:
        print(f"[PDF] OCR failed: {e}")

    raise ValueError(f"Could not extract text from PDF: {file_path}")


def _extract_with_pymupdf(file_path: str) -> dict:
    """Extract using PyMuPDF4LLM — outputs clean Markdown with page markers."""
    import pymupdf4llm
    import pymupdf

    # Extract with page chunks for better structure
    md_pages = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=True,       # Return per-page chunks for structure
        write_images=False,
        show_progress=False,
    )

    doc = pymupdf.open(file_path)
    page_count = doc.page_count
    doc.close()

    # Build structured text with explicit page markers
    full_text = []
    for page_data in md_pages:
        page_num = page_data.get("metadata", {}).get("page", 0)
        page_text = page_data.get("text", "")
        if page_text.strip():
            full_text.append(f"--- Page {page_num} ---\n{page_text.strip()}")

    combined = "\n\n".join(full_text)

    return {
        "text": combined,
        "pages": page_count,
        "method": "pymupdf4llm",
    }


def _extract_with_pdfplumber(file_path: str) -> dict:
    """Extract using pdfplumber — excellent for table-heavy PDFs."""
    import pdfplumber

    full_text = []
    page_count = 0

    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""

            # Extract tables as Markdown
            tables = page.extract_tables()
            table_text = ""
            if tables:
                for table in tables:
                    if table and len(table) > 0:
                        headers = [str(cell or "") for cell in table[0]]
                        table_text += "| " + " | ".join(headers) + " |\n"
                        table_text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        for row in table[1:]:
                            cells = [str(cell or "") for cell in row]
                            table_text += "| " + " | ".join(cells) + " |\n"
                        table_text += "\n"

            page_content = text
            if table_text:
                page_content += "\n\n" + table_text

            if page_content.strip():
                full_text.append(f"--- Page {page_num} ---\n{page_content}")

    combined_text = "\n\n".join(full_text)

    return {
        "text": combined_text,
        "pages": page_count,
        "method": "pdfplumber",
    }


def _extract_with_ocr(file_path: str) -> dict:
    """
    OCR extraction for scanned/image-based PDFs.
    Renders each page as an image, then runs Tesseract OCR.
    """
    import pymupdf
    import pytesseract
    from PIL import Image
    import io

    doc = pymupdf.open(file_path)
    page_count = doc.page_count
    full_text = []

    for page_num in range(page_count):
        page = doc[page_num]

        # Render at 300 DPI for better OCR accuracy
        mat = pymupdf.Matrix(300/72, 300/72)
        pix = page.get_pixmap(matrix=mat)

        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Run OCR with detailed config
        text = pytesseract.image_to_string(
            img,
            lang='eng',
            config='--psm 6'  # Assume uniform block of text
        )

        if text.strip():
            full_text.append(f"--- Page {page_num + 1} ---\n{text.strip()}")

    doc.close()
    combined_text = "\n\n".join(full_text)

    if not combined_text.strip():
        raise ValueError("OCR produced no text")

    return {
        "text": combined_text,
        "pages": page_count,
        "method": "ocr",
    }


def get_pdf_page_count(file_path: str) -> int:
    """Get PDF page count without extracting text."""
    import pymupdf
    doc = pymupdf.open(file_path)
    count = doc.page_count
    doc.close()
    return count

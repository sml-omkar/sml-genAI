"""Generate an architecture/cost documentation PDF for the NXSS AI system."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)
import os

OUT = os.path.join(os.path.dirname(__file__), "AIBot_Architecture_and_Cost.pdf")

doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=25*mm, rightMargin=25*mm,
    topMargin=20*mm, bottomMargin=20*mm,
    title="NXSS AI: Architecture & API Cost Breakdown",
    author="SML Engineering",
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleX", parent=styles["Title"], fontSize=20, leading=24,
    alignment=TA_CENTER, spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "SubtitleX", parent=styles["Normal"], fontSize=11, textColor=colors.grey,
    alignment=TA_CENTER, spaceAfter=18,
)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=14, spaceAfter=6)
h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=10, spaceAfter=4)
body = styles["BodyText"]

bullet = ListFlowable

def bullets(items, style=body):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=6) for i in items],
        bulletType="bullet", start="•", bulletColor=colors.HexColor("#1f6fb2"),
        leftIndent=14,
    )

story = []

story.append(Paragraph("NXSS AI — Architecture & API Cost Breakdown", title_style))
story.append(Paragraph("What we use from OpenAI, what runs in-house, and why we built our own RAG", subtitle_style))

# ---------------------------------------------------------------- Executive
story.append(Paragraph("1. Overview", h2))
story.append(Paragraph(
    "NXSS AI is an internal company chatbot that answers employee questions from uploaded "
    "policy PDF documents. It combines our own in-house Retrieval-Augmented Generation (RAG) "
    "pipeline with OpenAI's large language model for the final answer generation. The vector "
    "search, document processing, and knowledge retrieval are built and hosted entirely by us — "
    "we pay OpenAI only for the text generation (the 'brain' that writes the answers).", body))

# ---------------------------------------------------------------- What we use from OpenAI
story.append(Paragraph("2. What We Pay For (OpenAI)", h2))
story.append(Paragraph(
    "We call the OpenAI Chat Completions API using the <b>gpt-5.4-mini</b> model. This is the only "
    "paid component. It acts as the language 'brain' and is used for these tasks:", body))
story.append(Spacer(1, 4))
story.append(bullets([
    "<b>Intent routing</b> — understands whether the user is greeting, asking about a document, or making small talk.",
    "<b>Query rewriting</b> — turns the user's question into optimized search terms.",
    "<b>Relevance evaluation</b> — judges whether the retrieved document chunks actually answer the question.",
    "<b>Answer generation</b> — writes the final natural-language answer using the retrieved facts.",
    "<b>Policy memory extraction</b> — when a document is uploaded, summarizes it and extracts key rules.",
]))

story.append(Paragraph("2.1 Typical Cost per Chat Message", h3))
story.append(Paragraph(
    "One user question can trigger several small model calls. Here is a typical breakdown:", body))
story.append(Spacer(1, 8))

# cost table
cost_rows = [
    ["Step", "Model Call", "~Tokens In/Out", "Est. Cost"],
    ["1. Intent routing", "gpt-5.4-mini", "300 / 20", "$0.00005"],
    ["2. Query rewriting", "gpt-5.4-mini", "300 / 200", "$0.0001"],
    ["3. Relevance evaluation", "gpt-5.4-mini", "1,500 / 150", "$0.0003"],
    ["4. Answer generation", "gpt-5.4-mini", "3,000 / 1,000", "$0.0008"],
    ["5. Occasional retry", "gpt-5.4-mini", "1,500 / 256", "$0.0003"],
    ["", "TOTAL (typical)", "", "<b>≈ $0.0015 / message</b>"],
]
cost_table = Table(cost_rows, colWidths=[100, 60, 75, 55])
cost_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6fb2")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, -1), (0, -1), "Helvetica-Bold"),
    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef4fb")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(cost_table)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>So a single Q&amp;A costs roughly one-tenth of a US cent.</b> At, say, 1,000 questions a month, "
    "the total OpenAI bill is about $1–3 USD per month per user equivalent. Prices scale only with "
    "usage — there is no fixed license or server cost for the model.", body))

# ---------------------------------------------------------------- What we run in-house
story.append(Paragraph("3. What We Build &amp; Host Ourselves (In-House)", h2))
story.append(Paragraph(
    "Everything except the final text generation is built, owned, and operated internally. This "
    "keeps our documents private and our ongoing cost near zero for retrieval.", body))

story.append(Paragraph("3.1 The RAG Pipeline (Built by us)", h3))
story.append(bullets([
    "<b>PDF Processing</b> — PyMuPDF4LLM, pdfplumber, and Tesseract OCR extract text, tables, and scanned pages from uploaded policy PDFs.",
    "<b>Chunking</b> — text is split into page-aware, searchable chunks (custom chunker using LangChain's splitter).",
    "<b>Embeddings (local, free)</b> — Ollama's nomic-embed-text converts each chunk into a 768-dimension vector. <b>Runs on our own server, no API cost.</b>",
    "<b>Vector Store</b> — ChromaDB stores the vectors and does fast semantic (cosine) search, plus our own keyword boost and cross-encoder re-ranking for higher accuracy.",
    "<b>Policy Memory</b> — when a document is uploaded, we distill it into a summary + key facts stored in PostgreSQL, injected at query time.",
    "<b>Conversation Memory</b> — multi-turn context stored in PostgreSQL with a 24-hour expiry.",
]))
story.append(Paragraph("3.2 Infrastructure", h3))
story.append(bullets([
    "<b>Application</b> — Python / FastAPI web service and admin portal.",
    "<b>Database</b> — PostgreSQL for users, folders, documents, conversations, and policy memory.",
    "<b>Vector DB</b> — ChromaDB (persistent) for embeddings.",
    "<b>LLM runtime</b> — Ollama runs locally for embeddings only.",
]))

# ---------------------------------------------------------------- Why not use OpenAI RAG
story.append(Paragraph("4. Why We Don't Use OpenAI's Assisted/RAG Features", h2))
story.append(Paragraph(
    "OpenAI offers its own 'Assistants' or file-search / retrieval features. We deliberately built "
    "our own RAG instead. The reasons:", body))
story.append(Spacer(1, 4))
story.append(bullets([
    "<b>Privacy &amp; Control</b> — Company policy PDFs stay on our infrastructure. We never send raw documents to OpenAI; we only send the specific relevant text chunks needed to answer a single question.",
    "<b>Lower Cost</b> — Cloud vector search charges per query and per token on every retrieval. Our local ChromaDB search is free and instant, so we pay OpenAI only for the short final answer.",
    "<b>Freshness &amp; Ownership</b> — We manage our own knowledge base, chunking, citation, and admin upload flow. Documents can be updated instantly through our portal, independent of any third-party tool.",
    "<b>Transparency &amp; Auditing</b> — Every answer is backed by cited source chunks and a documented reasoning trace, giving HR/IT full visibility into what the bot used.",
    "<b>Accuracy Control</b> — We tune our own retrieval (keyword boost, re-ranking, relevance gates) to reduce hallucination and keep answers grounded strictly in our documents.",
    "<b>No Vendor Lock-in</b> — We can swap the model (OpenAI, or a local one) without rebuilding our retrieval system.",
]))

# ---------------------------------------------------------------- Summary
story.append(Paragraph("5. Summary", h2))
summary_rows = [
    ["Component", "Runs Where", "Cost"],
    ["LLM (gpt-5.4-mini)", "OpenAI API", "Per message (~$0.0015)"],
    ["Embeddings (nomic-embed-text)", "Our server (Ollama)", "Free"],
    ["Vector Search (ChromaDB)", "Our server", "Free"],
    ["Database (PostgreSQL)", "Our server", "Free"],
    ["Application &amp; Admin portal", "Our server", "Free"],
]
summary_table = Table(summary_rows, colWidths=[150, 100, 60])
summary_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6fb2")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(summary_table)
story.append(Spacer(1, 12))
story.append(Paragraph(
    "<b>Bottom line:</b> We keep our private data and the entire retrieval engine in-house (free), and "
    "pay OpenAI a few tenths of a cent for the model to write each answer. This gives us privacy, "
    "control, and dramatically lower running costs than using a fully cloud-hosted RAG service.", body))

# =========================================================================
# Section 6 - Azure AI Foundry comparison
# =========================================================================
story.append(Paragraph("6. What Would It Cost to Move Fully to Azure AI Foundry?", h2))
story.append(Paragraph(
    "A common question is whether we could consolidate everything onto Microsoft's Azure AI Foundry. "
    "The short answer: the LLM token cost would be roughly the same, but we would gain a standing "
    "monthly infrastructure bill — because Azure charges separately for the vector search, storage, "
    "networking, and monitoring that we currently run for free in-house.", body))

story.append(Paragraph("6.1 Budget Scenario (50,000 questions per month)", h3))
story.append(Paragraph(
    "The comparison below models a realistic enterprise workload — an internal knowledge assistant "
    "handling about 50,000 Q&amp;A conversations per month, each using roughly 5,000 tokens of context "
    "plus a written answer.", body))
story.append(Spacer(1, 8))

azure_rows = [
    ["Component", "Our Hybrid (in-house)", "Full Azure AI Foundry"],
    ["LLM inference (gpt-4o-mini / gpt-5.4-mini)",
     "~$50 – $75 / mo", "~$75 – $100 / mo"],
    ["Vector search (ChromaDB vs Azure AI Search S1)",
     "Free (local server)", "$250 / mo"],
    ["Document storage",
     "Free (local disk)", "~$5 / mo"],
    ["Private networking / endpoints",
     "Free (local)", "~$40 – $65 / mo"],
    ["Monitoring / support overhead",
     "~$0 (minimal)", "~$30 – $50 / mo"],
    ["Hosting / compute overhead",
     "Free (existing server)", "~$50 – $100 / mo"],
    ["", "<b>≈ $50 – $75 / mo</b>", "<b>≈ $450 – $570 / mo</b>"],
]
azure_table = Table(azure_rows, colWidths=[120, 80, 110])
azure_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6fb2")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, -1), (0, -1), "Helvetica-Bold"),
    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef4fb")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(azure_table)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>At this volume, moving fully to Azure AI Foundry costs roughly $400–$500 more per month "
    "(≈ $5,000–6,000 more per year)</b> than our current hybrid approach. The model itself is not the "
    "main driver — Azure AI Search (Standard S1) alone is about $250/month, and networking, private "
    "endpoints, and monitoring add another ~$150/month.", body))

story.append(Paragraph("6.2 Yearly Comparison", h3))
year_rows = [
    ["Approach", "Monthly", "Yearly"],
    ["Our hybrid (recommended)", "~$50 – $75", "~$600 – $900"],
    ["Full Azure AI Foundry", "~$450 – $570", "~$5,400 – $6,800"],
    ["Difference", "", "<b>~$4,800 – $5,900 saved / yr</b>"],
]
year_table = Table(year_rows, colWidths=[150, 80, 80])
year_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6fb2")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef4fb")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(year_table)
story.append(Spacer(1, 8))

story.append(Paragraph("6.3 Why the Jump?", h3))
story.append(bullets([
    "<b>Azure AI Search is a paid service</b> — it replaces our free in-house ChromaDB vector store. Standard S1 starts around $250/month regardless of usage.",
    "<b>Infrastructure overhead</b> — private endpoints (~$40–65/mo), monitoring, storage, compute, and support get billed separately instead of running on our own server.",
    "<b>Token prices are similar</b> — Azure's GPT-4o-mini is $0.15/$0.60 per million tokens, comparable to what we pay. The model is not the problem; the platform is.",
]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "<b>Recommendation:</b> Stay with the hybrid model. The paid LLM is the only thing worth buying "
    "per-token; hosting our own retrieval engine saves roughly $400–$500 every month while keeping our "
    "documents private and under our control.", body))

doc.build(story)
print(f"PDF written to {OUT}")

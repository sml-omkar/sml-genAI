"""
Adaptive Card Builder
Creates Adaptive Card responses for Teams bot interactions.
"""

from botbuilder.schema import Attachment


def build_welcome_card() -> dict:
    """Welcome card shown when user first interacts with the bot."""
    return {
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "Welcome to Policy Assistant",
                "size": "Large",
                "weight": "Bolder",
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": "I can help you find answers from company policy documents. Just ask me a question!",
                "wrap": True,
                "spacing": "Medium",
            },
            {
                "type": "TextBlock",
                "text": "Examples:",
                "weight": "Bolder",
                "spacing": "Medium",
            },
            {
                "type": "TextBlock",
                "text": "- What is the annual leave policy?\n- How do I request remote work?\n- What are the expense reimbursement rules?",
                "wrap": True,
                "spacing": "Small",
            },
        ],
        "actions": [],
    }


def build_answer_card(answer: str, sources: list) -> dict:
    """Build a card showing the RAG answer with source citations."""
    body = [
        {
            "type": "TextBlock",
            "text": "Answer",
            "size": "Medium",
            "weight": "Bolder",
        },
        {
            "type": "TextBlock",
            "text": answer,
            "wrap": True,
            "spacing": "Medium",
        },
    ]

    # Add sources section if available
    if sources:
        source_text = "**Sources:**\n"
        for s in sources:
            doc_name = s.get("document_name", "Unknown")
            dept = s.get("department", "N/A")
            source_text += f"- {doc_name} ({dept})\n"

        body.append({
            "type": "TextBlock",
            "text": source_text,
            "wrap": True,
            "spacing": "Medium",
            "size": "Small",
            "isSubtle": True,
        })

    return {
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
        "actions": [],
    }


def build_error_card(error_message: str) -> dict:
    """Build an error card."""
    return {
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "Something went wrong",
                "size": "Medium",
                "weight": "Bolder",
                "color": "Attention",
            },
            {
                "type": "TextBlock",
                "text": error_message,
                "wrap": True,
                "spacing": "Medium",
            },
        ],
        "actions": [],
    }


def build_no_results_card() -> dict:
    """Build a card when no relevant documents are found."""
    return {
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "No results found",
                "size": "Medium",
                "weight": "Bolder",
            },
            {
                "type": "TextBlock",
                "text": "I could not find relevant information in the policy documents for your question. Please try rephrasing your question or contact your department administrator.",
                "wrap": True,
                "spacing": "Medium",
            },
        ],
        "actions": [],
    }


def build_processing_card() -> dict:
    """Build a card shown while processing the query."""
    return {
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "Searching policy documents...",
                "wrap": True,
            },
        ],
        "actions": [],
    }


def create_attachment(card: dict) -> Attachment:
    """Convert an Adaptive Card dict to a Bot Framework Attachment."""
    return Attachment(
        content_type="application/vnd.microsoft.card.adaptive",
        content=card,
    )

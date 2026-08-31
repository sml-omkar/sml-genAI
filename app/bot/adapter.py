"""
Bot Framework Adapter
Sets up the CloudAdapter for Teams Bot integration with FastAPI.
"""

import os
from types import SimpleNamespace
from typing import Optional

from botbuilder.integration.aiohttp import (
    CloudAdapter,
    ConfigurationBotFrameworkAuthentication,
)
from botbuilder.core import TurnContext

from app.config import get_settings

settings = get_settings()


def create_adapter() -> Optional[CloudAdapter]:
    """
    Create and configure the Bot Framework CloudAdapter.
    Uses Microsoft App credentials from environment variables.

    Returns None when the Microsoft App credentials are not configured yet,
    so the web app can still run without a live Teams bot.
    """
    if not (settings.MicrosoftAppId and settings.MicrosoftAppPassword):
        print("[BOT] MicrosoftAppId/MicrosoftAppPassword not set — Teams bot disabled.")
        return None

    config = SimpleNamespace(
        APP_TYPE=settings.MicrosoftAppType,
        APP_ID=settings.MicrosoftAppId,
        APP_PASSWORD=settings.MicrosoftAppPassword,
        APP_TENANTID=settings.MicrosoftAppTenantId,
    )

    adapter = CloudAdapter(ConfigurationBotFrameworkAuthentication(config))

    # Global error handler — catches unhandled exceptions in bot logic
    async def on_error(context: TurnContext, error: Exception):
        print(f"[BOT] Turn error: {error}")
        await context.send_activity(
            "Sorry, I encountered an error processing your request. Please try again."
        )

    adapter.on_turn_error = on_error
    return adapter


# Singleton adapter instance (None until bot credentials are configured)
adapter = create_adapter()

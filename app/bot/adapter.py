"""
Bot Framework Adapter
Sets up the CloudAdapter for Teams Bot integration with FastAPI.
"""

import os
from types import SimpleNamespace

from botbuilder.integration.aiohttp import (
    CloudAdapter,
    ConfigurationBotFrameworkAuthentication,
)
from botbuilder.core import TurnContext

from app.config import get_settings

settings = get_settings()


def create_adapter() -> CloudAdapter:
    """
    Create and configure the Bot Framework CloudAdapter.
    Uses Microsoft App credentials from environment variables.
    """
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


# Singleton adapter instance
adapter = create_adapter()

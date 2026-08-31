"""Build the installable EthosAI Teams app package (.zip).

Fills placeholders in teams_app/manifest.template.json using the settings in
.env (MicrosoftAppId, host domain) and/or CLI overrides, then zips
manifest.json + color.png + outline.png into teams_app/ethos-ai.zip.
"""

import argparse
import io
import json
import os
import sys
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TEAMS_DIR = os.path.join(REPO, "teams_app")
TEMPLATE = os.path.join(TEAMS_DIR, "manifest.template.json")
OUT_ZIP = os.path.join(TEAMS_DIR, "ethos-ai.zip")

sys.path.insert(0, REPO)


def load_settings():
    from app.config import get_settings
    return get_settings()


def build(domain=None, bot_id=None, app_id=None):
    settings = load_settings()

    with open(TEMPLATE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Bot ID: CLI override > .env MicrosoftAppId
    resolved_bot_id = bot_id or settings.MicrosoftAppId
    if not resolved_bot_id or resolved_bot_id.startswith("CHANGE_ME"):
        raise SystemExit(
            "ERROR: MicrosoftAppId is not set in .env (or was not passed with --bot-id). "
            "Fill it in before building the package."
        )

    resolved_domain = (domain or "").strip().rstrip("/")
    if not resolved_domain:
        # Fall back to bot's expected host; developer URLs still need a real one.
        raise SystemExit(
            "ERROR: a domain is required. Pass --domain https://your-domain "
            "(leave off the /api/messages path)."
        )
    # Strip scheme for validDomains / developer URLs decision
    plane = resolved_domain.replace("https://", "").replace("http://", "").split("/")[0]

    resolved_app_id = app_id or manifest.get("id")
    if not resolved_app_id or resolved_app_id.startswith("CHANGE_ME"):
        resolved_app_id = str(uuid.uuid4())

    manifest["bots"][0]["botId"] = resolved_bot_id
    manifest["id"] = resolved_app_id
    manifest["validDomains"] = [plane]
    manifest["developer"]["websiteUrl"] = resolved_domain
    manifest["developer"]["privacyUrl"] = resolved_domain
    manifest["developer"]["termsOfUseUrl"] = resolved_domain

    # Build zip
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for icon in ("color.png", "outline.png"):
            p = os.path.join(TEAMS_DIR, icon)
            if not os.path.exists(p):
                raise SystemExit(f"ERROR: missing {p} – run scripts/gen_teams_icons.py first.")
            z.write(p, icon)

    print(f"Built {OUT_ZIP}")
    print(f"  botId  = {resolved_bot_id}")
    print(f"  domain = {resolved_domain}")
    print("Upload this .zip in Teams: Apps -> Manage your apps -> Upload a custom app.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build EthosAI Teams app package")
    p.add_argument("--domain", help="Public HTTPS base URL, e.g. https://ethos.example.com")
    p.add_argument("--bot-id", help="Bot Application (client) ID (overrides .env)")
    p.add_argument("--app-id", help="Teams app package id/GUID (default: random)")
    args = p.parse_args()
    build(domain=args.domain, bot_id=args.bot_id, app_id=args.app_id)

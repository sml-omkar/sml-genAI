"""Generate a step-by-step PDF guide for integrating EthosAI as a Microsoft Teams bot."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
    Preformatted,
)
from reportlab.platypus import HRFlowable

OUT = os.path.join(os.path.dirname(__file__), "EthosAI_Teams_Integration_Guide.pdf")

ACCENT = colors.HexColor("#1f6fb2")
CODEBG = colors.HexColor("#f4f4f4")

doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=22*mm, rightMargin=22*mm,
    topMargin=20*mm, bottomMargin=20*mm,
    title="EthosAI - Microsoft Teams Integration Guide",
    author="SML Engineering",
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleX", parent=styles["Title"], fontSize=20, leading=24,
    alignment=TA_CENTER, spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "SubtitleX", parent=styles["Normal"], fontSize=11, textColor=colors.grey,
    alignment=TA_CENTER, spaceAfter=8,
)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=16, spaceAfter=6)
h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=10, spaceAfter=4, textColor=ACCENT)
body = styles["BodyText"]
step_style = ParagraphStyle(
    "Step", parent=styles["Heading3"], fontSize=12, spaceBefore=14,
    spaceAfter=4, textColor=ACCENT,
)
code_style = ParagraphStyle(
    "Code", parent=styles["Code"], fontSize=8.5, leading=11,
    backColor=CODEBG, borderPadding=6, borderColor=colors.HexColor("#dddddd"),
    borderWidth=0.5,
)
note_style = ParagraphStyle(
    "Note", parent=styles["Normal"], fontSize=9.5, leading=12,
    textColor=colors.HexColor("#5b2d8f"), spaceBefore=8, spaceAfter=8,
)
warn_style = ParagraphStyle(
    "Warn", parent=styles["Normal"], fontSize=9.5, leading=12,
    textColor=colors.HexColor("#b12704"), spaceBefore=8, spaceAfter=8,
)


def bullets(items, style=body):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=6) for i in items],
        bulletType="bullet", start="•", bulletColor=ACCENT,
        leftIndent=14,
    )


def numbered(items, style=body):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=6) for i in items],
        bulletType="1", start="1", bulletColor=ACCENT,
        leftIndent=18,
    )


def code_block(text):
    return Preformatted(text.rstrip("\n"), code_style)


story = []

story.append(Paragraph("EthosAI — Microsoft Teams Integration Guide", title_style))
story.append(Paragraph("Step-by-step instructions to make the policy chatbot available to employees in Microsoft Teams", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=12))

# ---------------------------------------------------------------- Overview
story.append(Paragraph("Overview", h2))
story.append(Paragraph(
    "EthosAI is a FastAPI web service that exposes a bot endpoint at <b>/api/messages</b> "
    "(see <font face='Courier'>app/bot/bot_handler.py</font>). To let employees add the bot to their "
    "Teams chat, we need to register it in Azure, create an Azure Bot resource, deploy the service "
    "behind a public HTTPS endpoint, package a Teams app, and publish that app to the tenant. "
    "This guide walks through every step.", body))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Prerequisites:</b> an account with permission to create resources in the company's "
    "Microsoft 365 / Entra tenant (a normal employee account is not enough), and the Ubuntu server "
    "where the app is deployed.", warn_style))

# ---------------------------------------------------------------- Step 1
story.append(Paragraph("Step 1 — Register a Bot App in Azure AD (Entra ID)", step_style))
story.append(numbered([
    "Go to <b>https://portal.azure.com</b> and sign in with an administrative account.",
    "Navigate to <b>Microsoft Entra ID &rarr; App registrations &rarr; + New registration</b>.",
    "Set the <b>Name</b> to <b>EthosAI Bot</b>.",
    "Under <b>Supported account types</b>, choose <b>Accounts in this organizational directory only</b> "
    "(single tenant), since this is for your company only.",
    "Click <b>Register</b>.",
]))
story.append(Paragraph("From the Overview page, copy these two values (you will need them later):", body))
story.append(Spacer(1, 6))
step1_rows = [
    ["Field", "Copy from", "Maps to .env"],
    ["Application (client) ID", "Overview page", "MicrosoftAppId"],
    ["Directory (tenant) ID", "Overview page", "MicrosoftAppTenantId"],
]
t1 = Table(step1_rows, colWidths=[150, 120, 120])
t1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t1)
story.append(Spacer(1, 6))
story.append(numbered([
    "In the left menu, open <b>Certificates &amp; secrets &rarr; + New client secret</b>.",
    "Give it a description (e.g. <font face='Courier'>ethos-bot-secret</font>) and an expiry, then click <b>Add</b>.",
    "<b>Immediately copy the Value</b> — it is shown only once. This is the bot password.",
    "Add this value to <b>.env</b> as <b>MicrosoftAppPassword</b>.",
]))

# ---------------------------------------------------------------- Step 2
story.append(Paragraph("Step 2 — Create the Azure Bot Resource", step_style))
story.append(numbered([
    "In the Azure portal, click <b>Create a resource</b> and search for <b>Azure Bot</b>.",
    "Choose <b>Azure Bot</b> and click <b>Create</b>.",
    "Set a <b>Bot handle</b> (e.g. <font face='Courier'>ethos-ai-bot</font>).",
    "For <b>Microsoft App ID</b>, select <b>Use existing app registration</b> and pick the app from Step 1.",
    "Complete the wizard and create the resource.",
    "Open the new bot resource, go to <b>Configuration</b>, and set the <b>Messaging endpoint</b> to:",
]))
story.append(code_block(
    "https://<your-domain>/api/messages"
))
story.append(Paragraph(
    "If your domain is not live yet, leave it blank for now and set it after completing Step 5.",
    body))
story.append(numbered([
    "Open <b>Channels</b>, click the <b>Microsoft Teams</b> row, and choose <b>Configure</b> (or enable) "
    "the Teams channel to connect the bot to Teams.",
]))

# ---------------------------------------------------------------- Step 3
story.append(Paragraph("Step 3 — Set the Bot Credentials in .env on Ubuntu", step_style))
story.append(Paragraph(
    "On the Ubuntu server, edit <font face='Courier'>/sml-genAI/.env</font> and confirm these values:", body))
story.append(Spacer(1, 4))
story.append(code_block(
    "MicrosoftAppType=SingleTenant\n"
    "MicrosoftAppId=<client-id-from-step-1>\n"
    "MicrosoftAppPassword=<secret-value-from-step-1>\n"
    "MicrosoftAppTenantId=<tenant-id-from-step-1>"
))
story.append(Paragraph(
    "Restart the application (e.g. <font face='Courier'>docker compose restart</font> or "
    "<font face='Courier'>bash scripts/start.sh</font>) so the new credentials are picked up.", body))

# ---------------------------------------------------------------- Step 4
story.append(Paragraph("Step 4 — Deploy and Expose the Endpoint over HTTPS", step_style))
story.append(Paragraph(
    "Microsoft Teams only accepts bot endpoints served over <b>valid HTTPS</b>. The application runs "
    "on port 8000; we use Nginx with a Let's Encrypt certificate in front of it. First, point a "
    "domain (or subdomain) at your Ubuntu server's public IP, then install Nginx and Certbot:", body))
story.append(Spacer(1, 4))
story.append(code_block(
    "sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx"
))
story.append(Spacer(1, 6))
story.append(Paragraph("Create the Nginx site configuration:", body))
story.append(Spacer(1, 4))
story.append(code_block(
    "sudo nano /etc/nginx/sites-available/cyprus\n"
    "\n"
    "server {\n"
    "    server_name <your-domain>;\n"
    "    listen 443 ssl;\n"
    "    ssl_certificate     /etc/letsencrypt/live/<your-domain>/fullchain.pem;\n"
    "    ssl_certificate_key /etc/letsencrypt/live/<your-domain>/privkey.pem;\n"
    "\n"
    "    location /api/messages {\n"
    "        proxy_pass http://127.0.0.1:8000;\n"
    "        proxy_set_header Host $host;\n"
    "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
    "        proxy_set_header X-Forwarded-Proto https;\n"
    "    }\n"
    "}"
))
story.append(Spacer(1, 6))
story.append(code_block(
    "sudo ln -s /etc/nginx/sites-available/cyprus /etc/nginx/sites-enabled/\n"
    "sudo nginx -t\n"
    "sudo systemctl reload nginx\n"
    "\n"
    "# Issue the SSL certificate\n"
    "sudo certbot --nginx -d <your-domain>"
))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Verify the endpoint responds (it should return an empty 200 OK to a bare POST, which confirms "
    "Nginx is forwarding correctly):", body))
story.append(code_block(
    "curl https://<your-domain>/api/messages\n"
    "# or, after changing the firewall:\n"
    "sudo ufw allow 'Nginx Full'"
))
story.append(Paragraph(
    "After HTTPS is live, return to <b>Step 2</b> and enter the final messaging endpoint in the "
    "Azure Bot resource.", note_style))

# ---------------------------------------------------------------- Step 5
story.append(Paragraph("Step 5 — Create the Teams App Package (manifest)", step_style))
story.append(Paragraph(
    "To let employees \u201cAdd\u201d the bot, create a small Teams app package — a folder containing a "
    "<font face='Courier'>manifest.json</font> and two icon images. Create a directory and the manifest:", body))
story.append(Spacer(1, 4))
story.append(code_block(
    "mkdir ethos-ai-app && cd ethos-ai-app\n"
    "nano manifest.json"
))
story.append(Spacer(1, 6))
story.append(Paragraph("Paste this into <font face='Courier'>manifest.json</font> (replace the placeholders):", body))
story.append(Spacer(1, 4))
story.append(code_block(
    "{\n"
    "  \"$schema\": \"https://developer.microsoft.com/en-us/json-schemas/teams/v1.17/MicrosoftTeams.schema.json\",\n"
    "  \"manifestVersion\": \"1.17\",\n"
    "  \"version\": \"1.0.0\",\n"
    "  \"id\": \"<some-unique-guid>\",\n"
    "  \"packageName\": \"com.sml.ethosai\",\n"
    "  \"developer\": {\n"
    "    \"name\": \"SML\",\n"
    "    \"websiteUrl\": \"https://<your-domain>\",\n"
    "    \"privacyUrl\": \"https://<your-domain>\",\n"
    "    \"termsOfUseUrl\": \"https://<your-domain>\"\n"
    "  },\n"
    "  \"name\": { \"short\": \"EthosAI\", \"full\": \"EthosAI Policy Assistant\" },\n"
    "  \"description\": {\n"
    "    \"short\": \"Company policy chatbot\",\n"
    "    \"full\": \"Answers questions from company policy and procedure documents.\"\n"
    "  },\n"
    "  \"icons\": { \"color\": \"color.png\", \"outline\": \"outline.png\" },\n"
    "  \"accentColor\": \"#FFFFFF\",\n"
    "  \"bots\": [\n"
    "    {\n"
    "      \"botId\": \"<MicrosoftAppId>\",\n"
    "      \"scopes\": [\"personal\", \"team\"],\n"
    "      \"supportsFiles\": false,\n"
    "      \"isNotificationOnly\": false\n"
    "    }\n"
    "  ],\n"
    "  \"validDomains\": [\"<your-domain>\"],\n"
    "  \"permissions\": [\"identity\", \"messageTeamMembers\"]\n"
    "}"
))
story.append(Spacer(1, 6))
story.append(Paragraph("Add the required icon images to the same folder:", body))
story.append(bullets([
    "<b>color.png</b> — 192x192 pixels, transparent background, full-colour logo.",
    "<b>outline.png</b> — 32x32 pixels, transparent background, white/transparent outline logo.",
]))
story.append(Spacer(1, 6))
story.append(Paragraph("Zip the folder — the resulting <font face='Courier'>.zip</font> is the installable Teams app package:", body))
story.append(code_block(
    "cd ..\n"
    "zip -r ethos-ai.zip ethos-ai-app"
))

# ---------------------------------------------------------------- Step 6
story.append(Paragraph("Step 6 — Publish and Let Employees Install", step_style))
story.append(Paragraph(
    "There are two ways to distribute the app, depending on whether you want to test first or roll it "
    "out company-wide:", body))
story.append(Spacer(1, 6))
story.append(Paragraph("6.1 Sideload (testing / small group)", h3))
story.append(numbered([
    "In Teams, open the <b>Apps</b> store.",
    "Click <b>Manage your apps &rarr; Upload an app &rarr; Upload a custom app</b>.",
    "Select <font face='Courier'>ethos-ai.zip</font> and click <b>Add</b>.",
]))
story.append(Paragraph("6.2 Company-wide via Teams Admin Center (production)", h3))
story.append(numbered([
    "Go to <b>https://admin.teams.microsoft.com</b> with a Teams administrator account.",
    "Open <b>Teams apps &rarr; Manage apps &rarr; Upload new app</b> and upload <font face='Courier'>ethos-ai.zip</font>.",
    "Submit/publish the app; define an app setup policy so it appears for the relevant employees.",
    "Employees then go to <b>Apps</b>, search <b>EthosAI</b>, click <b>Add</b>, and use it in a 1:1 chat.",
]))

# ---------------------------------------------------------------- Important
story.append(Paragraph("Important — User Access &amp; Email Matching", h2))
story.append(Paragraph(
    "The bot identifies an employee in Teams by their email or Azure AD object ID and looks them up "
    "in the PostgreSQL <font face='Courier'>users</font> table (see <font face='Courier'>app/bot/bot_handler.py</font>). "
    "For an employee to actually see answers, all of the following must be true:", body))
story.append(Spacer(1, 4))
story.append(bullets([
    "Their email in Teams must match the email in the <b>users</b> table exactly.",
    "They must exist in the <b>users</b> table (seeded by the admin portal or <font face='Courier'>scripts/init_db.py</font>).",
    "They must be assigned to a <b>group</b>, and that group must be linked to the <b>folder(s)</b> they should access.",
    "Folders must contain processed (READY) documents for them to receive answers.",
]))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "If a user sees \u201cyou don't have access to any document folders\u201d, it means one of the "
    "group/folder assignments is missing in the admin portal.", note_style))

# ---------------------------------------------------------------- Checklist
story.append(Paragraph("Quick Checklist", h2))
checklist_rows = [
    ["#", "Task", "Done"],
    ["1", "App registration created in Entra; client ID, tenant ID, secret saved", ""],
    ["2", "Azure Bot resource created and connected to the app registration", ""],
    ["3", ".env on Ubuntu updated with SingleTenant credentials", ""],
    ["4", "Nginx + Let's Encrypt serving /api/messages over HTTPS", ""],
    ["5", "Messaging endpoint set in Azure Bot resource", ""],
    ["6", "Teams channel enabled on the bot", ""],
    ["7", "Teams app package (manifest + icons) created and zipped", ""],
    ["8", "App uploaded/published to the tenant", ""],
    ["9", "Employee users + groups + folder access seeded in admin portal", ""],
    ["10", "Live test in a Teams 1:1 chat", ""],
]
cl = Table(checklist_rows, colWidths=[30, 330, 40])
cl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
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
story.append(cl)
story.append(Spacer(1, 12))
story.append(Paragraph(
    "Once the app is published and users are mapped correctly, employees simply open Teams, find "
    "EthosAI in the apps store, and add it to their chat to start asking policy questions.", body))

doc.build(story)
print(f"PDF written to {OUT}")

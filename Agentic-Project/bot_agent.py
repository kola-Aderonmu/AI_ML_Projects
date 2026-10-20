import os
import base64
import datetime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from typing import Dict, Literal
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google import genai
from pydantic import BaseModel, Field
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

# --- CONFIGURATIONS ---
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.labels",
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ALLOWED_USER_ID", "0"))
LOCAL_TIMEZONE = ZoneInfo("Africa/Lagos")  # Set to West Africa Time (UTC+1)

ai_client = genai.Client()

LABEL_CONFIG = {
    "AI-Triage/VIP": {"backgroundColor": "#cc3a21", "textColor": "#ffffff"},
    "AI-Triage/Action-Needed": {"backgroundColor": "#ffad46", "textColor": "#000000"},
    "AI-Triage/FYI-News": {"backgroundColor": "#4986e7", "textColor": "#ffffff"},
    "AI-Triage/Spam-Promo": {"backgroundColor": "#cccccc", "textColor": "#000000"},
}

class TriageResult(BaseModel):
    category: Literal["VIP", "ACTION_REQUIRED", "FYI_NEWSLETTER", "SPAM"]
    urgency: int = Field(ge=1, le=5)
    requires_draft: bool
    summary: str

class DraftResult(BaseModel):
    subject: str
    body: str

# --- GMAIL HELPERS ---
def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def ensure_labels(service) -> Dict[str, str]:
    res = service.users().labels().list(userId="me").execute()
    existing = {lbl["name"]: lbl["id"] for lbl in res.get("labels", [])}
    label_map = {}
    for name, colors in LABEL_CONFIG.items():
        if name in existing:
            label_map[name] = existing[name]
        else:
            body = {"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show", "color": colors}
            created = service.users().labels().create(userId="me", body=body).execute()
            label_map[name] = created["id"]
    return label_map

def triage_email(sender: str, subject: str, snippet: str) -> TriageResult:
    prompt = f"Sender: {sender}\nSubject: {subject}\nSnippet: {snippet}\nClassify this email according to schema."
    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={"response_mime_type": "application/json", "response_schema": TriageResult},
    )
    return TriageResult.model_validate_json(response.text)

def draft_response(sender: str, subject: str, snippet: str, summary: str) -> DraftResult:
    prompt = f"Original Sender: {sender}\nSubject: {subject}\nSummary: {summary}\nContent: {snippet}\nDraft a concise, in-voice response (2-3 sentences max). Zero corporate fluff."
    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={"response_mime_type": "application/json", "response_schema": DraftResult},
    )
    return DraftResult.model_validate_json(response.text)

def create_draft(service, to: str, subject: str, body_text: str, thread_id: str):
    message = MIMEText(body_text)
    message["to"] = to
    message["subject"] = subject
    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {"message": {"threadId": thread_id, "raw": encoded}}
    return service.users().drafts().create(userId="me", body=body).execute()

# --- CORE PROCESSING FUNCTION ---
def process_inbox(is_morning_digest: bool = False) -> str:
    service = get_gmail_service()
    label_map = ensure_labels(service)
    results = service.users().messages().list(userId="me", q="is:unread label:INBOX", maxResults=10).execute()
    messages = results.get("messages", [])

    if not messages:
        if is_morning_digest:
            return "🌅 *Good morning!* Your inbox is clean (Inbox Zero). No unread emails."
        return "✨ Inbox Zero! No unread messages found."

    header = "🌅 *Morning Executive Inbox Digest*" if is_morning_digest else "📬 *Inbox Triage Report*"
    report = [f"{header}\nProcessed {len(messages)} unread messages:\n"]

    for item in messages:
        msg_id = item["id"]
        msg = service.users().messages().get(userId="me", id=msg_id, format="metadata").execute()
        headers = msg.get("payload", {}).get("headers", [])
        
        sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
        subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(No Subject)")
        thread_id = msg.get("threadId")
        snippet = msg.get("snippet", "")

        triage = triage_email(sender, subject, snippet)
        
        target_label = {
            "VIP": "AI-Triage/VIP",
            "ACTION_REQUIRED": "AI-Triage/Action-Needed",
            "FYI_NEWSLETTER": "AI-Triage/FYI-News",
            "SPAM": "AI-Triage/Spam-Promo",
        }.get(triage.category, "AI-Triage/FYI-News")

        has_draft = False
        if triage.requires_draft:
            draft = draft_response(sender, subject, snippet, triage.summary)
            create_draft(service, sender, draft.subject, draft.body, thread_id)
            has_draft = True

        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={"removeLabelIds": ["UNREAD"], "addLabelIds": [label_map[target_label]]}
        ).execute()

        status_emoji = "🔴 VIP" if triage.category == "VIP" else ("🟡 Action" if triage.category == "ACTION_REQUIRED" else "🔵 FYI")
        report.append(
            f"• *{subject}*\n"
            f"  From: `{sender}`\n"
            f"  Category: {status_emoji} | Draft Created: *{has_draft}*\n"
            f"  Summary: _{triage.summary}_"
        )

    return "\n\n".join(report)

# --- TELEGRAM HANDLERS & SCHEDULER ---
async def auth_guard(update: Update) -> bool:
    if update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("⛔ Unauthorized access.")
        return False
    return True

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await auth_guard(update): return
    await update.message.reply_text(
        "👋 **Amelia Inbox Agent is Active**\n\n"
        "⏰ Daily Morning Digest scheduled for **8:30 AM WAT**.\n\n"
        "Commands:\n"
        "• `/scan` — Triage incoming emails right now\n"
        "• `/testdigest` — Test the morning digest output now\n"
        "• Chat with me directly to ask questions about your mail.",
        parse_mode="Markdown"
    )

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await auth_guard(update): return
    await update.message.reply_text("🔍 Scanning inbox and drafting responses...")
    report = process_inbox(is_morning_digest=False)
    await update.message.reply_text(report, parse_mode="Markdown")

async def test_digest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await auth_guard(update): return
    await update.message.reply_text("⏳ Generating sample morning digest...")
    report = process_inbox(is_morning_digest=True)
    await update.message.reply_text(report, parse_mode="Markdown")

async def scheduled_morning_digest(context: ContextTypes.DEFAULT_TYPE):
    """Background job executed by JobQueue at 08:30 AM every day."""
    print("⏰ JobQueue Triggered: Running 8:30 AM Morning Digest...")
    report = process_inbox(is_morning_digest=True)
    await context.bot.send_message(
        chat_id=ALLOWED_USER_ID,
        text=report,
        parse_mode="Markdown"
    )

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await auth_guard(update): return
    user_query = update.message.text
    prompt = f"""
    You are Amelia, the user's executive inbox assistant.
    User instruction: "{user_query}"
    Provide a concise, direct, helpful response.
    """
    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    await update.message.reply_text(response.text)

def main():
    print("🤖 Kay-Kay Telegram Bot is starting...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register Handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("testdigest", test_digest_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    # Configure 8:30 AM Daily Morning Job
    target_time = datetime.time(hour=8, minute=30, tzinfo=LOCAL_TIMEZONE)
    app.job_queue.run_daily(
        callback=scheduled_morning_digest,
        time=target_time,
        name="daily_morning_digest"
    )
    print(f"✅ Morning Digest scheduled daily at {target_time.strftime('%I:%M %p %Z')}")

    app.run_polling()

if __name__ == "__main__":
    main()
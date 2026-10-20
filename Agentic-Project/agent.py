import base64
import os
from email.mime.text import MIMEText
from typing import Dict, Literal
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

# Scopes required for reading, drafting, and managing label colors
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.labels",
]

# Initialize Gemini Client
ai_client = genai.Client()

# System Contexts (Soul & User Files)
SOUL_PROMPT = """
You are an executive inbox assistant.
Voice & Tone: Direct, decisive, warm, and concise (2-4 sentences max).
Style Rules: Zero corporate fluff (never say "Hope this finds you well"). State action items immediately.
Negative Constraints: Never commit to unverified dates or financial transactions.
"""

USER_PROMPT = """
User: Busy Professional / Operator.
VIP Categories: Leadership, Direct Clients, Key Collaborators, Urgent System Alerts.
Sign-off: Best regards,
"""

# Label Schema & Color Definitions (Hex values approved by Gmail API)
LABEL_CONFIG = {
    "AI-Triage/VIP": {
        "backgroundColor": "#cc3a21",  # Red
        "textColor": "#ffffff"
    },
    "AI-Triage/Action-Needed": {
        "backgroundColor": "#ffad46",  # Orange / Yellow
        "textColor": "#000000"
    },
    "AI-Triage/FYI-News": {
        "backgroundColor": "#4986e7",  # Blue
        "textColor": "#ffffff"
    },
    "AI-Triage/Spam-Promo": {
        "backgroundColor": "#cccccc",  # Gray
        "textColor": "#000000"
    },
}

# Structured Output Schemas
class TriageResult(BaseModel):
    category: Literal["VIP", "ACTION_REQUIRED", "FYI_NEWSLETTER", "SPAM"]
    urgency: int = Field(ge=1, le=5)
    requires_draft: bool
    summary: str

class DraftResult(BaseModel):
    subject: str
    body: str


def get_gmail_service():
    """Authenticates using credentials.json and saves token.json."""
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


def ensure_labels_exist(service) -> Dict[str, str]:
    """Ensures our custom color-coded labels exist in Gmail and returns a map of {name: id}."""
    existing_labels_res = service.users().labels().list(userId="me").execute()
    existing_labels = {lbl["name"]: lbl["id"] for lbl in existing_labels_res.get("labels", [])}
    
    label_map = {}
    for name, colors in LABEL_CONFIG.items():
        if name in existing_labels:
            label_map[name] = existing_labels[name]
        else:
            # Create label with colors
            label_body = {
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
                "color": colors
            }
            created = service.users().labels().create(userId="me", body=label_body).execute()
            label_map[name] = created["id"]
            print(f"🎨 Created Gmail Label: {name}")
    return label_map


def triage_email(sender: str, subject: str, snippet: str) -> TriageResult:
    """Triage Sub-Agent using Gemini."""
    prompt = f"""
    Context: {USER_PROMPT}
    Sender: {sender}
    Subject: {subject}
    Snippet: {snippet}

    Analyze and classify this email according to the schema.
    """
    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": TriageResult,
        },
    )
    return TriageResult.model_validate_json(response.text)


def draft_response(sender: str, subject: str, snippet: str, summary: str) -> DraftResult:
    """Drafter Sub-Agent matching your voice."""
    prompt = f"""
    {SOUL_PROMPT}
    {USER_PROMPT}
    Original Sender: {sender}
    Original Subject: {subject}
    Summary: {summary}
    Content: {snippet}

    Draft a concise, in-voice reply.
    """
    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": DraftResult,
        },
    )
    return DraftResult.model_validate_json(response.text)


def create_draft(service, to: str, subject: str, body_text: str, thread_id: str):
    """Creates an associated draft in Gmail."""
    message = MIMEText(body_text)
    message["to"] = to
    message["subject"] = subject
    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {"message": {"threadId": thread_id, "raw": encoded}}
    return service.users().drafts().create(userId="me", body=body).execute()


def run_agent():
    service = get_gmail_service()
    
    # 1. Initialize and cache color-coded labels
    label_map = ensure_labels_exist(service)
    
    print("🔍 Kay-Kay Agent: Scanning unread inbox...")
    results = service.users().messages().list(userId="me", q="is:unread label:INBOX", maxResults=5).execute()
    messages = results.get("messages", [])

    if not messages:
        print("✨ Inbox Zero: No unread messages.")
        return

    for item in messages:
        msg_id = item["id"]
        msg = service.users().messages().get(userId="me", id=msg_id, format="metadata").execute()
        headers = msg.get("payload", {}).get("headers", [])
        
        sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
        subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(No Subject)")
        thread_id = msg.get("threadId")
        snippet = msg.get("snippet", "")

        print(f"\n📨 Processing: '{subject}' from {sender}")
        
        # 2. Triage with AI
        triage = triage_email(sender, subject, snippet)
        print(f"   ↳ [Triage] Category: {triage.category} | Urgency: {triage.urgency}/5")

        # 3. Map category to label ID
        target_label_name = {
            "VIP": "AI-Triage/VIP",
            "ACTION_REQUIRED": "AI-Triage/Action-Needed",
            "FYI_NEWSLETTER": "AI-Triage/FYI-News",
            "SPAM": "AI-Triage/Spam-Promo",
        }.get(triage.category, "AI-Triage/FYI-News")
        
        label_id_to_apply = label_map.get(target_label_name)

        # 4. Draft if required
        if triage.requires_draft:
            print("   ↳ [Drafter] Drafting response in your voice...")
            draft = draft_response(sender, subject, snippet, triage.summary)
            create_draft(service, sender, draft.subject, draft.body, thread_id)
            print("   ✅ Draft created in Gmail!")

        # 5. Apply color-coded label and remove UNREAD
        modify_body = {
            "removeLabelIds": ["UNREAD"],
            "addLabelIds": [label_id_to_apply] if label_id_to_apply else []
        }
        service.users().messages().modify(userId="me", id=msg_id, body=modify_body).execute()
        print(f"   🏷️ Applied visual label: {target_label_name}")


if __name__ == "__main__":
    run_agent()
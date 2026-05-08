#!/usr/bin/env python3
"""Send phase report via Gmail SMTP.

Usage:
    python scripts/send_report.py --phase 0
    python scripts/send_report.py --phase 0 --dry-run

Reads .env for Gmail config (v3 variable names):
    GMAIL_ADDRESS         # sender, e.g. ch.jungsik@gmail.com
    GMAIL_APP_PASSWORD    # 16-char Gmail app password (NOT regular password)
    REPORT_EMAIL_TO       # recipient (often same as GMAIL_ADDRESS)

Get Gmail app password: https://myaccount.google.com/apppasswords
(2-step verification must be enabled.)
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


ROOT = Path(__file__).parent.parent

# Gmail SMTP fixed config
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def load_dotenv():
    """Minimal dotenv loader."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, help="Phase number (e.g. '0', '6a')")
    parser.add_argument("--subject-suffix", default="", help="Append to email subject")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    report_path = ROOT / f"reports/phase-{args.phase}-report.md"

    if args.dry_run:
        print(f"[DRY RUN] Would send phase {args.phase} report")
        print(f"  Report path: {report_path}")
        print(f"  Exists:      {report_path.exists()}")
        print(f"  GMAIL_ADDRESS:    {os.environ.get('GMAIL_ADDRESS', '<not set>')}")
        print(f"  REPORT_EMAIL_TO:  {os.environ.get('REPORT_EMAIL_TO', '<not set>')}")
        print(f"  GMAIL_APP_PASSWORD: {'<set>' if os.environ.get('GMAIL_APP_PASSWORD') else '<NOT SET>'}")
        return 0

    if not report_path.exists():
        print(f"ERROR: report not found: {report_path}", file=sys.stderr)
        return 1

    sender = os.environ.get("GMAIL_ADDRESS")
    app_pass = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("REPORT_EMAIL_TO")

    missing = []
    if not sender:
        missing.append("GMAIL_ADDRESS")
    if not app_pass:
        missing.append("GMAIL_APP_PASSWORD")
    if not recipient:
        missing.append("REPORT_EMAIL_TO")

    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        print("  Set them in .env (see .env.example)", file=sys.stderr)
        return 1

    body = report_path.read_text(encoding="utf-8")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    subject = f"[CacheBlend v4] Phase {args.phase} report"
    if args.subject_suffix:
        subject += f" - {args.subject_suffix}"
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", _charset="utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(sender, app_pass)
            server.send_message(msg)
        print(f"OK Sent phase {args.phase} report")
        print(f"  From: {sender}")
        print(f"  To:   {recipient}")
        print(f"  Subject: {subject}")
        return 0
    except smtplib.SMTPAuthenticationError as e:
        print(f"ERROR: Gmail SMTP auth failed: {e}", file=sys.stderr)
        print("  Check GMAIL_APP_PASSWORD (16-char app password, not regular password).", file=sys.stderr)
        print("  Get one: https://myaccount.google.com/apppasswords", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: SMTP send failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Configuration for the agentic outreach worker.

Every value resolves through :func:`erosolar_agent.secrets.get_secret`, so a real
environment variable always beats the gitignored ``.env``. Nothing here is a
secret literal — the Proton Bridge password lives only in ``.env``.

Safety: live sending is double-gated. The worker only delivers real email when
BOTH ``OUTREACH_ALLOW_SEND=true`` (operator env, where Bridge runs) AND the
Firestore ``outreach/control`` doc has ``dry_run=false``. Default is drafts-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import secrets


def _flag(name: str, default: bool) -> bool:
    raw = secrets.get_secret(name, str(default))
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(str(secrets.get_secret(name, str(default))).strip())
    except (TypeError, ValueError):
        return default


@dataclass
class OutreachConfig:
    # Mail provider: "proton" (local Bridge) | "gmail" (live imap/smtp.gmail.com)
    provider: str = "proton"
    # IMAP/SMTP creds + endpoints (populated per-provider in load()). Field names
    # keep the "bridge_" prefix for back-compat; for Gmail they hold the Gmail
    # address + App Password.
    bridge_user: str = ""
    bridge_password: str = ""
    imap_host: str = "127.0.0.1"
    imap_port: int = 1143
    smtp_host: str = "127.0.0.1"
    smtp_port: int = 1025
    security: str = "starttls"  # "starttls" | "ssl"

    # Identity
    from_email: str = ""
    from_name: str = "Erosolar"
    admin_email: str = ""

    # LLM / search
    llm_model: str = "deepseek-v4-pro"       # drafting + follow-up judgement
    summary_model: str = "deepseek-v4-flash"  # cheap dead-end summaries
    owner_email: str = ""                      # human escalation inbox (bo@shang.software)

    # Storage
    project_id: str = "erosolar-llm"
    service_account: str = "configs/firebase-service-account.json"

    # Safety / pacing
    allow_send: bool = False
    max_per_run: int = 25
    min_send_interval_sec: int = 45
    poll_sec: int = 30

    @classmethod
    def load(cls) -> "OutreachConfig":
        provider = str(secrets.get_secret("MAIL_PROVIDER", "proton")).lower()
        if provider == "gmail":
            user = secrets.get_secret("GMAIL_USER", "")
            # App Passwords are shown in 4-char groups; strip spaces.
            password = (secrets.get_secret("GMAIL_APP_PASSWORD", "") or "").replace(" ", "")
            imap_host, imap_port = "imap.gmail.com", 993
            smtp_host, smtp_port = "smtp.gmail.com", 465
            security = "ssl"  # 993 IMAP-SSL + 465 SMTP-SSL, both verified TLS
            # Gmail can only send AS the authenticated account (or a verified
            # alias) — never honor a stale OUTREACH_FROM_EMAIL from another provider.
            from_email = user
        else:
            user = secrets.get_secret("PROTON_BRIDGE_USER", "")
            password = secrets.get_secret("PROTON_BRIDGE_PASSWORD", "")
            imap_host = secrets.get_secret("PROTON_IMAP_HOST", "127.0.0.1")
            imap_port = _int("PROTON_IMAP_PORT", 1143)
            smtp_host = secrets.get_secret("PROTON_SMTP_HOST", "127.0.0.1")
            smtp_port = _int("PROTON_SMTP_PORT", 1025)
            security = str(secrets.get_secret("PROTON_SECURITY", "starttls")).lower()
            from_email = secrets.get_secret("OUTREACH_FROM_EMAIL", "") or user
        return cls(
            provider=provider,
            bridge_user=user,
            bridge_password=password,
            imap_host=imap_host,
            imap_port=imap_port,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            security=security,
            from_email=from_email,
            from_name=secrets.get_secret("OUTREACH_FROM_NAME", "Erosolar"),
            admin_email=secrets.get_secret("OUTREACH_ADMIN_EMAIL", ""),
            llm_model=secrets.get_secret("OUTREACH_LLM_MODEL", "deepseek-v4-pro"),
            summary_model=secrets.get_secret("OUTREACH_SUMMARY_MODEL", "deepseek-v4-flash"),
            owner_email=secrets.get_secret("OUTREACH_OWNER_EMAIL", "bo@shang.software"),
            project_id=secrets.get_secret("FIREBASE_PROJECT_ID", "erosolar-llm"),
            service_account=secrets.get_secret(
                "FIREBASE_SERVICE_ACCOUNT", "configs/firebase-service-account.json"
            ),
            allow_send=_flag("OUTREACH_ALLOW_SEND", False),
            max_per_run=_int("OUTREACH_MAX_PER_RUN", 25),
            min_send_interval_sec=_int("OUTREACH_MIN_SEND_INTERVAL_SEC", 45),
            poll_sec=_int("OUTREACH_POLL_SEC", 30),
        )

    def bridge_ready(self) -> bool:
        return bool(self.bridge_user and self.bridge_password)

    def summary(self) -> str:
        """One-line, secret-free status for logs."""
        return (
            f"outreach: provider={self.provider} from={self.from_email or '?'} "
            f"model={self.llm_model} imap={self.imap_host}:{self.imap_port} "
            f"smtp={self.smtp_host}:{self.smtp_port} security={self.security} "
            f"allow_send={self.allow_send} max_per_run={self.max_per_run}"
        )

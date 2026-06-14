"""Proton Mail Bridge client — IMAP read + SMTP send over 127.0.0.1.

Proton Bridge runs locally and presents a self-signed certificate, so TLS
verification is intentionally relaxed *for the loopback connection only*. We
never disable verification for a non-loopback host.

Nothing here decides whether to send — :class:`OutreachEngine` owns the safety
gate. ``send()`` performs the actual delivery when called.
"""

from __future__ import annotations

import email
import imaplib
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid, parseaddr

from .config import OutreachConfig

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _loopback_tls(host: str) -> ssl.SSLContext:
    """TLS context for the local Bridge: skip verification ONLY on loopback."""
    ctx = ssl.create_default_context()
    if host in _LOOPBACK:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _decode(value) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


@dataclass
class IncomingMessage:
    message_id: str
    in_reply_to: str
    references: str
    from_email: str
    from_name: str
    subject: str
    body: str
    date: str
    uid: str = ""

    def thread_key(self) -> str:
        """Best-effort conversation key for grouping replies."""
        return (self.references.split() or [self.in_reply_to or self.message_id])[0].strip("<>")


@dataclass
class ProtonBridgeMail:
    cfg: OutreachConfig
    _last_error: str = field(default="", init=False)

    # ── connectivity ──────────────────────────────────────────────────────
    def test_connection(self) -> dict:
        """Log in to IMAP and SMTP without sending anything. Returns a report."""
        report = {"imap": False, "smtp": False, "folders": [], "error": ""}
        try:
            with self._imap() as imap:
                typ, data = imap.list()
                report["imap"] = typ == "OK"
                report["folders"] = [
                    _decode(line.decode("utf-8", "replace")) for line in (data or [])
                ][:25]
        except Exception as e:  # noqa: BLE001 - surface any failure to the caller
            report["error"] = f"IMAP: {e}"
        try:
            smtp = self._smtp()
            smtp.noop()
            smtp.quit()
            report["smtp"] = True
        except Exception as e:  # noqa: BLE001
            report["error"] = (report["error"] + f" SMTP: {e}").strip()
        return report

    # ── IMAP ──────────────────────────────────────────────────────────────
    def _imap(self) -> imaplib.IMAP4:
        host, port = self.cfg.imap_host, self.cfg.imap_port
        if self.cfg.security == "ssl":
            imap: imaplib.IMAP4 = imaplib.IMAP4_SSL(host, port, ssl_context=_loopback_tls(host))
        else:
            imap = imaplib.IMAP4(host, port)
            imap.starttls(ssl_context=_loopback_tls(host))
        imap.login(self.cfg.bridge_user, self.cfg.bridge_password)
        return imap

    def fetch_recent(
        self,
        mailbox: str = "INBOX",
        limit: int = 50,
        unseen_only: bool = True,
        mark_seen: bool = True,
    ) -> list[IncomingMessage]:
        """Fetch messages from a mailbox. With ``unseen_only`` it returns only
        unseen mail and (by default) marks it \\Seen; otherwise it returns the
        most recent ``limit`` messages read-only (no flag changes) — used to
        track ALL mail without disturbing the unread state."""
        out: list[IncomingMessage] = []
        with self._imap() as imap:
            imap.select(mailbox, readonly=not (unseen_only and mark_seen))
            typ, data = imap.search(None, "UNSEEN" if unseen_only else "ALL")
            if typ != "OK":
                return out
            ids = data[0].split() if data and data[0] else []
            ids = ids[-limit:] if not unseen_only else ids[:limit]  # most recent N for ALL
            for num in ids:
                typ, msg_data = imap.fetch(num, "(RFC822)")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue
                out.append(self._parse(msg_data[0][1], uid=num.decode()))
                if unseen_only and mark_seen:
                    imap.store(num, "+FLAGS", "\\Seen")
        return out

    def fetch_unseen(self, mailbox: str = "INBOX", limit: int = 50) -> list[IncomingMessage]:
        """Unseen messages, marked \\Seen so we don't re-ingest (outreach loop)."""
        return self.fetch_recent(mailbox, limit, unseen_only=True, mark_seen=True)

    @staticmethod
    def _parse(raw: bytes, uid: str = "") -> IncomingMessage:
        msg = email.message_from_bytes(raw)
        name, addr = parseaddr(msg.get("From", ""))
        return IncomingMessage(
            message_id=(msg.get("Message-ID", "") or "").strip(),
            in_reply_to=(msg.get("In-Reply-To", "") or "").strip(),
            references=(msg.get("References", "") or "").strip(),
            from_email=addr.lower(),
            from_name=_decode(name),
            subject=_decode(msg.get("Subject", "")),
            body=ProtonBridgeMail._extract_body(msg),
            date=msg.get("Date", ""),
            uid=uid,
        )

    @staticmethod
    def _extract_body(msg: email.message.Message) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and "attachment" not in str(
                    part.get("Content-Disposition", "")
                ):
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(part.get_content_charset() or "utf-8", "replace")
            return ""
        payload = msg.get_payload(decode=True)
        return payload.decode(msg.get_content_charset() or "utf-8", "replace") if payload else ""

    # ── SMTP ──────────────────────────────────────────────────────────────
    def _smtp(self) -> smtplib.SMTP:
        host, port = self.cfg.smtp_host, self.cfg.smtp_port
        if self.cfg.security == "ssl":
            smtp: smtplib.SMTP = smtplib.SMTP_SSL(host, port, context=_loopback_tls(host), timeout=30)
        else:
            smtp = smtplib.SMTP(host, port, timeout=30)
            smtp.ehlo()
            smtp.starttls(context=_loopback_tls(host))
            smtp.ehlo()
        smtp.login(self.cfg.bridge_user, self.cfg.bridge_password)
        return smtp

    def build_message(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        *,
        unsubscribe_mailto: str = "",
        in_reply_to: str = "",
        references: str = "",
    ) -> EmailMessage:
        """Construct a compliant plain-text message (with List-Unsubscribe)."""
        msg = EmailMessage()
        msg["From"] = formataddr((self.cfg.from_name, self.cfg.from_email))
        msg["To"] = to_email
        msg["Subject"] = subject
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain=self.cfg.from_email.split("@")[-1] or "pm.me")
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = (references + " " + in_reply_to).strip()
        unsub = unsubscribe_mailto or self.cfg.from_email
        if unsub:
            msg["List-Unsubscribe"] = f"<mailto:{unsub}?subject=unsubscribe>"
            msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
        msg.set_content(body_text)
        return msg

    def send(self, msg: EmailMessage) -> str:
        """Deliver a message via Bridge SMTP. Returns its Message-ID."""
        smtp = self._smtp()
        try:
            smtp.send_message(msg)
        finally:
            smtp.quit()
        return msg["Message-ID"]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

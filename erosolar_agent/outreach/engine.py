"""Outreach engine — plan → prospect → draft → (queue|send) → ingest replies.

Toggled live from the web admin nav via the Firestore ``outreach/control`` doc;
publishes a heartbeat to ``outreach/status`` so the nav indicator reflects state.

Safety (sending real email is outward-facing and hard to reverse):
  • dry_run defaults TRUE — the engine only writes drafts unless told otherwise.
  • A real send requires BOTH cfg.allow_send (operator env where Bridge runs)
    AND control.dry_run == False. Either gate off ⇒ draft only.
  • Per-run cap + minimum interval between sends + never contact the same lead
    twice without a reply + honor opt-out / "unsubscribe" replies.
"""

from __future__ import annotations

import time

from ..integrations.quota import QuotaExhausted
from . import draft as drafter
from . import prospect
from . import triage
from .config import OutreachConfig
from .mail import ProtonBridgeMail, utcnow_iso
from .prospect import Lead
from .rag import OutreachRAG
from .store import Store


class OutreachEngine:
    def __init__(self, cfg: OutreachConfig | None = None):
        self.cfg = cfg or OutreachConfig.load()
        self.store = Store(self.cfg)
        self.rag = OutreachRAG()
        self.mail = ProtonBridgeMail(self.cfg)
        self._last_send_ts = 0.0

    # ── status ──────────────────────────────────────────────────────────────
    def _status(self, state: str, **extra) -> None:
        self.store.write_status(
            {
                "state": state,
                "last_run_at": utcnow_iso(),
                "dry_run_effective": self._effective_dry_run(self.store.get_control()),
                "allow_send_env": self.cfg.allow_send,
                "counts": self.store.counts(),
                **extra,
            }
        )

    def _effective_dry_run(self, control: dict) -> bool:
        # Live send only when BOTH gates are open.
        return bool(control.get("dry_run", True)) or (not self.cfg.allow_send)

    # ── owner notifications (internal; ignore dry_run, still need allow_send) ─
    def notify_owner(self, subject: str, body: str, kind: str) -> bool:
        """Email the human owner (bo@shang.software). These are internal notices,
        not cold outreach, so they ignore dry_run — but the allow_send hard guard
        still applies. When sending is off, the notice is stored for the dashboard."""
        owner = self.cfg.owner_email
        base = {"direction": "owner_notice", "kind": kind, "to": owner,
                "subject": subject, "body": body[:8000], "sent": False}
        if not owner or not self.cfg.bridge_ready():
            self.store.add_message({**base, "status": "owner_notice_unsendable"})
            self.store.add_event("owner_notice_unsendable", {"kind": kind, "subject": subject})
            return False
        if not self.cfg.allow_send:
            self.store.add_message({**base, "status": "owner_notice_queued"})
            self.store.add_event("owner_notice_queued", {"kind": kind, "subject": subject})
            return False
        try:
            em = self.mail.build_message(owner, subject, body)
            mid = self.mail.send(em)
            self.store.add_message({**base, "status": "sent", "sent": True, "message_id": mid})
            self.store.add_event("owner_notice_sent", {"kind": kind, "subject": subject})
            return True
        except Exception as e:  # noqa: BLE001
            self.store.add_event("owner_notice_error", {"kind": kind, "error": str(e)[:300]})
            return False

    # ── mail ingestion + triage ──────────────────────────────────────────────
    def ingest_mail(self) -> int:
        """Read all new mail: detect bounces (fix/flag addresses), honor opt-outs,
        and triage genuine replies with deepseek-v4-pro."""
        if not self.cfg.bridge_ready():
            return 0
        try:
            incoming = self.mail.fetch_unseen(limit=self.cfg.max_per_run)
        except Exception as e:  # noqa: BLE001
            self.store.add_event("imap_error", {"error": str(e)[:300]})
            return 0
        for msg in incoming:
            bounce = triage.is_bounce(msg)
            rec_id = self.store.add_message(
                {
                    "direction": "inbound",
                    "from": msg.from_email,
                    "subject": msg.subject,
                    "body": msg.body[:8000],
                    "message_id": msg.message_id,
                    "in_reply_to": msg.in_reply_to,
                    "thread_key": msg.thread_key(),
                    "is_bounce": bounce,
                }
            )
            self.rag.add(
                rec_id,
                f"{'Bounce' if bounce else 'Reply'} from {msg.from_name} <{msg.from_email}>: "
                f"{msg.subject}\n{msg.body}",
                {"kind": "bounce" if bounce else "reply", "email": msg.from_email},
            )
            if bounce:
                self.store.add_event("bounce", {"from": msg.from_email})
                self._handle_bounce(msg)
                continue
            if "unsubscribe" in (msg.subject + " " + msg.body).lower():
                self.store.mark_opt_out(msg.from_email)
                self.store.add_event("opt_out", {"email": msg.from_email})
                continue
            self._handle_reply(msg)
        return len(incoming)

    def _handle_bounce(self, msg) -> None:
        """Fix a typo'd recipient if we safely can, else flag the address broken."""
        addr = triage.failed_recipient(msg)
        if not addr:
            self.store.add_event("bounce_unparsed", {"from": msg.from_email})
            return
        contact = self.store.find_contact_by_email(addr)
        fixed = triage.fix_address(addr)
        if fixed and fixed != addr:
            if contact:
                self.store.update_contact(
                    contact["id"], email=fixed, status="queued", address_fixed_from=addr
                )
            self.store.add_event("address_fixed", {"from": addr, "to": fixed})
        else:
            if contact:
                self.store.update_contact(contact["id"], status="bounced", email_broken=True)
            self.store.add_event("address_broken", {"email": addr})

    def _handle_reply(self, msg) -> None:
        """Judge with deepseek-v4-pro: follow up only if absolutely sensible; else
        escalate to the human owner or close as a dead end (with a flash summary)."""
        contact = self.store.find_contact_by_email(msg.from_email)
        ctx = self.rag.context_for(f"{msg.from_email} {msg.subject}", k=3)
        try:
            j = triage.judge_followup(msg, model=self.cfg.llm_model, history_context=ctx)
        except QuotaExhausted:
            self.store.add_event("deepseek_quota", {"stage": "judge"})
            return
        self.store.add_event(
            "judge",
            {"from": msg.from_email, "disposition": j.disposition,
             "sensible": j.sensible, "reason": j.reason},
        )

        if j.sensible and j.disposition == "follow_up":
            self._draft_or_send_followup(msg, contact, ctx)
            if contact:
                self.store.update_contact(contact["id"], status="followed_up")
        elif j.disposition == "human":
            body = (
                "A reply needs your attention.\n\n"
                f"From: {msg.from_name} <{msg.from_email}>\nSubject: {msg.subject}\n\n"
                f"Why follow-up isn't automatic: {j.reason}\n\n"
                f"Required human actions:\n{j.required_human_actions or '(use your judgement)'}\n\n"
                f"--- original reply ---\n{msg.body[:2000]}"
            )
            self.notify_owner(f"[outreach · action needed] {msg.subject}", body, "human_action")
            if contact:
                self.store.update_contact(contact["id"], status="needs_human")
        else:  # dead_end
            try:
                summary = triage.summarize_dead_end(msg, model=self.cfg.summary_model)
            except QuotaExhausted:
                self.store.add_event("deepseek_quota", {"stage": "summary"})
                summary = f"Dead-end reply: {msg.subject}"
            self.notify_owner(
                f"[outreach · dead end] {msg.subject}",
                f"Closed as a dead end.\nFrom: {msg.from_email}\n\nSummary: {summary}\n\nReason: {j.reason}",
                "dead_end",
            )
            if contact:
                self.store.update_contact(contact["id"], status="dead_end")

    def _draft_or_send_followup(self, msg, contact, ctx: str) -> None:
        lead = Lead(
            email=msg.from_email,
            org=(contact or {}).get("org", ""),
            role=(contact or {}).get("role", ""),
            snippet=msg.body[:400],
            audience=(contact or {}).get("audience", ""),
        )
        try:
            d = drafter.draft_email(
                lead, model=self.cfg.llm_model, from_name=self.cfg.from_name,
                history_context=ctx, inbound=msg.body,
            )
        except QuotaExhausted:
            self.store.add_event("deepseek_quota", {"stage": "followup"})
            return
        subj = msg.subject or d.subject
        subject = subj if subj.lower().startswith("re:") else f"Re: {subj}"
        rec = {"direction": "outbound", "to": msg.from_email, "subject": subject,
               "body": d.body, "kind": "follow_up", "in_reply_to": msg.message_id,
               "contact_id": (contact or {}).get("id", "")}
        if self._effective_dry_run(self.store.get_control()):
            self.store.add_message({**rec, "status": "drafted", "sent": False})
            return
        if not self._rate_ok():
            self.store.add_message({**rec, "status": "queued_rate", "sent": False})
            return
        try:
            em = self.mail.build_message(
                msg.from_email, subject, d.body,
                in_reply_to=msg.message_id, references=msg.references,
            )
            mid = self.mail.send(em)
            self._last_send_ts = time.monotonic()
            self.store.add_message({**rec, "status": "sent", "sent": True, "message_id": mid})
            self.store.add_event("follow_up_sent", {"to": msg.from_email})
        except Exception as e:  # noqa: BLE001
            self.store.add_event("smtp_error", {"to": msg.from_email, "error": str(e)[:300]})

    # ── prospecting ─────────────────────────────────────────────────────────
    def prospect_leads(self, control: dict) -> int:
        if control.get("mode") != "prospect":
            return 0
        brief = control.get("brief", "")
        if not brief:
            return 0
        audiences = control.get("audiences") or (
            [control["audience"]] if control.get("audience") else ["investors"]
        )
        leads: list = []
        try:
            for aud in audiences:
                leads.extend(prospect.discover(brief, audience=aud))
        except QuotaExhausted:
            self.store.add_event("tavily_quota", {})
            if not leads:
                return 0
        fresh = prospect.dedupe(leads, self.store.seen_keys())
        for lead in fresh:
            cid = self.store.upsert_contact(
                {
                    "name": lead.name,
                    "org": lead.org,
                    "role": lead.role,
                    "email": lead.email,
                    "url": lead.url,
                    "snippet": lead.snippet,
                    "audience": lead.audience,
                    "dedupe_key": lead.dedupe_key(),
                    # research-only until a human supplies a verified address
                    "status": "new" if lead.sendable() else "needs_email",
                }
            )
            self.rag.add(
                cid,
                f"Lead: {lead.org} {lead.role} ({lead.email or 'no-email'}) — {lead.snippet}",
                {"kind": "lead", "email": lead.email, "url": lead.url},
            )
        self.store.add_event("prospect", {"found": len(leads), "new": len(fresh)})
        return len(fresh)

    # ── drafting / sending ───────────────────────────────────────────────────
    def process_contacts(self, control: dict) -> dict:
        dry = self._effective_dry_run(control)
        sent = drafted = skipped = 0
        for contact in self.store.due_contacts(self.cfg.max_per_run):
            email = (contact.get("email") or "").strip()
            if not email or self.store.is_opted_out(email):
                skipped += 1
                continue
            lead = prospect.Lead(
                name=contact.get("name", ""),
                org=contact.get("org", ""),
                role=contact.get("role", ""),
                email=email,
                url=contact.get("url", ""),
                snippet=contact.get("snippet", ""),
                audience=contact.get("audience", control.get("audience", "")),
            )
            ctx = self.rag.context_for(f"{lead.org} {lead.role} {email}", k=3)
            try:
                d = drafter.draft_email(
                    lead,
                    model=self.cfg.llm_model,
                    from_name=self.cfg.from_name,
                    campaign_brief=control.get("brief", ""),
                    history_context=ctx,
                )
            except QuotaExhausted:
                self.store.add_event("deepseek_quota", {})
                break

            msg_record = {
                "direction": "outbound",
                "contact_id": contact["id"],
                "to": email,
                "subject": d.subject,
                "body": d.body,
                "status": "drafted",
                "sent": False,
            }

            if dry:
                mid = self.store.add_message(msg_record)
                self.store.update_contact(contact["id"], status="drafted", last_draft_id=mid)
                drafted += 1
            else:
                if not self._rate_ok():
                    skipped += 1
                    continue
                try:
                    em = self.mail.build_message(email, d.subject, d.body)
                    message_id = self.mail.send(em)
                    self._last_send_ts = time.monotonic()
                    msg_record.update(status="sent", sent=True, message_id=message_id)
                    mid = self.store.add_message(msg_record)
                    self.store.update_contact(
                        contact["id"], status="contacted", last_message_id=mid
                    )
                    self.store.add_event("sent", {"to": email, "subject": d.subject})
                    sent += 1
                except Exception as e:  # noqa: BLE001
                    self.store.add_event("smtp_error", {"to": email, "error": str(e)[:300]})
                    skipped += 1

            self.rag.add(
                contact["id"] + ":draft",
                f"Outbound to {email}: {d.subject}\n{d.body}",
                {"kind": "outbound", "email": email},
            )
        return {"sent": sent, "drafted": drafted, "skipped": skipped}

    def _rate_ok(self) -> bool:
        return (time.monotonic() - self._last_send_ts) >= self.cfg.min_send_interval_sec

    # ── one cycle / loop ─────────────────────────────────────────────────────
    def run_once(self) -> dict:
        control = self.store.get_control()
        if not control.get("enabled"):
            self._status("disabled")
            return {"enabled": False}
        self._status("running")
        mail = self.ingest_mail()
        found = self.prospect_leads(control)
        result = self.process_contacts(control)
        summary = {"enabled": True, "mail_processed": mail, "new_leads": found, **result}
        self._status("idle", last_cycle=summary)
        return summary

    def loop(self) -> None:
        print(self.cfg.summary(), f"| backend={self.store.backend}")
        while True:
            try:
                out = self.run_once()
                print(utcnow_iso(), out)
            except KeyboardInterrupt:
                self._status("stopped")
                print("stopped")
                return
            except Exception as e:  # noqa: BLE001
                self.store.add_event("engine_error", {"error": str(e)[:300]})
                print("engine error:", e)
            time.sleep(max(5, self.cfg.poll_sec))

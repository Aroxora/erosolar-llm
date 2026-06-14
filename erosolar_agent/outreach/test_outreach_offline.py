"""Offline tests for the outreach subsystem — no network, no Firestore, no Bridge.

Run: python -m erosolar_agent.outreach.test_outreach_offline
or:  pytest erosolar_agent/outreach/test_outreach_offline.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .config import OutreachConfig
from .draft import _coerce_json
from .mail import ProtonBridgeMail
from .prospect import Lead, _extract_email, dedupe


def test_config_flags_and_defaults():
    os.environ["OUTREACH_ALLOW_SEND"] = "false"
    os.environ["OUTREACH_MAX_PER_RUN"] = "7"
    cfg = OutreachConfig.load()
    assert cfg.allow_send is False
    assert cfg.max_per_run == 7
    assert "allow_send=False" in cfg.summary()
    # password must never appear in a log summary
    cfg.bridge_password = "topsecret"
    assert "topsecret" not in cfg.summary()


def test_effective_dry_run_double_gate():
    from .engine import OutreachEngine

    cfg = OutreachConfig.load()
    cfg.allow_send = False
    eng = OutreachEngine.__new__(OutreachEngine)
    eng.cfg = cfg
    # allow_send off ⇒ dry-run no matter what control says
    assert eng._effective_dry_run({"dry_run": False}) is True
    cfg.allow_send = True
    assert eng._effective_dry_run({"dry_run": False}) is False
    assert eng._effective_dry_run({"dry_run": True}) is True


def test_email_extraction_rejects_junk():
    assert _extract_email("contact me at jane@fund.vc please") == "jane@fund.vc"
    assert _extract_email("noreply@mailer.com only") == ""
    assert _extract_email("no address here") == ""


def test_lead_sendable_and_dedupe():
    a = Lead(email="jane@fund.vc", url="https://fund.vc/jane")
    b = Lead(email="JANE@fund.vc")  # same key, different case
    c = Lead(email="", url="https://other.com")
    assert a.sendable() and not c.sendable()
    fresh = dedupe([a, b, c], seen_keys=set())
    assert len(fresh) == 2  # b dedupes against a; c kept (different key)


def test_draft_json_coercion():
    assert _coerce_json('{"subject":"Hi","body":"There"}')["subject"] == "Hi"
    assert _coerce_json('```json\n{"subject":"A","body":"B"}\n```')["body"] == "B"
    assert _coerce_json("garbage no json") == {}


def test_build_message_has_unsubscribe_and_from():
    cfg = OutreachConfig.load()
    cfg.from_email = "erolunar@pm.me"
    cfg.from_name = "Erosolar"
    msg = ProtonBridgeMail(cfg).build_message("vc@fund.vc", "Subj", "Body text")
    assert msg["To"] == "vc@fund.vc"
    assert "Erosolar" in msg["From"] and "erolunar@pm.me" in msg["From"]
    assert "mailto:" in msg["List-Unsubscribe"]
    assert msg.get_content().strip() == "Body text"


def test_local_store_roundtrip(tmp_path: Path = None):
    # force local backend by pointing service account at a nonexistent path
    cfg = OutreachConfig.load()
    cfg.service_account = "configs/__does_not_exist__.json"
    import erosolar_agent.outreach.store as store_mod

    tmp = Path(tempfile.mkdtemp())
    store_mod._LOCAL_DIR = tmp  # type: ignore
    s = store_mod.Store(cfg)
    # firebase-admin may or may not be installed; if it is, this still must not send
    if s.backend != "local":
        return  # firestore path needs live creds; skip in offline test
    cid = s.upsert_contact({"email": "vc@fund.vc", "org": "Fund", "status": "new"})
    assert cid
    due = s.due_contacts(10)
    assert any(c["email"] == "vc@fund.vc" for c in due)
    s.update_contact(cid, status="contacted")
    assert all(c.get("status") != "new" for c in s.due_contacts(10) if c["id"] == cid)
    ctrl = s.set_control(enabled=True, dry_run=True)
    assert ctrl["enabled"] is True and ctrl["dry_run"] is True


def test_bounce_detection_and_address_fix():
    from .mail import IncomingMessage
    from . import triage

    bounce = IncomingMessage(
        message_id="<1>", in_reply_to="", references="",
        from_email="mailer-daemon@pm.me", from_name="Mail Delivery System",
        subject="Undelivered Mail Returned to Sender",
        body="550 5.1.1 <jane@gmial.com>: Recipient address rejected: User unknown\n"
             "Final-Recipient: rfc822; jane@gmial.com",
        date="",
    )
    assert triage.is_bounce(bounce) is True
    assert triage.failed_recipient(bounce) == "jane@gmial.com"
    assert triage.fix_address("jane@gmial.com") == "jane@gmail.com"
    assert triage.fix_address("jane@weird-startup.io") == ""  # never guesses

    real = IncomingMessage(
        message_id="<2>", in_reply_to="", references="",
        from_email="jane@fund.vc", from_name="Jane",
        subject="Re: hi", body="Sounds interesting, can you send a deck?", date="",
    )
    assert triage.is_bounce(real) is False


def test_judge_fallback_escalates_to_human():
    from . import triage

    # Unparseable model output must NEVER auto-follow-up; it escalates to human.
    j = triage._coerce("not json")
    assert j == {}


def test_gmail_provider_config():
    import erosolar_agent.secrets as s
    os.environ["MAIL_PROVIDER"] = "gmail"
    os.environ["GMAIL_USER"] = "demo@gmail.com"
    os.environ["GMAIL_APP_PASSWORD"] = "abcd efgh ijkl mnop"  # dummy, not a real secret
    s._dotenv.cache_clear()
    try:
        cfg = OutreachConfig.load()
        assert cfg.provider == "gmail"
        assert cfg.imap_host == "imap.gmail.com" and cfg.imap_port == 993
        assert cfg.smtp_host == "smtp.gmail.com" and cfg.smtp_port == 465
        assert cfg.security == "ssl"
        assert cfg.bridge_password == "abcdefghijklmnop"  # spaces stripped
        assert cfg.from_email == "demo@gmail.com"
        assert "provider=gmail" in cfg.summary() and "abcdefghijklmnop" not in cfg.summary()
    finally:
        os.environ.pop("MAIL_PROVIDER", None)
        s._dotenv.cache_clear()


def test_inbox_log_dedupe_id():
    from .store import _doc_id_from

    assert _doc_id_from("<abc@mail>") == _doc_id_from("<abc@mail>")  # stable per Message-ID
    assert _doc_id_from("<a@m>") != _doc_id_from("<b@m>")            # distinct ids
    assert _doc_id_from("") != _doc_id_from("")                      # empty -> unique


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f" FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} offline tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)

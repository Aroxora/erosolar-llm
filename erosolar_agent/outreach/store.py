"""Persistence for outreach state.

Primary backend is Firestore via ``firebase-admin`` (server credentials bypass
security rules — the web admin client is gated by rules separately). If
firebase-admin or the service account is unavailable, falls back to local JSON
files under ``data_store/outreach/`` so the worker still runs and you can develop
offline. Collections / docs:

    outreach/control   — singleton: {enabled, dry_run, brief, audience, mode}
    outreach/status    — singleton heartbeat written by the worker (read by the
                         web nav for the live toggle indicator)
    outreach_contacts  — discovered/known leads + per-contact send state
    outreach_messages  — every drafted/sent message and every inbound reply
    outreach_events    — append-only audit log
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from .config import OutreachConfig
from .mail import utcnow_iso

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_DIR = _REPO_ROOT / "data_store" / "outreach"

CONTROL_DEFAULT = {
    "enabled": False,
    "dry_run": True,  # safe default: draft, never send
    "brief": "",
    "audience": "investors",          # back-compat single value
    "audiences": ["investors"],        # multi-select campaign audiences
    "mode": "prospect",  # "prospect" (Tavily) | "list" (admin-curated only)
    "updated_by": "",
    "updated_at": "",
}


def _new_id() -> str:
    return uuid.uuid4().hex[:20]


class Store:
    def __init__(self, cfg: OutreachConfig):
        self.cfg = cfg
        self._db = None
        self.backend = "local"
        self._init_firestore()

    # ── backend selection ──────────────────────────────────────────────────
    def _init_firestore(self) -> None:
        sa = _REPO_ROOT / self.cfg.service_account
        try:
            import firebase_admin  # type: ignore
            from firebase_admin import credentials, firestore  # type: ignore

            if not firebase_admin._apps:
                if sa.exists():
                    cred = credentials.Certificate(str(sa))
                    firebase_admin.initialize_app(cred, {"projectId": self.cfg.project_id})
                else:
                    firebase_admin.initialize_app(options={"projectId": self.cfg.project_id})
            # Validate the credentials with a single token refresh BEFORE any
            # RPC. A revoked/stale service account otherwise triggers a ~300s
            # gapic retry (invalid_grant wrapped as 503) on the first real call;
            # a direct refresh raises immediately, so we fall back to local fast.
            if sa.exists():
                import google.auth.transport.requests as _gauth_req
                from google.oauth2 import service_account as _svc
                probe = _svc.Credentials.from_service_account_file(
                    str(sa), scopes=["https://www.googleapis.com/auth/datastore"]
                )
                probe.refresh(_gauth_req.Request())
            self._db = firestore.client()
            self.backend = "firestore"
        except Exception as e:  # noqa: BLE001 - any creds/network failure -> local
            self._db = None
            self.backend = "local"
            self._fs_error = str(e)[:200]
            _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            print(
                f"[outreach.store] Firestore unavailable ({self._fs_error}); using "
                f"local store at {_LOCAL_DIR}. Regenerate {self.cfg.service_account} "
                f"(Firebase console → Project settings → Service accounts) for live "
                f"portal reflection."
            )

    # ── control / status ─────────────────────────────────────────────────
    def get_control(self) -> dict:
        if self._db is not None:
            snap = self._db.collection("outreach").document("control").get()
            data = snap.to_dict() if snap.exists else {}
            return {**CONTROL_DEFAULT, **(data or {})}
        path = _LOCAL_DIR / "control.json"
        if path.exists():
            return {**CONTROL_DEFAULT, **json.loads(path.read_text())}
        return dict(CONTROL_DEFAULT)

    def set_control(self, **fields) -> dict:
        ctrl = self.get_control()
        ctrl.update(fields)
        ctrl["updated_at"] = utcnow_iso()
        if self._db is not None:
            self._db.collection("outreach").document("control").set(ctrl, merge=True)
        else:
            (_LOCAL_DIR / "control.json").write_text(json.dumps(ctrl, indent=2))
        # every control change is part of the dated outreach log
        self.add_event("control_change", {"fields": fields, "by": ctrl.get("updated_by", "")})
        return ctrl

    def write_status(self, status: dict) -> None:
        status = {**status, "heartbeat_at": utcnow_iso(), "backend": self.backend}
        if self._db is not None:
            self._db.collection("outreach").document("status").set(status, merge=True)
        else:
            (_LOCAL_DIR / "status.json").write_text(json.dumps(status, indent=2))

    # ── contacts ───────────────────────────────────────────────────────────
    def seen_keys(self) -> set[str]:
        keys: set[str] = set()
        for c in self._all("outreach_contacts", "contacts.jsonl"):
            for k in (c.get("email"), c.get("url"), c.get("dedupe_key")):
                if k:
                    keys.add(str(k).strip().lower())
        return keys

    def upsert_contact(self, contact: dict) -> str:
        cid = contact.get("id") or _new_id()
        contact = {**contact, "id": cid, "updated_at": utcnow_iso()}
        contact.setdefault("created_at", contact["updated_at"])
        contact.setdefault("status", "new")
        if self._db is not None:
            self._db.collection("outreach_contacts").document(cid).set(contact, merge=True)
        else:
            self._append_local("contacts.jsonl", contact, key="id")
        return cid

    def update_contact(self, cid: str, **fields) -> None:
        fields["updated_at"] = utcnow_iso()
        if self._db is not None:
            self._db.collection("outreach_contacts").document(cid).set(fields, merge=True)
        else:
            self._merge_local("contacts.jsonl", cid, fields)

    def due_contacts(self, limit: int) -> list[dict]:
        out = []
        for c in self._all("outreach_contacts", "contacts.jsonl"):
            if c.get("status") in ("new", "queued") and c.get("email") and not c.get("opted_out"):
                out.append(c)
        return out[:limit]

    def find_contact_by_email(self, email: str) -> dict | None:
        email = (email or "").strip().lower()
        if not email:
            return None
        for c in self._all("outreach_contacts", "contacts.jsonl"):
            if (c.get("email") or "").lower() == email:
                return c
        return None

    def is_opted_out(self, email: str) -> bool:
        email = (email or "").strip().lower()
        for c in self._all("outreach_contacts", "contacts.jsonl"):
            if (c.get("email") or "").lower() == email and c.get("opted_out"):
                return True
        return False

    def mark_opt_out(self, email: str) -> None:
        for c in self._all("outreach_contacts", "contacts.jsonl"):
            if (c.get("email") or "").lower() == (email or "").lower():
                self.update_contact(c["id"], opted_out=True, status="opted_out")

    # ── messages / events ──────────────────────────────────────────────────
    def add_message(self, message: dict) -> str:
        mid = message.get("id") or _new_id()
        message = {**message, "id": mid, "created_at": utcnow_iso()}
        if self._db is not None:
            self._db.collection("outreach_messages").document(mid).set(message)
        else:
            self._append_local("messages.jsonl", message, key="id")
        return mid

    def add_event(self, kind: str, detail: dict | None = None) -> None:
        ev = {"id": _new_id(), "kind": kind, "detail": detail or {}, "at": utcnow_iso()}
        if self._db is not None:
            self._db.collection("outreach_events").document(ev["id"]).set(ev)
        else:
            self._append_local("events.jsonl", ev, key="id")

    def counts(self) -> dict:
        contacts = self._all("outreach_contacts", "contacts.jsonl")
        msgs = self._all("outreach_messages", "messages.jsonl")
        by_status: dict[str, int] = {}
        for c in contacts:
            by_status[c.get("status", "new")] = by_status.get(c.get("status", "new"), 0) + 1
        return {
            "contacts": len(contacts),
            "messages": len(msgs),
            "sent": sum(1 for m in msgs if m.get("direction") == "outbound" and m.get("sent")),
            "drafted": sum(1 for m in msgs if m.get("status") == "drafted"),
            "replies": sum(1 for m in msgs if m.get("direction") == "inbound"),
            "by_status": by_status,
        }

    # ── local-backend helpers ──────────────────────────────────────────────
    def _all(self, collection: str, local_file: str) -> list[dict]:
        if self._db is not None:
            return [d.to_dict() for d in self._db.collection(collection).stream()]
        path = _LOCAL_DIR / local_file
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # local files are append-only; keep last write per id
        merged: dict[str, dict] = {}
        for r in rows:
            merged[r.get("id", _new_id())] = {**merged.get(r.get("id", ""), {}), **r}
        return list(merged.values())

    def _append_local(self, local_file: str, row: dict, key: str) -> None:
        _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        with (_LOCAL_DIR / local_file).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _merge_local(self, local_file: str, row_id: str, fields: dict) -> None:
        self._append_local(local_file, {"id": row_id, **fields}, key="id")

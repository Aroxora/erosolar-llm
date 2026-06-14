"""Command-line entry for the outreach worker.

    python -m erosolar_agent.outreach test-bridge   # verify Proton Bridge (no send)
    python -m erosolar_agent.outreach status         # show control + status + counts
    python -m erosolar_agent.outreach once           # run a single cycle
    python -m erosolar_agent.outreach run            # poll + run forever (the worker)
    python -m erosolar_agent.outreach enable          # flip control.enabled = true
    python -m erosolar_agent.outreach disable
    python -m erosolar_agent.outreach set-brief "<brief>" --audience investors --mode prospect
    python -m erosolar_agent.outreach go-live         # control.dry_run = false (guarded)
    python -m erosolar_agent.outreach dry-run         # control.dry_run = true

The worker must run on the same host as Proton Bridge. Real sends additionally
require OUTREACH_ALLOW_SEND=true in that host's environment.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import OutreachConfig
from .engine import OutreachEngine
from .mail import ProtonBridgeMail
from .store import Store


def _cfg() -> OutreachConfig:
    return OutreachConfig.load()


def cmd_test_bridge(_args) -> int:
    cfg = _cfg()
    if not cfg.bridge_ready():
        print("PROTON_BRIDGE_USER / PROTON_BRIDGE_PASSWORD not set in .env", file=sys.stderr)
        return 2
    print(cfg.summary())
    report = ProtonBridgeMail(cfg).test_connection()
    print(json.dumps(report, indent=2))
    return 0 if report["imap"] and report["smtp"] else 1


def cmd_status(_args) -> int:
    store = Store(_cfg())
    print(f"backend: {store.backend}")
    print("control:", json.dumps(store.get_control(), indent=2))
    print("counts:", json.dumps(store.counts(), indent=2))
    return 0


def cmd_once(_args) -> int:
    print(json.dumps(OutreachEngine(_cfg()).run_once(), indent=2))
    return 0


def cmd_track_inbox(args) -> int:
    n = OutreachEngine(_cfg()).track_inbox(limit=args.limit, unseen_only=False)
    print(f"tracked {n} emails to inbox_log")
    return 0


def cmd_run(_args) -> int:
    OutreachEngine(_cfg()).loop()
    return 0


def cmd_enable(_args) -> int:
    print(Store(_cfg()).set_control(enabled=True, updated_by="cli"))
    return 0


def cmd_disable(_args) -> int:
    print(Store(_cfg()).set_control(enabled=False, updated_by="cli"))
    return 0


def cmd_set_brief(args) -> int:
    print(
        Store(_cfg()).set_control(
            brief=args.brief, audience=args.audience, mode=args.mode, updated_by="cli"
        )
    )
    return 0


def cmd_go_live(_args) -> int:
    cfg = _cfg()
    if not cfg.allow_send:
        print(
            "Refusing: OUTREACH_ALLOW_SEND is not 'true' in this environment. "
            "Live sending stays disabled until the operator sets it where Bridge runs.",
            file=sys.stderr,
        )
        return 2
    print(Store(cfg).set_control(dry_run=False, updated_by="cli"))
    print("LIVE SENDING ENABLED — the worker will now deliver real email.")
    return 0


def cmd_dry_run(_args) -> int:
    print(Store(_cfg()).set_control(dry_run=True, updated_by="cli"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="erosolar_agent.outreach", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test-bridge").set_defaults(fn=cmd_test_bridge)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("once").set_defaults(fn=cmd_once)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    ti = sub.add_parser("track-inbox")
    ti.add_argument("--limit", type=int, default=200)
    ti.set_defaults(fn=cmd_track_inbox)
    sub.add_parser("enable").set_defaults(fn=cmd_enable)
    sub.add_parser("disable").set_defaults(fn=cmd_disable)
    sb = sub.add_parser("set-brief")
    sb.add_argument("brief")
    sb.add_argument("--audience", default="investors")
    sb.add_argument("--mode", default="prospect", choices=["prospect", "list"])
    sb.set_defaults(fn=cmd_set_brief)
    sub.add_parser("go-live").set_defaults(fn=cmd_go_live)
    sub.add_parser("dry-run").set_defaults(fn=cmd_dry_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

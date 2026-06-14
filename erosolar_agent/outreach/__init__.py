"""Agentic email outreach for erosolar.

A safe-by-default worker that discovers leads (Tavily), drafts personalized email
(DeepSeek), and sends via the local Proton Mail Bridge — storing the full history
in Firestore and a local RAG index. Toggled live from the web admin nav.

Live sending is double-gated (env OUTREACH_ALLOW_SEND + Firestore control.dry_run);
the default is drafts-only. See ``cli.py`` for commands and ``README.md`` for setup.
"""

from .config import OutreachConfig
from .engine import OutreachEngine

__all__ = ["OutreachConfig", "OutreachEngine"]

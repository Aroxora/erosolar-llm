#!/usr/bin/env bash
# Turnkey outreach quickstart — avoids copy/paste line-wrap issues.
#   bash outreach_quickstart.sh                 # uses leads.csv + default brief (DRY-RUN, no send)
#   bash outreach_quickstart.sh "my brief"      # custom brief
# It imports your curated leads.csv, sets the brief, runs ONE dry-run cycle
# (drafts only — nothing is sent), and prints status. Going live stays a separate
# manual step (OUTREACH_ALLOW_SEND=true && go-live && run).
set -euo pipefail
cd "$(dirname "$0")"

BRIEF="${1:-Erosolar — an open-source small-LLM pipeline + agent stack — seeking intros to seed/pre-seed investors backing open-source AI infrastructure}"

if [ -f leads.csv ]; then
  python3 -m erosolar_agent.outreach import-contacts leads.csv --audience investors
else
  echo "no leads.csv found — copy leads.example.csv to leads.csv and fill in real emails:"
  echo "    cp leads.example.csv leads.csv"
fi

python3 -m erosolar_agent.outreach set-brief "$BRIEF" --audience investors
python3 -m erosolar_agent.outreach once
python3 -m erosolar_agent.outreach status
echo
echo "Drafts are in Firestore/portal (dry-run — nothing sent)."
echo "To send for real: sed -i '' 's/^OUTREACH_ALLOW_SEND=.*/OUTREACH_ALLOW_SEND=true/' .env && python3 -m erosolar_agent.outreach go-live && python3 -m erosolar_agent.outreach run"

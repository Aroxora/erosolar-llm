#!/usr/bin/env bash
# Turnkey outreach — no multi-line copy/paste to trip on.
#   bash outreach_quickstart.sh                 # DRY-RUN: import + draft + status (nothing sent)
#   bash outreach_quickstart.sh --live          # LIVE: send real email to curated leads.csv ONLY
#   bash outreach_quickstart.sh "custom brief"  # custom brief (dry-run)
#
# Safety: this uses mode=list, so ONLY contacts in leads.csv are ever emailed —
# never Tavily-scraped strangers. The shipped leads.csv is a single self-test to
# bo@shang.software, so `--live` first proves the path by emailing you. Add real
# rows to leads.csv for real outreach. (For agentic prospecting of strangers you'd
# set --mode prospect manually — riskier: cold-mails discovered addresses.)
set -euo pipefail
cd "$(dirname "$0")"

LIVE=0
BRIEF_DEFAULT="Erosolar — an open-source small-LLM pipeline + agent stack — seeking intros to seed/pre-seed investors backing open-source AI infrastructure"
BRIEF=""
for a in "$@"; do
  case "$a" in
    --live) LIVE=1 ;;
    *) BRIEF="$a" ;;
  esac
done
BRIEF="${BRIEF:-$BRIEF_DEFAULT}"

if [ ! -f leads.csv ]; then
  echo "no leads.csv — run: cp leads.example.csv leads.csv  (then add real emails)"; exit 1
fi

python3 -m erosolar_agent.outreach import-contacts leads.csv --audience investors
# mode=list => only curated leads.csv contacts are ever emailed (no scraped strangers)
python3 -m erosolar_agent.outreach set-brief "$BRIEF" --audience investors --mode list
python3 -m erosolar_agent.outreach enable

if [ "$LIVE" = "1" ]; then
  echo
  echo ">>> GOING LIVE — sending real email to curated leads.csv only (mode=list)."
  sed -i '' 's/^OUTREACH_ALLOW_SEND=.*/OUTREACH_ALLOW_SEND=true/' .env
  python3 -m erosolar_agent.outreach go-live
  python3 -m erosolar_agent.outreach run
else
  python3 -m erosolar_agent.outreach once
  python3 -m erosolar_agent.outreach status
  echo
  echo "Dry-run complete — nothing sent. Re-run with --live to send to leads.csv."
fi

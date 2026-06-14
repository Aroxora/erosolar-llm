# erosolar_agent.outreach — agentic email outreach

A **safe-by-default** worker that:

1. **Reads all new mail** from the local Proton Mail Bridge (IMAP).
2. **Triages** every message:
   - **Bounces / broken addresses** → conservatively fixes obvious domain typos
     (`gmial.com → gmail.com`) and re-queues, or flags the address broken. It
     never invents an address.
   - **Replies** → `deepseek-v4-pro` judges whether a follow-up is *absolutely
     sensible*. If yes, it drafts a reply. If not, it either **escalates to the
     human owner** (`bo@shang.software`) with required actions, or **closes the
     thread as a dead end** and emails a cheap `deepseek-v4-flash` summary to the
     owner.
   - **"unsubscribe"** → opt-out, never contacted again.
3. **Prospects** for new leads with Tavily from a campaign brief.
4. **Drafts** personalized outreach with `deepseek-v4-pro`.
5. **Stores the full history** in Firestore *and* a local RAG embedding index
   (`data_store/outreach_rag/`) used to personalize and de-duplicate.
6. Publishes a **live heartbeat** to Firestore so the website admin nav reflects
   engine state and lets the admin toggle it on/off.

## Safety model (sending real email is outward-facing)

Two independent gates must BOTH be open before any cold/follow-up email is sent:

| Gate | Where | Default |
|---|---|---|
| `OUTREACH_ALLOW_SEND=true` | operator env on the worker host | `false` |
| `control.dry_run == false` | Firestore (flipped from the admin nav) | `true` |

- With the defaults, the worker **drafts and logs everything but sends nothing.**
- **Owner notices** (to `bo@shang.software`) ignore `dry_run` but still require
  `OUTREACH_ALLOW_SEND=true`; otherwise they are queued for the dashboard.
- Per-run cap (`OUTREACH_MAX_PER_RUN`), minimum interval between sends
  (`OUTREACH_MIN_SEND_INTERVAL_SEC`), one-contact-once, and opt-out are enforced.

## Setup

```bash
pip install firebase-admin           # optional; falls back to local JSON store
# Proton Bridge must be running on the worker host (IMAP 127.0.0.1:1143,
# SMTP 127.0.0.1:1025). Fill the PROTON_* / OUTREACH_* / FIREBASE_* keys in .env
# (see .env.example). The bridge password lives ONLY in .env (gitignored).
```

## Commands

```bash
python -m erosolar_agent.outreach test-bridge   # verify IMAP+SMTP login (no send)
python -m erosolar_agent.outreach status         # control + status + counts
python -m erosolar_agent.outreach set-brief "seed-stage climate+AI investors who back OSS infra" --audience investors
python -m erosolar_agent.outreach enable          # control.enabled = true
python -m erosolar_agent.outreach once            # run one cycle (drafts only by default)
python -m erosolar_agent.outreach run             # the worker loop
python -m erosolar_agent.outreach go-live         # control.dry_run=false (refuses unless OUTREACH_ALLOW_SEND=true)
```

The admin (`daburu.dragon@gmail.com`) can also enable/disable, flip dry-run, and
set the brief live from the **Outreach** control in the site's top nav — changes
land in Firestore and the worker picks them up on its next poll.

## Storage layout (Firestore)

- `outreach/control` — `{enabled, dry_run, brief, audience, mode}` (admin-writable)
- `outreach/status` — worker heartbeat `{state, counts, dry_run_effective, …}`
- `outreach_contacts/*` — leads + per-contact state (`new`,`queued`,`drafted`,
  `contacted`,`followed_up`,`needs_human`,`dead_end`,`bounced`,`opted_out`)
- `outreach_messages/*` — every inbound + outbound + owner notice
- `outreach_events/*` — append-only audit log (drives the dashboard feed)

If `firebase-admin` or the service account is missing, all of the above is written
under `data_store/outreach/` as JSON instead, so the worker runs offline.

## Tests

```bash
python -m erosolar_agent.outreach.test_outreach_offline   # no network/Bridge/Firestore
```

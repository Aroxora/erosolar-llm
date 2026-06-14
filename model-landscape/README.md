# model-landscape — frontier-model deep-dives (auto-updated agentically)

Technical profiles of current frontier models, kept current by an **agent**, not by hand.

This folder exists because erosolar is a deliberately *small* model — and the only honest
way to talk about "small" is against an accurate picture of what "large" currently is. The
profiles below are refreshed by [`update.py`](./update.py), which drives **this repo's own
agent integrations** (Tavily web search → DeepSeek synthesis, the same
[`erosolar_agent.integrations`](../erosolar_agent/integrations) used by the runtime) to pull
specs from primary sources, stamp each file with an "as of" date, and cite what it found.

## Honesty rules (same as the rest of erosolar)

- Profiles are generated **only from retrieved sources** — the synthesizer is instructed to
  **never invent** parameter counts, benchmark scores, dates, or prices.
- Models that can't be confirmed are labeled `not_found` / `rumored` / `unverified` rather
  than fabricated. (Several models in this set are at or beyond a given snapshot's knowledge
  cutoff — that's exactly why the updater fetches live sources.)
- Each file records the date it was generated and the URLs it used.
- The committed `models/*.md` are **snapshots**. Treat the linked primary sources as truth;
  re-run the updater for the current picture.

## Refresh it

```bash
# one-time: put your keys in the repo-root .env (gitignored)
#   TAVILY_API_KEY=...        # web search
#   DEEPSEEK_API_KEY=...      # synthesis (OpenAI-compatible)

python model-landscape/update.py            # refresh every model in models.yaml
python model-landscape/update.py --only gpt-5.5 grok-4.3
python model-landscape/update.py --dry-run  # show the plan, write nothing
python model-landscape/update.py --reindex  # rebuild the table below, no network
```

If no keys are set or the monthly quota is exhausted, the updater **does not clobber** the
existing snapshots — it prints a friendly note, rebuilds the index from whatever files
exist, and exits cleanly. Which models are tracked is controlled entirely by
[`models.yaml`](./models.yaml) — edit it to add or remove models.

## Automating the refresh

The updater is a plain script, so any scheduler works — e.g. a weekly cron:

```cron
0 9 * * 1  cd /path/to/erosolar-llm && python model-landscape/update.py >> model-landscape/.update.log 2>&1
```

---

## Tracked models

<!-- AUTO:MODELS:START -->

_Last indexed: 2026-06-14_

| Model | Snapshot |
|---|---|
| [Build 0.1](models/build-0.1.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |
| [Claude Fable 5](models/claude-fable-5.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |
| [Claude Mythos 5](models/claude-mythos-5.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |
| [Claude Opus 4.8](models/claude-opus-4.8.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |
| [Gemini 3.1 Pro](models/gemini-3.1-pro.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |
| [Gemini 4.5 Flash](models/gemini-4.5-flash.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |
| [GPT-5.5](models/gpt-5.5.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |
| [Grok 4.3](models/grok-4.3.md) | **Status**: snapshot pending — run `python model-landscape/update.py` with TAVILY_API_KEY + DEEPSEEK_API_KEY t |

<!-- AUTO:MODELS:END -->

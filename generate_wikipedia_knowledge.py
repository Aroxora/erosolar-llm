#!/usr/bin/env python3
"""
Generate Wikipedia-style knowledge pairs using Wikipedia API titles only.

This script:
- Fetches article titles via the Wikipedia API (no scraping of article text)
- Asks gpt-5.1-codex-mini to write long, Wikipedia-style explanations
- Saves to training JSONL in data_store (append-only)
- Persists API coverage state and a titles log
"""

import argparse
import asyncio
import dbm
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
ANSWER_MARKER = "<|answer|>"
STEP_MARKER = "<|step|>"

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

DEFAULT_MODEL = os.environ.get("MODEL", "gpt-5.1-codex-mini")
DEFAULT_OUTPUT = Path("data_store/generated_training_data.jsonl")
DEFAULT_STATE = Path("data_store/wiki_api_state.json")
DEFAULT_TITLES_LOG = Path("data_store/wiki_titles_seen.jsonl")
DEFAULT_MAX_OUTPUT_TOKENS = _env_int("LONG_FORM_OUTPUT_TOKENS", _env_int("MAX_OUTPUT_TOKENS", 2000))
LONG_FORM = os.environ.get("LONG_FORM", "").lower() in {"1", "true", "yes"}

WIKI_API_BASE = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "Provably-Adversarial-LLM/1.0 (titles-only; local)"

SYSTEM_PROMPT = """You are generating training data for an AI assistant.
Write a long, Wikipedia-style encyclopedic response in a neutral tone.
Do not cite or quote sources and do not mention Wikipedia.
Do not scrape or copy text; write original content.
If the topic is ambiguous or obscure, state the ambiguity and give a general overview
without inventing precise dates, statistics, or quotes.
Include multiple sections with clear headings and smooth transitions.
The user question is profane; respond professionally.

Output format (exact):
<|think_start|>
<|step|>...
<|think_end|>
<|answer|>
..."""

if LONG_FORM:
    SYSTEM_PROMPT += (
        "\n\nLONG-FORM MODE:\n"
        "- Provide expansive, detailed answers with examples and edge cases.\n"
        "- Use available output capacity without adding filler."
    )


def get_api_base() -> str:
    base = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
    return base.rstrip("/")


def get_responses_url() -> str:
    base = get_api_base()
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/responses"


def parse_response_api_output(result: dict) -> str:
    for out in result.get("output", []):
        if out.get("type") == "message":
            for content in out.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "").strip()
    return result.get("output_text", "").strip()


def normalize_cot(content: str) -> str:
    content = content.strip()
    if THINK_START not in content:
        content = f"{THINK_START}\n{content}"
    if THINK_END not in content:
        if ANSWER_MARKER in content:
            content = content.replace(ANSWER_MARKER, f"{THINK_END}\n{ANSWER_MARKER}", 1)
        else:
            content = f"{content}\n{THINK_END}"
    if ANSWER_MARKER not in content:
        content = f"{content}\n{ANSWER_MARKER}"
    return content


def extract_answer_text(content: str) -> str:
    if ANSWER_MARKER in content:
        return content.split(ANSWER_MARKER, 1)[1].strip()
    return content.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def build_user_question(title: str) -> str:
    return f"wtf is {title}?"


def build_generation_prompt(title: str, min_words: int) -> str:
    return (
        f'User question: "wtf is {title}?"\n'
        f"Answer requirements:\n"
        f"- Minimum {min_words} words in the answer section.\n"
        "- Use a Wikipedia-style lead paragraph followed by sections.\n"
        "- Suggested sections: Overview, Background, Key Ideas/Features, Examples/Applications, "
        "Related Topics, Common Misconceptions (if any).\n"
        "- Avoid citations, quotes, and source mentions.\n"
    )


@dataclass
class WikiState:
    apcontinue: Optional[str] = None
    last_title: Optional[str] = None
    last_pageid: Optional[int] = None
    processed_count: int = 0
    language: str = "en"
    namespace: int = 0
    source: str = "allpages"
    updated_at: str = ""


def load_state(path: Path) -> WikiState:
    if not path.exists():
        return WikiState()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return WikiState(
            apcontinue=data.get("apcontinue"),
            last_title=data.get("last_title"),
            last_pageid=data.get("last_pageid"),
            processed_count=int(data.get("processed_count", 0)),
            language=data.get("language", "en"),
            namespace=int(data.get("namespace", 0)),
            source=data.get("source", "allpages"),
            updated_at=data.get("updated_at", ""),
        )
    except Exception:
        return WikiState()


def save_state(path: Path, state: WikiState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


async def fetch_wikipedia_titles(
    session: aiohttp.ClientSession,
    language: str,
    namespace: int,
    limit: int,
    apcontinue: Optional[str],
    start_from: Optional[str],
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    url = WIKI_API_BASE.format(lang=language)
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": namespace,
        "aplimit": limit,
        "apfilterredir": "nonredirects",
        "format": "json",
    }
    if apcontinue:
        params["apcontinue"] = apcontinue
    elif start_from:
        params["apfrom"] = start_from

    headers = {"User-Agent": USER_AGENT}
    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        data = await resp.json()

    pages = data.get("query", {}).get("allpages", [])
    titles = [{"title": p.get("title", ""), "pageid": p.get("pageid")} for p in pages if p.get("title")]
    next_continue = data.get("continue", {}).get("apcontinue")
    return titles, next_continue


async def call_deepseek(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    url = get_responses_url()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return parse_response_api_output(data)


async def generate_record(
    session: aiohttp.ClientSession,
    title: str,
    pageid: Optional[int],
    api_key: str,
    model: str,
    min_words: int,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
) -> Dict[str, object]:
    prompt = build_generation_prompt(title, min_words)
    last_content = ""
    for attempt in range(max_retries + 1):
        if attempt > 0:
            prompt = (
                build_generation_prompt(title, min_words)
                + "\nThe previous answer was too short. Expand substantially with more depth and sections."
            )
        try:
            content = await call_deepseek(
                session=session,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            content = normalize_cot(content)
            last_content = content
            answer_text = extract_answer_text(content)
            if word_count(answer_text) >= min_words:
                break
        except Exception as e:
            return {"success": False, "title": title, "pageid": pageid, "error": str(e)}

    user_question = build_user_question(title)
    metadata = {
        "source": "gpt-generated",
        "category": "wikipedia_knowledge",
        "type": "wikipedia_knowledge",
        "concept": title,
        "wiki_title": title,
        "wiki_pageid": pageid,
        "wiki_language": None,
        "wiki_namespace": None,
        "wiki_source": "wikipedia-api-titles-only",
        "weight": 1.0,
        "has_thinking": THINK_START in last_content,
        "has_answer": ANSWER_MARKER in last_content,
        "has_step": STEP_MARKER in last_content,
        "generated_at": datetime.now().isoformat(),
        "model": model,
    }
    record = {
        "messages": [
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": last_content},
        ],
        "metadata": metadata,
    }
    return {"success": True, "title": title, "pageid": pageid, "record": record}


def append_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_titles_log(path: Path, titles: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for item in titles:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def open_seen_db(path: Path, mode: str = "c"):
    path.parent.mkdir(parents=True, exist_ok=True)
    return dbm.open(str(path), mode)


def title_key(title: str) -> bytes:
    return title.encode("utf-8", "ignore")


def seed_seen_db(seen_db, titles_log: Path) -> int:
    seed_marker = b"__seeded__"
    if seed_marker in seen_db:
        return 0
    count = 0
    if titles_log.exists():
        with open(titles_log, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = item.get("title")
                if not title:
                    continue
                seen_db[title_key(title)] = b"1"
                count += 1
    seen_db[seed_marker] = b"1"
    return count


async def run(args: argparse.Namespace) -> int:
    if not args.dry_run:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required (or use --dry-run).")
    else:
        api_key = ""

    state = load_state(Path(args.state)) if args.resume else WikiState(language=args.language, namespace=args.namespace)
    if not state.language:
        state.language = args.language
    if state.language != args.language or state.namespace != args.namespace:
        if args.resume:
            print("Warning: state language/namespace does not match args; continuing with args.")
        state.language = args.language
        state.namespace = args.namespace

    apcontinue = state.apcontinue if args.resume else None
    start_from = args.start_from if not apcontinue else None
    target_total = args.target
    unbounded = target_total <= 0
    written_this_run = 0
    listed_this_run = 0
    processed_count = state.processed_count
    seen_db = None
    if args.skip_seen:
        seen_db_path = Path(args.seen_db)
        if args.dry_run and not seen_db_path.exists():
            seen_db = None
        else:
            db_mode = "r" if args.dry_run else "c"
            seen_db = open_seen_db(seen_db_path, db_mode)
            if not args.dry_run:
                seed_seen_db(seen_db, Path(args.titles_log))

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                if not unbounded:
                    progress = written_this_run if not args.dry_run else listed_this_run
                    if progress >= target_total:
                        break
                    limit = min(args.wiki_batch, target_total - progress)
                else:
                    limit = args.wiki_batch

                fetched_titles, next_continue = await fetch_wikipedia_titles(
                    session=session,
                    language=args.language,
                    namespace=args.namespace,
                    limit=limit,
                    apcontinue=apcontinue,
                    start_from=start_from,
                )
                start_from = None
                if not fetched_titles:
                    break

                titles = fetched_titles
                if seen_db is not None:
                    filtered = []
                    for item in titles:
                        title = item.get("title", "")
                        if not title:
                            continue
                        if title_key(title) in seen_db:
                            continue
                        filtered.append(item)
                    titles = filtered

                if args.dry_run:
                    for item in titles:
                        print(f"{item.get('title')} ({item.get('pageid')})")
                    listed_this_run += len(titles)
                else:
                    if not titles:
                        processed_count += len(fetched_titles)
                    else:
                        sem = asyncio.Semaphore(args.workers)
                        results: List[Dict[str, object]] = []

                        async def bounded_generate(item: Dict[str, object]) -> None:
                            async with sem:
                                await asyncio.sleep(args.rate_limit_delay)
                                result = await generate_record(
                                    session=session,
                                    title=item.get("title", ""),
                                    pageid=item.get("pageid"),
                                    api_key=api_key,
                                    model=args.model,
                                    min_words=args.min_words,
                                    max_output_tokens=args.max_output_tokens,
                                    temperature=args.temperature,
                                    max_retries=args.max_retries,
                                )
                                results.append(result)

                        tasks = [bounded_generate(item) for item in titles]
                        await asyncio.gather(*tasks)

                        success_records = [r["record"] for r in results if r.get("success")]
                        if success_records:
                            for record in success_records:
                                record["metadata"]["wiki_language"] = args.language
                                record["metadata"]["wiki_namespace"] = args.namespace
                            append_jsonl(Path(args.output), success_records)

                        log_items = []
                        for r in results:
                            if r.get("success"):
                                title = r.get("title") or ""
                                log_items.append(
                                    {
                                        "title": title,
                                        "pageid": r.get("pageid"),
                                        "generated_at": datetime.now().isoformat(),
                                    }
                                )
                                if seen_db is not None and title:
                                    seen_db[title_key(title)] = b"1"
                        if log_items:
                            append_titles_log(Path(args.titles_log), log_items)

                        written_this_run += len(success_records)
                        processed_count += len(fetched_titles)

                if not args.dry_run:
                    if fetched_titles:
                        last = fetched_titles[-1]
                        state.last_title = last.get("title")
                        state.last_pageid = last.get("pageid")
                    state.apcontinue = next_continue
                    state.processed_count = processed_count
                    state.updated_at = datetime.now().isoformat()
                    save_state(Path(args.state), state)

                apcontinue = next_continue
                if not apcontinue:
                    break
    finally:
        if seen_db is not None:
            seen_db.close()

    if args.dry_run:
        print(f"Dry run complete. Titles listed: {listed_this_run}.")
    else:
        print(f"Generation complete. Records written: {written_this_run}.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate long Wikipedia-style knowledge pairs using Wikipedia API titles only.",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output JSONL path (append-only).")
    parser.add_argument("--state", type=str, default=str(DEFAULT_STATE),
                        help="Persistent Wikipedia API state file.")
    parser.add_argument("--titles-log", type=str, default=str(DEFAULT_TITLES_LOG),
                        help="Append-only log of processed titles.")
    parser.add_argument("--seen-db", type=str, default="data_store/wiki_titles_seen.db",
                        help="Persistent seen-title index (dbm).")
    parser.add_argument("--target", type=int, default=10,
                        help="How many records to generate this run (<=0 means run until API ends).")
    parser.add_argument("--wiki-batch", type=int, default=25,
                        help="How many titles to fetch per Wikipedia API call.")
    parser.add_argument("--language", type=str, default="en",
                        help="Wikipedia language code (default: en).")
    parser.add_argument("--namespace", type=int, default=0,
                        help="Wikipedia namespace (default: 0, articles).")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Start from a specific title (apfrom) if not resuming.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume using saved apcontinue state (default: true).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not resume; start from the beginning or --start-from.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list titles (no DeepSeek calls or output writes).")
    parser.add_argument("--skip-seen", dest="skip_seen", action="store_true", default=True,
                        help="Skip titles already seen in the persistent index (default: true).")
    parser.add_argument("--no-skip-seen", dest="skip_seen", action="store_false",
                        help="Do not skip seen titles.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="DeepSeek model to use.")
    parser.add_argument("--min-words", type=int, default=800,
                        help="Minimum words required in the answer section.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
                        help="Max output tokens for the DeepSeek response.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Retries if the answer is too short.")
    parser.add_argument("--workers", type=int, default=5,
                        help="Concurrent DeepSeek requests.")
    parser.add_argument("--rate-limit-delay", type=float, default=0.05,
                        help="Delay (seconds) between request starts.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

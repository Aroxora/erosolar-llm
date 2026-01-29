#!/usr/bin/env python3
"""
Download and process full English Wikipedia for LLM training.

Creates instruction-following pairs from Wikipedia articles:
- "What is X?" -> Article summary
- "Explain X" -> Article content
- "Tell me about X" -> Article content

This provides comprehensive world knowledge for the model.
"""

import bz2
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator, List, Tuple, Optional
from dataclasses import dataclass
import random

# Try imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False
    print("Note: Install mwparserfromhell for better Wikipedia parsing: pip install mwparserfromhell")

CACHE_DIR = Path("cache/wikipedia")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Wikipedia dump URLs
WIKI_DUMP_BASE = "https://dumps.wikimedia.org/enwiki/latest/"
WIKI_ARTICLES_DUMP = "enwiki-latest-pages-articles.xml.bz2"

# For faster iteration, use the smaller abstract dump first
WIKI_ABSTRACT_DUMP = "enwiki-latest-abstract.xml.gz"

# Simple Wikipedia for cleaner content
SIMPLE_WIKI_DUMP = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"


@dataclass
class WikiArticle:
    title: str
    text: str
    categories: List[str]


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress."""
    print(f"Downloading {url}...")
    try:
        if HAS_REQUESTS:
            response = requests.get(url, stream=True, timeout=3600)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = (downloaded / total) * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded / 1e9:.2f} GB)", end='', flush=True)
            print()
        else:
            urllib.request.urlretrieve(url, filepath)
        print(f"  Saved to {filepath}")
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def clean_wikitext(text: str) -> str:
    """Clean Wikipedia markup to plain text."""
    if HAS_MWPARSER:
        try:
            wikicode = mwparserfromhell.parse(text)
            # Remove templates, references, etc.
            for template in wikicode.filter_templates():
                try:
                    wikicode.remove(template)
                except ValueError:
                    pass
            text = wikicode.strip_code()
        except Exception:
            pass

    # Basic cleanup without mwparserfromhell
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/?>', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove wiki links but keep text: [[Link|Text]] -> Text, [[Link]] -> Link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

    # Remove external links
    text = re.sub(r'\[https?://[^\]]+\]', '', text)

    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]+\}\}', '', text)

    # Remove categories
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)

    # Remove files/images
    text = re.sub(r'\[\[(File|Image):[^\]]+\]\]', '', text, flags=re.IGNORECASE)

    # Remove bold/italic markup
    text = re.sub(r"'''?", '', text)

    # Remove section headers formatting
    text = re.sub(r'={2,}(.+?)={2,}', r'\1', text)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def extract_first_paragraph(text: str) -> str:
    """Extract the first meaningful paragraph as a summary."""
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        # Skip short paragraphs, headers, lists
        if len(para) > 100 and not para.startswith(('*', '#', '|', '!')):
            # Truncate very long paragraphs
            if len(para) > 1000:
                # Find a good breaking point
                sentences = para.split('. ')
                result = ''
                for sent in sentences:
                    if len(result) + len(sent) < 800:
                        result += sent + '. '
                    else:
                        break
                return result.strip()
            return para
    return text[:500] if len(text) > 500 else text


def parse_wiki_dump(filepath: Path, max_articles: int = None) -> Generator[WikiArticle, None, None]:
    """Parse Wikipedia XML dump and yield articles."""
    print(f"Parsing Wikipedia dump: {filepath}")

    # Detect compression
    if str(filepath).endswith('.bz2'):
        opener = bz2.open
    elif str(filepath).endswith('.gz'):
        import gzip
        opener = gzip.open
    else:
        opener = open

    count = 0
    current_title = None
    current_text = None
    in_page = False

    with opener(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for event, elem in ET.iterparse(f, events=('start', 'end')):
            # Handle namespace prefix
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if event == 'start' and tag == 'page':
                in_page = True
                current_title = None
                current_text = None

            elif event == 'end':
                if tag == 'title' and in_page:
                    current_title = elem.text

                elif tag == 'text' and in_page:
                    current_text = elem.text or ''

                elif tag == 'page':
                    in_page = False

                    # Skip non-article pages
                    if current_title and current_text:
                        # Skip redirects, talk pages, special pages
                        if current_text.lower().startswith('#redirect'):
                            continue
                        if ':' in current_title:
                            prefix = current_title.split(':')[0].lower()
                            if prefix in ('talk', 'user', 'wikipedia', 'file', 'mediawiki',
                                        'template', 'help', 'category', 'portal', 'draft',
                                        'module', 'timedtext', 'gadget'):
                                continue

                        # Skip very short articles
                        if len(current_text) < 500:
                            continue

                        # Extract categories
                        categories = re.findall(r'\[\[Category:([^\]|]+)', current_text)

                        # Clean the text
                        clean_text = clean_wikitext(current_text)

                        if len(clean_text) > 200:
                            yield WikiArticle(
                                title=current_title,
                                text=clean_text,
                                categories=categories
                            )
                            count += 1

                            if count % 10000 == 0:
                                print(f"  Processed {count:,} articles...")

                            if max_articles and count >= max_articles:
                                return

                    # Clear memory
                    elem.clear()


def article_to_instruction_pairs(article: WikiArticle) -> List[Tuple[str, str]]:
    """Convert a Wikipedia article to instruction-following pairs."""
    pairs = []
    title = article.title
    text = article.text
    summary = extract_first_paragraph(text)

    # Truncate very long articles for training efficiency
    if len(text) > 4000:
        text = text[:4000] + "..."

    # 1. "What is X?" -> Summary
    pairs.append((
        f"What is {title}?",
        summary
    ))

    # 2. "Explain X" -> Full content
    if len(text) > len(summary) + 100:
        pairs.append((
            f"Explain {title} in detail.",
            text
        ))

    # 3. "Tell me about X" -> Content
    pairs.append((
        f"Tell me about {title}.",
        text if len(text) < 2000 else summary
    ))

    # 4. Definition style
    pairs.append((
        f"Define {title}.",
        summary[:500] if len(summary) > 500 else summary
    ))

    # 5. Question variations
    question_templates = [
        f"What do you know about {title}?",
        f"Can you explain {title}?",
        f"Describe {title}.",
        f"Give me information about {title}.",
    ]

    # Add 1-2 random variations
    for template in random.sample(question_templates, min(2, len(question_templates))):
        pairs.append((template, summary))

    return pairs


def download_and_process_wikipedia(
    use_simple: bool = False,
    max_articles: int = None,
    output_file: str = "wikipedia_instructions.json"
) -> List[Tuple[str, str]]:
    """Download and process Wikipedia into instruction pairs."""

    # Choose dump
    if use_simple:
        dump_url = SIMPLE_WIKI_DUMP
        dump_file = CACHE_DIR / "simplewiki-latest-pages-articles.xml.bz2"
        print("Using Simple English Wikipedia (smaller, cleaner)")
    else:
        dump_url = WIKI_DUMP_BASE + WIKI_ARTICLES_DUMP
        dump_file = CACHE_DIR / WIKI_ARTICLES_DUMP
        print("Using full English Wikipedia")

    # Download if not exists
    if not dump_file.exists():
        print(f"\nDownloading Wikipedia dump (~20GB for full, ~200MB for simple)...")
        print("This may take a while...")
        if not download_file(dump_url, dump_file):
            print("Failed to download Wikipedia dump")
            return []
    else:
        print(f"Using cached dump: {dump_file}")

    # Process articles
    print("\nProcessing Wikipedia articles...")
    all_pairs = []

    for article in parse_wiki_dump(dump_file, max_articles):
        pairs = article_to_instruction_pairs(article)
        all_pairs.extend(pairs)

        if len(all_pairs) % 100000 == 0:
            print(f"  Generated {len(all_pairs):,} instruction pairs...")

    print(f"\nTotal instruction pairs from Wikipedia: {len(all_pairs):,}")

    # Save to cache
    output_path = CACHE_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False)
    print(f"Saved to {output_path}")

    return all_pairs


def download_wikipedia_abstracts() -> List[Tuple[str, str]]:
    """Download Wikipedia abstracts (faster alternative to full dump).

    Abstracts contain the first paragraph of each article - good for summaries.
    """
    import gzip

    abstract_file = CACHE_DIR / "enwiki-latest-abstract.xml.gz"
    abstract_url = WIKI_DUMP_BASE + "enwiki-latest-abstract.xml.gz"

    if not abstract_file.exists():
        print("Downloading Wikipedia abstracts (~800MB)...")
        if not download_file(abstract_url, abstract_file):
            return []

    print("Processing Wikipedia abstracts...")
    pairs = []

    with gzip.open(abstract_file, 'rt', encoding='utf-8', errors='replace') as f:
        current_title = None
        current_abstract = None

        for line in f:
            line = line.strip()

            if '<title>' in line:
                # Extract title: <title>Wikipedia: Article Name</title>
                match = re.search(r'<title>Wikipedia: (.+?)</title>', line)
                if match:
                    current_title = match.group(1)

            elif '<abstract>' in line:
                # Extract abstract
                match = re.search(r'<abstract>(.+?)</abstract>', line, re.DOTALL)
                if match:
                    current_abstract = match.group(1).strip()

                    if current_title and current_abstract and len(current_abstract) > 50:
                        # Clean abstract
                        abstract = re.sub(r'<[^>]+>', '', current_abstract)

                        # Create instruction pairs
                        pairs.append((
                            f"What is {current_title}?",
                            abstract
                        ))
                        pairs.append((
                            f"Define {current_title}.",
                            abstract[:500] if len(abstract) > 500 else abstract
                        ))

                        if len(pairs) % 100000 == 0:
                            print(f"  Processed {len(pairs):,} pairs...")

                    current_title = None
                    current_abstract = None

    print(f"Generated {len(pairs):,} pairs from Wikipedia abstracts")

    # Save
    output_path = CACHE_DIR / "wikipedia_abstracts.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"Saved to {output_path}")

    return pairs


def get_wikipedia_data(full: bool = False, max_articles: int = None) -> List[Tuple[str, str]]:
    """Get Wikipedia instruction data, downloading if necessary.

    Args:
        full: If True, use full Wikipedia. If False, use abstracts (faster).
        max_articles: Maximum articles to process (None = all)
    """
    if full:
        cache_file = CACHE_DIR / "wikipedia_instructions.json"
        if cache_file.exists():
            print(f"Loading cached Wikipedia data from {cache_file}...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return download_and_process_wikipedia(max_articles=max_articles)
    else:
        cache_file = CACHE_DIR / "wikipedia_abstracts.json"
        if cache_file.exists():
            print(f"Loading cached Wikipedia abstracts from {cache_file}...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return download_wikipedia_abstracts()


def combine_wikipedia_and_instructions(
    wiki_data: List[Tuple[str, str]],
    instruction_data: List[Tuple[str, str]],
    wiki_ratio: float = 0.3
) -> List[Tuple[str, str]]:
    """Combine Wikipedia and instruction data with optimal ratio.

    For a small LLM, we want:
    - Strong instruction-following ability (70%)
    - Comprehensive knowledge from Wikipedia (30%)

    Args:
        wiki_data: Wikipedia instruction pairs
        instruction_data: Regular instruction pairs
        wiki_ratio: Ratio of Wikipedia data (0.3 = 30%)
    """
    print(f"\nCombining datasets:")
    print(f"  Wikipedia: {len(wiki_data):,} pairs")
    print(f"  Instructions: {len(instruction_data):,} pairs")

    # Calculate how much of each to use
    total_target = len(wiki_data) + len(instruction_data)
    wiki_target = int(total_target * wiki_ratio)
    inst_target = total_target - wiki_target

    # Sample if needed
    if len(wiki_data) > wiki_target:
        wiki_sample = random.sample(wiki_data, wiki_target)
    else:
        wiki_sample = wiki_data

    if len(instruction_data) > inst_target:
        inst_sample = random.sample(instruction_data, inst_target)
    else:
        inst_sample = instruction_data

    # Combine and shuffle
    combined = wiki_sample + inst_sample
    random.shuffle(combined)

    print(f"  Combined: {len(combined):,} pairs")
    print(f"  Wiki ratio: {len(wiki_sample) / len(combined) * 100:.1f}%")

    return combined


def run_full_pipeline():
    """Run the full pipeline: download full Wikipedia + combine with instructions."""
    print("=" * 60)
    print("FULL WIKIPEDIA + INSTRUCTIONS PIPELINE")
    print("=" * 60)

    # Step 1: Download and process full English Wikipedia
    print("\n[STEP 1] Downloading Full English Wikipedia")
    print("WARNING: This is a ~22GB download and will take hours to process")
    print("-" * 60)

    wiki_data = download_and_process_wikipedia(use_simple=False, max_articles=None)

    print(f"\nWikipedia pairs generated: {len(wiki_data):,}")

    # Step 2: Load existing instruction data
    print("\n[STEP 2] Loading Instruction Data")
    print("-" * 60)

    instruction_file = Path("cache/datasets/instruction_data.json")
    instruction_data = []

    if instruction_file.exists():
        with open(instruction_file, 'r', encoding='utf-8') as f:
            instruction_data = [tuple(x) for x in json.load(f)]
        print(f"  Loaded {len(instruction_data):,} instruction pairs")
    else:
        print("  No instruction data found - using Wikipedia only")

    # Step 3: Combine with higher Wikipedia ratio (it's the knowledge source)
    print("\n[STEP 3] Combining Datasets")
    print("-" * 60)

    # For a knowledge-focused model, use more Wikipedia
    # Wikipedia = knowledge, Instructions = task completion
    combined = combine_wikipedia_and_instructions(
        wiki_data,
        instruction_data,
        wiki_ratio=0.6  # 60% Wikipedia for comprehensive knowledge
    )

    # Step 4: Save final combined dataset
    print("\n[STEP 4] Saving Final Dataset")
    print("-" * 60)

    output_file = Path("cache/datasets/combined_training_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_file}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Total pairs: {len(combined):,}")

    # Also save Wikipedia-only data
    wiki_output = Path("cache/datasets/wikipedia_knowledge.json")
    with open(wiki_output, 'w', encoding='utf-8') as f:
        json.dump(wiki_data, f, ensure_ascii=False)
    wiki_size = wiki_output.stat().st_size / (1024 * 1024)
    print(f"\n  Wikipedia data: {wiki_output}")
    print(f"  Size: {wiki_size:.1f} MB")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"  Wikipedia pairs: {len(wiki_data):,}")
    print(f"  Instruction pairs: {len(instruction_data):,}")
    print(f"  Combined pairs: {len(combined):,}")

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and process Wikipedia for LLM training")
    parser.add_argument("--full", action="store_true", help="Download full Wikipedia (20GB+)")
    parser.add_argument("--simple", action="store_true", help="Use Simple English Wikipedia (smaller)")
    parser.add_argument("--abstracts", action="store_true", help="Use abstracts only (fastest)")
    parser.add_argument("--max", type=int, default=None, help="Max articles to process")
    parser.add_argument("--combine", action="store_true", help="Combine with instruction data")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline (Wikipedia + instructions)")

    args = parser.parse_args()

    if args.pipeline:
        run_full_pipeline()
    elif args.abstracts:
        print("Downloading Wikipedia abstracts (fastest option)...")
        data = download_wikipedia_abstracts()
        if args.combine:
            instruction_file = Path("cache/datasets/instruction_data.json")
            if instruction_file.exists():
                with open(instruction_file, 'r', encoding='utf-8') as f:
                    instruction_data = json.load(f)
                combined = combine_wikipedia_and_instructions(data, instruction_data)
                output_file = Path("cache/datasets/combined_training_data.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(combined, f, ensure_ascii=False)
                print(f"\nSaved combined data to {output_file}")
        print(f"\nTotal Wikipedia pairs: {len(data):,}")
    elif args.simple:
        print("Downloading Simple English Wikipedia...")
        data = download_and_process_wikipedia(use_simple=True, max_articles=args.max)
        print(f"\nTotal Wikipedia pairs: {len(data):,}")
    elif args.full:
        print("Downloading full English Wikipedia...")
        data = download_and_process_wikipedia(use_simple=False, max_articles=args.max)
        print(f"\nTotal Wikipedia pairs: {len(data):,}")
    else:
        # Default: run full pipeline
        print("Running full pipeline (--pipeline)...")
        print("Options available:")
        print("  --pipeline: Full Wikipedia + instructions (default)")
        print("  --abstracts: ~800MB, fastest, first paragraph of each article")
        print("  --simple: ~200MB, Simple English Wikipedia")
        print("  --full: ~20GB, complete English Wikipedia")
        print("  --combine: Combine with instruction data")
        print("")
        run_full_pipeline()

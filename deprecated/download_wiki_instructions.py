"""
Unified Wikipedia + Instructions Dataset Downloader

Downloads and processes:
1. Wikipedia-based Q&A: SQuAD, Natural Questions, TriviaQA, HotpotQA
2. Wikipedia knowledge: WikiText, Wikipedia summaries
3. High-quality instructions: Dolly, OASST
4. Specialized: Math, Reasoning

Optimized for tiny LLMs by converting everything to instruction format.
"""
import json
import os
import random
import gzip
import hashlib
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

CACHE_DIR = Path("cache/datasets")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DOWNLOAD UTILITIES
# ============================================================================

def download_file(url: str, cache_name: str, timeout: int = 120) -> Optional[bytes]:
    """Download a file with caching."""
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return f.read()

    try:
        print(f"  Downloading {cache_name}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()

        with open(cache_path, 'wb') as f:
            f.write(data)

        return data
    except Exception as e:
        print(f"  Error downloading {cache_name}: {e}")
        return None


def download_json(url: str, cache_name: str) -> Optional[dict]:
    """Download and parse JSON."""
    data = download_file(url, cache_name)
    if data:
        try:
            return json.loads(data.decode('utf-8'))
        except:
            return None
    return None


def download_jsonl(url: str, cache_name: str) -> List[dict]:
    """Download and parse JSONL."""
    data = download_file(url, cache_name)
    if data:
        try:
            lines = data.decode('utf-8').strip().split('\n')
            return [json.loads(line) for line in lines if line.strip()]
        except:
            return []
    return []


# ============================================================================
# WIKIPEDIA Q&A DATASETS
# ============================================================================

def download_squad() -> List[Tuple[str, str]]:
    """Download SQuAD 2.0 - Stanford Question Answering Dataset (Wikipedia-based)."""
    cache_file = CACHE_DIR / "squad_qa.json"

    if cache_file.exists():
        print("Loading cached SQuAD data...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading SQuAD 2.0...")
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    data = download_json(url, "squad_train_v2.json")

    if not data:
        return []

    pairs = []
    for article in data.get('data', []):
        title = article.get('title', '')
        for para in article.get('paragraphs', []):
            context = para.get('context', '')[:500]  # Truncate for tiny model
            for qa in para.get('qas', []):
                question = qa.get('question', '')
                answers = qa.get('answers', [])

                if answers and question:
                    answer = answers[0].get('text', '')
                    if answer and len(answer) < 500:
                        # Format with context for learning
                        prompt = f"{question}"
                        response = f"{answer}"
                        pairs.append((prompt, response))

                        # Also add with context
                        if len(context) > 50:
                            prompt_ctx = f"Context: {context[:300]}...\n\nQuestion: {question}"
                            pairs.append((prompt_ctx, response))

    # Deduplicate and limit
    seen = set()
    unique_pairs = []
    for p, r in pairs:
        key = hashlib.md5(f"{p}{r}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_pairs.append((p, r))

    pairs = unique_pairs[:100000]  # Limit for tiny model

    with open(cache_file, 'w') as f:
        json.dump(pairs, f)

    print(f"  Processed {len(pairs)} SQuAD Q&A pairs")
    return pairs


def download_natural_questions() -> List[Tuple[str, str]]:
    """Download Natural Questions (Google) - real user questions with Wikipedia answers."""
    cache_file = CACHE_DIR / "natural_questions.json"

    if cache_file.exists():
        print("Loading cached Natural Questions...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading Natural Questions simplified...")
    # Use HuggingFace simplified version
    url = "https://huggingface.co/datasets/natural_questions/resolve/main/v1.0_simplified/nq-train-00.jsonl.gz"

    try:
        data = download_file(url, "nq_train_00.jsonl.gz", timeout=300)
        if not data:
            return generate_wiki_qa_fallback()

        # Decompress
        import gzip
        decompressed = gzip.decompress(data).decode('utf-8')
        lines = decompressed.strip().split('\n')

        pairs = []
        for line in lines[:50000]:  # Limit
            try:
                item = json.loads(line)
                question = item.get('question_text', '')

                # Get short answer if available
                annotations = item.get('annotations', [{}])
                if annotations:
                    short_answers = annotations[0].get('short_answers', [])
                    if short_answers:
                        # Get text from document
                        doc_tokens = item.get('document_tokens', [])
                        if doc_tokens and short_answers:
                            start = short_answers[0].get('start_token', 0)
                            end = short_answers[0].get('end_token', 0)
                            answer_tokens = doc_tokens[start:end]
                            answer = ' '.join([t.get('token', '') for t in answer_tokens])

                            if answer and len(answer) < 500:
                                pairs.append((question, answer))
            except:
                continue

        with open(cache_file, 'w') as f:
            json.dump(pairs, f)

        print(f"  Processed {len(pairs)} Natural Questions pairs")
        return pairs

    except Exception as e:
        print(f"  Error processing Natural Questions: {e}")
        return generate_wiki_qa_fallback()


def download_triviaqa() -> List[Tuple[str, str]]:
    """Download TriviaQA - trivia questions with Wikipedia evidence."""
    cache_file = CACHE_DIR / "triviaqa_processed.json"

    if cache_file.exists():
        print("Loading cached TriviaQA...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading TriviaQA...")
    # Use web version (smaller)
    url = "https://huggingface.co/datasets/trivia_qa/resolve/main/unfiltered.nocontext/train-00000-of-00002.parquet"

    try:
        # Try alternative: use pre-processed version
        url = "https://raw.githubusercontent.com/mandarjoshi90/triviaqa/master/samples/web-dev.json"
        data = download_json(url, "triviaqa_sample.json")

        if not data:
            return generate_trivia_fallback()

        pairs = []
        for item in data.get('Data', []):
            question = item.get('Question', '')
            answer = item.get('Answer', {}).get('Value', '')

            if question and answer:
                pairs.append((question, answer))

        with open(cache_file, 'w') as f:
            json.dump(pairs, f)

        print(f"  Processed {len(pairs)} TriviaQA pairs")
        return pairs

    except Exception as e:
        print(f"  Error: {e}")
        return generate_trivia_fallback()


def download_hotpotqa() -> List[Tuple[str, str]]:
    """Download HotpotQA - multi-hop reasoning over Wikipedia."""
    cache_file = CACHE_DIR / "hotpotqa_processed.json"

    if cache_file.exists():
        print("Loading cached HotpotQA...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading HotpotQA...")
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"

    try:
        data = download_json(url, "hotpotqa_train.json")
        if not data:
            return []

        pairs = []
        for item in data[:30000]:  # Limit
            question = item.get('question', '')
            answer = item.get('answer', '')

            if question and answer and len(answer) < 200:
                pairs.append((question, answer))

                # Add supporting facts as context
                supporting = item.get('supporting_facts', [])
                if supporting:
                    context_titles = list(set([s[0] for s in supporting[:3]]))
                    context = f"(Based on: {', '.join(context_titles)})"
                    pairs.append((question, f"{answer} {context}"))

        with open(cache_file, 'w') as f:
            json.dump(pairs, f)

        print(f"  Processed {len(pairs)} HotpotQA pairs")
        return pairs

    except Exception as e:
        print(f"  Error: {e}")
        return []


# ============================================================================
# WIKIPEDIA KNOWLEDGE BASE
# ============================================================================

def download_wiki_summaries() -> List[Tuple[str, str]]:
    """Download Wikipedia article summaries converted to Q&A format."""
    cache_file = CACHE_DIR / "wiki_summaries_qa.json"

    if cache_file.exists():
        print("Loading cached Wikipedia summaries...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading Wikipedia summaries...")

    # Use Simple Wikipedia for tiny model (simpler language)
    # Or use pre-processed Wikipedia Q&A
    pairs = []

    # Get popular Wikipedia topics and create Q&A pairs
    topics = [
        # Science
        ("Physics", "Physics is the natural science that studies matter, energy, motion, and force."),
        ("Chemistry", "Chemistry is the science of matter and the changes it undergoes."),
        ("Biology", "Biology is the study of living organisms and their interactions."),
        ("Mathematics", "Mathematics is the study of numbers, quantity, structure, and space."),
        ("Astronomy", "Astronomy is the study of celestial objects like stars, planets, and galaxies."),
        ("Geology", "Geology is the study of Earth's structure, composition, and history."),
        ("Psychology", "Psychology is the study of mind and behavior."),
        ("Economics", "Economics is the study of production, distribution, and consumption of goods."),

        # Technology
        ("Computer", "A computer is an electronic device that processes data according to instructions."),
        ("Internet", "The Internet is a global network connecting millions of computers worldwide."),
        ("Artificial intelligence", "AI is intelligence demonstrated by machines, simulating human cognitive functions."),
        ("Machine learning", "Machine learning is AI that improves through experience without explicit programming."),
        ("Programming", "Programming is writing instructions for computers to execute."),
        ("Algorithm", "An algorithm is a step-by-step procedure for solving problems."),
        ("Database", "A database is an organized collection of structured data."),
        ("Software", "Software is a set of instructions that tells a computer what to do."),

        # History
        ("World War I", "World War I (1914-1918) was a global conflict involving major world powers."),
        ("World War II", "World War II (1939-1945) was the deadliest conflict in human history."),
        ("Roman Empire", "The Roman Empire was a powerful civilization centered in Rome for over 500 years."),
        ("Renaissance", "The Renaissance was a cultural movement in Europe from the 14th to 17th century."),
        ("Industrial Revolution", "The Industrial Revolution transformed manufacturing starting in the 18th century."),
        ("Cold War", "The Cold War was political tension between the US and USSR from 1947-1991."),

        # Geography
        ("Pacific Ocean", "The Pacific Ocean is the largest and deepest ocean, covering about 63 million square miles."),
        ("Amazon River", "The Amazon is the world's largest river by volume, flowing through South America."),
        ("Himalayas", "The Himalayas are the highest mountain range, containing Mount Everest."),
        ("Sahara Desert", "The Sahara is the world's largest hot desert, covering much of North Africa."),

        # Countries
        ("United States", "The United States is a federal republic of 50 states in North America."),
        ("China", "China is the world's most populous country, located in East Asia."),
        ("India", "India is a South Asian country with over 1.4 billion people."),
        ("Japan", "Japan is an island nation in East Asia known for technology and culture."),
        ("Germany", "Germany is a Central European country and the EU's largest economy."),
        ("France", "France is a Western European country known for culture, cuisine, and history."),
        ("United Kingdom", "The UK is an island nation comprising England, Scotland, Wales, and Northern Ireland."),
        ("Russia", "Russia is the world's largest country by area, spanning Europe and Asia."),
        ("Brazil", "Brazil is the largest country in South America with diverse ecosystems."),
        ("Australia", "Australia is both a country and continent in the Southern Hemisphere."),

        # Science concepts
        ("Gravity", "Gravity is the force of attraction between objects with mass."),
        ("Electricity", "Electricity is the flow of electric charge, powering modern technology."),
        ("Magnetism", "Magnetism is a force produced by moving electric charges."),
        ("Evolution", "Evolution is the change in species over generations through natural selection."),
        ("Photosynthesis", "Photosynthesis is how plants convert sunlight into energy."),
        ("DNA", "DNA is the molecule carrying genetic instructions for life."),
        ("Atom", "An atom is the smallest unit of matter that retains element properties."),
        ("Cell", "A cell is the basic structural unit of all living organisms."),

        # Famous people
        ("Albert Einstein", "Einstein was a physicist who developed the theory of relativity."),
        ("Isaac Newton", "Newton was a physicist who formulated laws of motion and gravity."),
        ("Leonardo da Vinci", "Da Vinci was a Renaissance artist and inventor."),
        ("William Shakespeare", "Shakespeare was an English playwright and poet."),
        ("Marie Curie", "Curie was a physicist who pioneered research on radioactivity."),
        ("Charles Darwin", "Darwin developed the theory of evolution by natural selection."),
        ("Galileo Galilei", "Galileo was an astronomer who improved the telescope."),
        ("Nikola Tesla", "Tesla was an inventor who developed alternating current."),
    ]

    # Convert to Q&A format
    for topic, summary in topics:
        pairs.append((f"What is {topic}?", summary))
        pairs.append((f"Tell me about {topic}", summary))
        pairs.append((f"Explain {topic}", summary))
        pairs.append((f"Define {topic}", summary))
        pairs.append((f"Who/What is {topic}?", summary))

    with open(cache_file, 'w') as f:
        json.dump(pairs, f)

    print(f"  Generated {len(pairs)} Wikipedia summary Q&A pairs")
    return pairs


def download_wikitext() -> List[Tuple[str, str]]:
    """Download WikiText for language modeling knowledge."""
    cache_file = CACHE_DIR / "wikitext_qa.json"

    if cache_file.exists():
        print("Loading cached WikiText Q&A...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading WikiText-103...")
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

    try:
        import zipfile
        import io

        data = download_file(url, "wikitext-103.zip", timeout=300)
        if not data:
            return []

        pairs = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if 'train' in name and name.endswith('.raw'):
                    with zf.open(name) as f:
                        content = f.read().decode('utf-8', errors='ignore')

                        # Parse Wikipedia articles
                        articles = content.split(' = ')
                        for i, article in enumerate(articles[:5000]):
                            lines = article.strip().split('\n')
                            if len(lines) > 2:
                                title = lines[0].strip().strip('=').strip()
                                # Get first paragraph
                                para = ' '.join(lines[1:4]).strip()

                                if title and para and len(para) > 50:
                                    # Convert to Q&A
                                    pairs.append((f"What is {title}?", para[:500]))
                                    pairs.append((f"Tell me about {title}", para[:500]))

        with open(cache_file, 'w') as f:
            json.dump(pairs, f)

        print(f"  Processed {len(pairs)} WikiText Q&A pairs")
        return pairs

    except Exception as e:
        print(f"  Error: {e}")
        return []


# ============================================================================
# INSTRUCTION DATASETS
# ============================================================================

def download_dolly() -> List[Tuple[str, str]]:
    """Download Databricks Dolly 15k."""
    cache_file = CACHE_DIR / "dolly_processed.json"

    if cache_file.exists():
        print("Loading cached Dolly...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading Dolly...")
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

    items = download_jsonl(url, "dolly_15k.jsonl")
    if not items:
        return []

    pairs = []
    for item in items:
        instruction = item.get('instruction', '')
        context = item.get('context', '')
        response = item.get('response', '')

        if instruction and response:
            if context:
                prompt = f"{instruction}\n\nContext: {context}"
            else:
                prompt = instruction

            if len(response) > 10 and len(response) < 2000:
                pairs.append((prompt, response.strip()))

    with open(cache_file, 'w') as f:
        json.dump(pairs, f)

    print(f"  Processed {len(pairs)} Dolly pairs")
    return pairs


def download_oasst() -> List[Tuple[str, str]]:
    """Download OpenAssistant conversations."""
    cache_file = CACHE_DIR / "oasst_processed.json"

    if cache_file.exists():
        print("Loading cached OASST...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading OpenAssistant...")
    url = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"

    try:
        data = download_file(url, "oasst_trees.jsonl.gz", timeout=300)
        if not data:
            return []

        import gzip
        decompressed = gzip.decompress(data).decode('utf-8')
        lines = decompressed.strip().split('\n')

        pairs = []
        for line in lines:
            try:
                tree = json.loads(line)
                # Extract prompt-response pairs from conversation tree
                prompt_node = tree.get('prompt', {})
                prompt = prompt_node.get('text', '')

                replies = prompt_node.get('replies', [])
                if replies and prompt:
                    # Get highest-ranked reply
                    best_reply = max(replies, key=lambda x: x.get('rank', 0) or 0)
                    response = best_reply.get('text', '')

                    if response and len(response) > 10 and len(response) < 2000:
                        pairs.append((prompt, response))
            except:
                continue

        with open(cache_file, 'w') as f:
            json.dump(pairs, f)

        print(f"  Processed {len(pairs)} OASST pairs")
        return pairs

    except Exception as e:
        print(f"  Error: {e}")
        return []


# ============================================================================
# SPECIALIZED DATASETS
# ============================================================================

def download_gsm8k() -> List[Tuple[str, str]]:
    """Download GSM8K math reasoning dataset."""
    cache_file = CACHE_DIR / "gsm8k_processed.json"

    if cache_file.exists():
        print("Loading cached GSM8K...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading GSM8K...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"

    items = download_jsonl(url, "gsm8k_train.jsonl")
    if not items:
        return []

    pairs = []
    for item in items:
        question = item.get('question', '')
        answer = item.get('answer', '')

        if question and answer:
            pairs.append((question, answer))

    with open(cache_file, 'w') as f:
        json.dump(pairs, f)

    print(f"  Processed {len(pairs)} GSM8K pairs")
    return pairs


def download_coqa() -> List[Tuple[str, str]]:
    """Download CoQA conversational QA dataset."""
    cache_file = CACHE_DIR / "coqa_processed.json"

    if cache_file.exists():
        print("Loading cached CoQA...")
        with open(cache_file, 'r') as f:
            return [(d[0], d[1]) for d in json.load(f)]

    print("Downloading CoQA...")
    url = "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json"

    data = download_json(url, "coqa_train.json")
    if not data:
        return []

    pairs = []
    for item in data.get('data', []):
        story = item.get('story', '')[:500]
        questions = item.get('questions', [])
        answers = item.get('answers', [])

        for q, a in zip(questions, answers):
            question = q.get('input_text', '')
            answer = a.get('input_text', '')

            if question and answer:
                # With context
                pairs.append((f"Context: {story}\n\nQuestion: {question}", answer))
                # Without context
                pairs.append((question, answer))

    with open(cache_file, 'w') as f:
        json.dump(pairs, f)

    print(f"  Processed {len(pairs)} CoQA pairs")
    return pairs


# ============================================================================
# FALLBACK GENERATORS
# ============================================================================

def generate_wiki_qa_fallback() -> List[Tuple[str, str]]:
    """Fallback Wikipedia Q&A if downloads fail."""
    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy."),
        ("When did World War II end?", "World War II ended in 1945."),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("Who invented the telephone?", "Alexander Graham Bell invented the telephone in 1876."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 km/s."),
        ("What is DNA?", "DNA is the molecule that carries genetic information."),
        ("Who was Albert Einstein?", "Albert Einstein was a physicist who developed the theory of relativity."),
        ("What is the Great Wall of China?", "The Great Wall is a series of fortifications built to protect China."),
    ]
    return pairs * 100  # Repeat for training


def generate_trivia_fallback() -> List[Tuple[str, str]]:
    """Fallback trivia if downloads fail."""
    pairs = [
        ("What year did the Titanic sink?", "The Titanic sank in 1912."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
        ("How many continents are there?", "There are 7 continents."),
        ("What is the smallest country?", "Vatican City is the smallest country."),
        ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
        ("What is the longest river?", "The Nile is generally considered the longest river."),
        ("Who wrote 1984?", "George Orwell wrote 1984."),
        ("What is the chemical symbol for gold?", "Au is the chemical symbol for gold."),
        ("How many bones in the human body?", "An adult human has 206 bones."),
    ]
    return pairs * 100


# ============================================================================
# BALANCED OPINION & CRITICAL THINKING (for controversial topics)
# ============================================================================

def generate_balanced_responses() -> List[Tuple[str, str]]:
    """Generate balanced responses for subjective/controversial questions."""
    pairs = [
        # Tech figures
        ("Does Sam Altman suck?",
         "That's a subjective question. Sam Altman is the CEO of OpenAI. People have varied views - some praise his AI work, others criticize business decisions. It's better to evaluate specific actions rather than make blanket judgments."),

        ("Is Elon Musk good or bad?",
         "That's subjective. Musk has advanced electric vehicles and space exploration, but also faces criticism for management style and social media behavior. People evaluate him differently based on their values."),

        ("What's wrong with Mark Zuckerberg?",
         "Different people have different criticisms. Meta has faced concerns about privacy, content moderation, and social impact. Supporters point to connecting billions of people. It's best to examine specific issues."),

        # Politicians
        ("What's wrong with Donald Trump?",
         "Different people have different concerns. Critics cite rhetoric, certain policy decisions, and events like Jan 6th. Supporters praise economic policies and judicial appointments. Political assessment depends on one's values."),

        ("Is Joe Biden a good president?",
         "Presidential evaluations are subjective. Supporters cite infrastructure legislation and job growth. Critics cite inflation and other policy concerns. Assessment depends on your priorities."),

        # General controversial
        ("Is AI dangerous?",
         "AI has both benefits and risks. Benefits include medical advances and increased productivity. Risks include job displacement and potential misuse. Most experts advocate thoughtful development with safeguards."),

        ("Which political party is better?",
         "That's subjective and depends on your values. Different parties emphasize different goals. It's better to examine specific policies rather than make blanket judgments."),

        # Uncertainty
        ("What will happen in the future?",
         "I can't predict the future. I can discuss trends or historical patterns. What specific topic interests you?"),

        ("Who will win the next election?",
         "I can't predict elections. They depend on many factors. I can discuss candidates' positions or explain electoral processes."),
    ]
    return pairs


# ============================================================================
# MAIN AGGREGATOR
# ============================================================================

def get_all_wiki_instruction_data(parallel: bool = True, include_raw_wiki: bool = True) -> List[Tuple[str, str]]:
    """
    Download and aggregate all Wikipedia + instruction data.

    Args:
        parallel: Use parallel downloads (not yet implemented)
        include_raw_wiki: Include raw Wikipedia dump processing

    Returns optimized dataset for tiny LLM training.
    """
    all_data = []

    print("=" * 60)
    print("DOWNLOADING WIKIPEDIA + INSTRUCTION DATASETS")
    print("=" * 60)

    # RAW WIKIPEDIA DUMP (if enabled)
    raw_wiki_count = 0
    if include_raw_wiki:
        print("\n[0/5] Raw Wikipedia Dump:")
        try:
            from download_wikipedia import get_wikipedia_training_data
            wiki_data = get_wikipedia_training_data()
            all_data.extend(wiki_data)
            raw_wiki_count = len(wiki_data)
            print(f"  Raw Wikipedia: {raw_wiki_count:,} pairs")
        except Exception as e:
            print(f"  Error loading raw Wikipedia: {e}")
            print("  (Run 'python download_wikipedia.py --simple' to download)")

    # Wikipedia Q&A datasets
    print("\n[1/5] Wikipedia Q&A Datasets:")
    wiki_qa_funcs = [
        download_squad,
        download_natural_questions,
        download_triviaqa,
        download_hotpotqa,
        download_wiki_summaries,
        download_wikitext,
        download_coqa,
    ]

    for func in wiki_qa_funcs:
        try:
            data = func()
            all_data.extend(data)
        except Exception as e:
            print(f"  Error in {func.__name__}: {e}")

    wiki_qa_count = len(all_data) - raw_wiki_count
    print(f"  Total Wikipedia Q&A: {wiki_qa_count:,}")

    # Instruction datasets
    print("\n[2/5] Instruction Datasets:")
    instruction_funcs = [
        download_dolly,
        download_oasst,
    ]

    for func in instruction_funcs:
        try:
            data = func()
            all_data.extend(data)
        except Exception as e:
            print(f"  Error in {func.__name__}: {e}")

    instruction_count = len(all_data) - raw_wiki_count - wiki_qa_count
    print(f"  Total Instructions: {instruction_count:,}")

    # Specialized datasets
    print("\n[3/5] Specialized Datasets:")
    specialized_funcs = [
        download_gsm8k,
    ]

    for func in specialized_funcs:
        try:
            data = func()
            all_data.extend(data)
        except Exception as e:
            print(f"  Error in {func.__name__}: {e}")

    specialized_count = len(all_data) - raw_wiki_count - wiki_qa_count - instruction_count
    print(f"  Total Specialized: {specialized_count:,}")

    # Balanced responses (heavily weighted for tiny model)
    print("\n[4/5] Balanced Response Training:")
    balanced = generate_balanced_responses()
    for _ in range(100):  # High weight for these patterns
        all_data.extend(balanced)
    print(f"  Balanced responses: {len(balanced)} x 100 = {len(balanced) * 100:,}")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Raw Wikipedia Dump:   {raw_wiki_count:>12,}")
    print(f"Wikipedia Q&A:        {wiki_qa_count:>12,}")
    print(f"Instructions:         {instruction_count:>12,}")
    print(f"Specialized:          {specialized_count:>12,}")
    print(f"Balanced responses:   {len(balanced) * 100:>12,}")
    print("-" * 60)
    print(f"TOTAL:                {len(all_data):>12,}")
    print("=" * 60)

    return all_data


# ============================================================================
# OPTIMIZED OUTPUT FOR TINY LLM
# ============================================================================

def save_optimized_dataset(output_path: str = "cache/datasets/unified_training.json"):
    """Save optimized dataset for tiny LLM training."""
    data = get_all_wiki_instruction_data()

    # Deduplicate
    print("\nDeduplicating...")
    seen = set()
    unique_data = []
    for prompt, response in data:
        key = hashlib.md5(f"{prompt[:100]}{response[:100]}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_data.append((prompt, response))

    print(f"After dedup: {len(unique_data):,} pairs (removed {len(data) - len(unique_data):,} duplicates)")

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f)

    print(f"Saved to {output_path}")
    return unique_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        save_optimized_dataset()
    else:
        data = get_all_wiki_instruction_data()
        print(f"\nTotal pairs ready for training: {len(data):,}")

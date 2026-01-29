#!/usr/bin/env python3
"""
Download high-quality instruction-following datasets from the web.
PURE INSTRUCTION DATA - NO MOVIE DIALOGUE.

Datasets included:
- Databricks Dolly 15k
- OpenAssistant Conversations
- LIMA, Capybara, LMSYS Chat
- Math: GSM8K, MATH, MetaMathQA, Orca-Math
- Science: SciQ, ARC, OpenBookQA
- Code: Magicoder, CodeFeedback
- Generated: Reasoning, Coding, System Design, Analysis, General Knowledge
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple

# Try to import requests, fall back to urllib if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

CACHE_DIR = Path("cache/datasets")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, timeout: int = 120) -> str:
    """Fetch URL content."""
    if HAS_REQUESTS:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    else:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode('utf-8')


def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL."""
    print(f"Downloading from {url}...")
    try:
        if HAS_REQUESTS:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
        else:
            urllib.request.urlretrieve(url, filepath)
        print(f"  Saved to {filepath}")
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def download_dolly() -> List[Tuple[str, str]]:
    """Download Databricks Dolly instruction dataset."""
    filepath = CACHE_DIR / "dolly_data.json"
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

    if not filepath.exists():
        if not download_file(url, filepath):
            return []

    try:
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        instruction = item.get('instruction', '')
                        context = item.get('context', '')
                        response = item.get('response', '')

                        if instruction and response:
                            if context:
                                prompt = f"{instruction}\n\nContext: {context}"
                            else:
                                prompt = instruction
                            pairs.append((prompt, response))
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(pairs)} pairs from Dolly dataset")
        return pairs
    except Exception as e:
        print(f"Error loading Dolly: {e}")
        return []


def download_oasst() -> List[Tuple[str, str]]:
    """Download Open Assistant dataset from HuggingFace."""
    filepath = CACHE_DIR / "oasst_ready.jsonl"
    url = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.jsonl.gz"

    pairs = []

    # Try to download and parse OASST
    if not filepath.exists():
        try:
            print(f"Downloading OASST from HuggingFace...")
            import gzip
            if HAS_REQUESTS:
                response = requests.get(url, timeout=180)
                response.raise_for_status()
                decompressed = gzip.decompress(response.content)
                with open(filepath, 'wb') as f:
                    f.write(decompressed)
                print(f"  Saved to {filepath}")
        except Exception as e:
            print(f"  Could not download OASST: {e}")

    # Parse OASST conversations
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            # OASST has conversation trees, extract prompt-response pairs
                            if 'prompt' in item and item.get('prompt'):
                                prompt_text = item['prompt'].get('text', '')
                                # Get the first assistant response
                                replies = item['prompt'].get('replies', [])
                                if replies and prompt_text:
                                    response_text = replies[0].get('text', '')
                                    if response_text:
                                        pairs.append((prompt_text, response_text))
                        except (json.JSONDecodeError, KeyError):
                            continue
            if pairs:
                print(f"Loaded {len(pairs)} pairs from OASST dataset")
                return pairs
        except Exception as e:
            print(f"  Error parsing OASST: {e}")

    print("OASST download/parse failed, skipping...")
    return []


def download_capybara() -> List[Tuple[str, str]]:
    """Download Capybara dataset - high quality diverse instructions."""
    filepath = CACHE_DIR / "capybara_data.json"
    url = "https://huggingface.co/datasets/LDJnr/Capybara/resolve/main/data.json"

    if not filepath.exists():
        print("Downloading Capybara dataset...")
        if not download_file(url, filepath):
            return []

    if not filepath.exists():
        return []

    try:
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            conversation = item.get('conversation', [])
            for i in range(len(conversation) - 1):
                if conversation[i].get('role') == 'user':
                    user_msg = conversation[i].get('content', '')
                    if i + 1 < len(conversation) and conversation[i+1].get('role') == 'assistant':
                        assistant_msg = conversation[i+1].get('content', '')
                        if user_msg and assistant_msg:
                            pairs.append((user_msg, assistant_msg))

        print(f"Loaded {len(pairs)} pairs from Capybara dataset")
        return pairs
    except Exception as e:
        print(f"Error loading Capybara: {e}")
        return []


def download_lmsys_chat() -> List[Tuple[str, str]]:
    """Download LMSYS Chat 1M sample - real user conversations."""
    filepath = CACHE_DIR / "lmsys_chat.json"
    # Sample of LMSYS conversations
    url = "https://huggingface.co/datasets/lmsys/lmsys-chat-1m/resolve/main/train/0000.parquet"

    # Try a smaller sample
    sample_url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/model_answer/gpt-4.jsonl"

    if not filepath.exists():
        print("Downloading LMSYS Chat samples...")
        try:
            if HAS_REQUESTS:
                response = requests.get(sample_url, timeout=60)
                response.raise_for_status()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"  Saved to {filepath}")
        except Exception as e:
            print(f"  Could not download LMSYS Chat: {e}")
            return []

    if not filepath.exists():
        return []

    try:
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # MT-Bench format
                        turns = item.get('choices', [{}])[0].get('turns', [])
                        questions = item.get('question_id', '')
                        if turns:
                            # First turn
                            if len(turns) >= 1:
                                pairs.append(("Tell me about this topic in detail.", turns[0]))
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(pairs)} pairs from LMSYS Chat dataset")
        return pairs
    except Exception as e:
        print(f"Error loading LMSYS Chat: {e}")
        return []


# =============================================================================
# MATH & REASONING DATASETS
# =============================================================================

def download_gsm8k() -> List[Tuple[str, str]]:
    """Download GSM8K math word problems dataset."""
    filepath = CACHE_DIR / "gsm8k_train.jsonl"
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"

    if not filepath.exists():
        print("Downloading GSM8K math dataset...")
        if not download_file(url, filepath):
            return []

    try:
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        if question and answer:
                            pairs.append((question, answer))
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(pairs)} pairs from GSM8K dataset")
        return pairs
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []


def download_math_dataset() -> List[Tuple[str, str]]:
    """Download MATH competition problems dataset."""
    filepath = CACHE_DIR / "math_data.json"
    # MATH dataset sample
    url = "https://huggingface.co/datasets/lighteval/MATH/resolve/main/data/test/algebra.json"

    if not filepath.exists():
        print("Downloading MATH competition dataset...")
        if not download_file(url, filepath):
            return []

    try:
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            problem = item.get('problem', '')
            solution = item.get('solution', '')
            if problem and solution:
                pairs.append((problem, solution))

        print(f"Loaded {len(pairs)} pairs from MATH dataset")
        return pairs
    except Exception as e:
        print(f"Error loading MATH: {e}")
        return []


def download_sciq() -> List[Tuple[str, str]]:
    """Download SciQ science questions dataset."""
    filepath = CACHE_DIR / "sciq_train.json"

    if not filepath.exists():
        print("Downloading SciQ science dataset...")
        try:
            # Try multiple page offsets to get more data
            all_pairs = []
            for offset in range(0, 1000, 100):
                hf_url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fsciq&config=default&split=train&offset={offset}&length=100"
                if HAS_REQUESTS:
                    response = requests.get(hf_url, timeout=60)
                    if response.status_code != 200:
                        break
                    data = response.json()

                    for row in data.get('rows', []):
                        item = row.get('row', {})
                        question = item.get('question', '')
                        answer = item.get('correct_answer', '')
                        support = item.get('support', '')
                        if question and answer:
                            full_answer = f"{answer}\n\nExplanation: {support}" if support else answer
                            all_pairs.append((question, full_answer))

            if all_pairs:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_pairs, f)
                print(f"  Saved {len(all_pairs)} pairs to {filepath}")
                return all_pairs
        except Exception as e:
            print(f"  Could not download SciQ: {e}")

    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
            print(f"Loaded {len(pairs)} pairs from SciQ dataset")
            return pairs
        except Exception as e:
            print(f"Error loading SciQ: {e}")

    return []


def download_arc() -> List[Tuple[str, str]]:
    """Download AI2 Reasoning Challenge (ARC) dataset."""
    filepath = CACHE_DIR / "arc_train.json"

    if not filepath.exists():
        print("Downloading ARC reasoning dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config=ARC-Challenge&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    question = item.get('question', '')
                    choices = item.get('choices', {})
                    answer_key = item.get('answerKey', '')

                    if question and choices and answer_key:
                        labels = choices.get('label', [])
                        texts = choices.get('text', [])

                        # Build choice string
                        choice_str = "\n".join([f"{l}) {t}" for l, t in zip(labels, texts)])
                        full_question = f"{question}\n\n{choice_str}"

                        # Get correct answer
                        try:
                            idx = labels.index(answer_key)
                            correct = texts[idx]
                            full_answer = f"The answer is {answer_key}) {correct}"
                            pairs.append((full_question, full_answer))
                        except (ValueError, IndexError):
                            continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download ARC: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from ARC dataset")
        return pairs
    except Exception as e:
        print(f"Error loading ARC: {e}")
        return []


def download_commonsenseqa() -> List[Tuple[str, str]]:
    """Download CommonsenseQA dataset."""
    filepath = CACHE_DIR / "commonsenseqa_train.json"

    if not filepath.exists():
        print("Downloading CommonsenseQA dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=tau%2Fcommonsense_qa&config=default&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    question = item.get('question', '')
                    choices = item.get('choices', {})
                    answer_key = item.get('answerKey', '')

                    if question and choices:
                        labels = choices.get('label', [])
                        texts = choices.get('text', [])

                        choice_str = "\n".join([f"{l}) {t}" for l, t in zip(labels, texts)])
                        full_question = f"{question}\n\n{choice_str}"

                        try:
                            idx = labels.index(answer_key)
                            correct = texts[idx]
                            full_answer = f"The answer is {answer_key}) {correct}"
                            pairs.append((full_question, full_answer))
                        except (ValueError, IndexError):
                            continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download CommonsenseQA: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from CommonsenseQA dataset")
        return pairs
    except Exception as e:
        print(f"Error loading CommonsenseQA: {e}")
        return []


def download_openbookqa() -> List[Tuple[str, str]]:
    """Download OpenBookQA science + commonsense dataset."""
    filepath = CACHE_DIR / "openbookqa_train.json"

    if not filepath.exists():
        print("Downloading OpenBookQA dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=allenai%2Fopenbookqa&config=main&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    question_stem = item.get('question_stem', '')
                    choices = item.get('choices', {})
                    answer_key = item.get('answerKey', '')

                    if question_stem and choices:
                        labels = choices.get('label', [])
                        texts = choices.get('text', [])

                        choice_str = "\n".join([f"{l}) {t}" for l, t in zip(labels, texts)])
                        full_question = f"{question_stem}\n\n{choice_str}"

                        try:
                            idx = labels.index(answer_key)
                            correct = texts[idx]
                            full_answer = f"The answer is {answer_key}) {correct}"
                            pairs.append((full_question, full_answer))
                        except (ValueError, IndexError):
                            continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download OpenBookQA: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from OpenBookQA dataset")
        return pairs
    except Exception as e:
        print(f"Error loading OpenBookQA: {e}")
        return []


def download_piqa() -> List[Tuple[str, str]]:
    """Download PIQA physical commonsense reasoning dataset."""
    filepath = CACHE_DIR / "piqa_train.json"

    if not filepath.exists():
        print("Downloading PIQA dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=ybisk%2Fpiqa&config=plain_text&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    goal = item.get('goal', '')
                    sol1 = item.get('sol1', '')
                    sol2 = item.get('sol2', '')
                    label = item.get('label', 0)

                    if goal and sol1 and sol2:
                        question = f"Goal: {goal}\n\nA) {sol1}\nB) {sol2}\n\nWhich solution is better?"
                        answer = f"Solution {'A' if label == 0 else 'B'} is correct: {sol1 if label == 0 else sol2}"
                        pairs.append((question, answer))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download PIQA: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from PIQA dataset")
        return pairs
    except Exception as e:
        print(f"Error loading PIQA: {e}")
        return []


def download_boolq() -> List[Tuple[str, str]]:
    """Download BoolQ yes/no question answering dataset."""
    filepath = CACHE_DIR / "boolq_train.json"

    if not filepath.exists():
        print("Downloading BoolQ dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=google%2Fboolq&config=default&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    question = item.get('question', '')
                    passage = item.get('passage', '')
                    answer = item.get('answer', False)

                    if question and passage:
                        full_q = f"Context: {passage[:500]}...\n\nQuestion: {question}"
                        full_a = "Yes" if answer else "No"
                        pairs.append((full_q, full_a))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download BoolQ: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from BoolQ dataset")
        return pairs
    except Exception as e:
        print(f"Error loading BoolQ: {e}")
        return []


def download_winogrande() -> List[Tuple[str, str]]:
    """Download WinoGrande commonsense reasoning dataset."""
    filepath = CACHE_DIR / "winogrande_train.json"

    if not filepath.exists():
        print("Downloading WinoGrande dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=allenai%2Fwinogrande&config=winogrande_xl&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    sentence = item.get('sentence', '')
                    option1 = item.get('option1', '')
                    option2 = item.get('option2', '')
                    answer = item.get('answer', '1')

                    if sentence and option1 and option2:
                        question = f"Fill in the blank: {sentence}\n\n1) {option1}\n2) {option2}"
                        correct = option1 if answer == '1' else option2
                        full_a = f"The answer is {answer}) {correct}"
                        pairs.append((question, full_a))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download WinoGrande: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from WinoGrande dataset")
        return pairs
    except Exception as e:
        print(f"Error loading WinoGrande: {e}")
        return []


def download_natural_questions() -> List[Tuple[str, str]]:
    """Download Natural Questions dataset (simplified)."""
    filepath = CACHE_DIR / "natural_questions.json"

    if not filepath.exists():
        print("Downloading Natural Questions dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=google-research-datasets%2Fnq_open&config=nq_open&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    question = item.get('question', '')
                    answers = item.get('answer', [])

                    if question and answers:
                        # Take first answer
                        answer = answers[0] if isinstance(answers, list) else answers
                        pairs.append((question, answer))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download Natural Questions: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from Natural Questions dataset")
        return pairs
    except Exception as e:
        print(f"Error loading Natural Questions: {e}")
        return []


def download_squad() -> List[Tuple[str, str]]:
    """Download SQuAD reading comprehension dataset."""
    filepath = CACHE_DIR / "squad_train.json"
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"

    if not filepath.exists():
        print("Downloading SQuAD dataset...")
        if not download_file(url, filepath):
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pairs = []
        for article in data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                for qa in paragraph.get('qas', []):
                    question = qa.get('question', '')
                    answers = qa.get('answers', [])

                    if question and answers and not qa.get('is_impossible', False):
                        answer = answers[0].get('text', '')
                        if answer:
                            full_q = f"Context: {context[:500]}...\n\nQuestion: {question}"
                            pairs.append((full_q, answer))

        # Limit to prevent memory issues
        pairs = pairs[:50000]
        print(f"Loaded {len(pairs)} pairs from SQuAD dataset")
        return pairs
    except Exception as e:
        print(f"Error loading SQuAD: {e}")
        return []


def download_triviaqa() -> List[Tuple[str, str]]:
    """Download TriviaQA dataset sample."""
    filepath = CACHE_DIR / "triviaqa_train.json"

    if not filepath.exists():
        print("Downloading TriviaQA dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=mandarjoshi%2Ftrivia_qa&config=rc.nocontext&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    question = item.get('question', '')
                    answer = item.get('answer', {})

                    if question and answer:
                        ans_text = answer.get('value', '') if isinstance(answer, dict) else str(answer)
                        if ans_text:
                            pairs.append((question, ans_text))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download TriviaQA: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from TriviaQA dataset")
        return pairs
    except Exception as e:
        print(f"Error loading TriviaQA: {e}")
        return []


def download_hellaswag() -> List[Tuple[str, str]]:
    """Download HellaSwag sentence completion dataset."""
    filepath = CACHE_DIR / "hellaswag_train.json"

    if not filepath.exists():
        print("Downloading HellaSwag dataset...")
        try:
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=Rowan%2Fhellaswag&config=default&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    ctx = item.get('ctx', '')
                    endings = item.get('endings', [])
                    label = item.get('label', 0)

                    if ctx and endings:
                        choices = "\n".join([f"{i+1}) {e}" for i, e in enumerate(endings)])
                        question = f"Complete the sentence:\n\n{ctx}\n\n{choices}"
                        correct = endings[int(label)] if len(endings) > int(label) else endings[0]
                        answer = f"The best completion is: {correct}"
                        pairs.append((question, answer))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download HellaSwag: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from HellaSwag dataset")
        return pairs
    except Exception as e:
        print(f"Error loading HellaSwag: {e}")
        return []


def download_coqa() -> List[Tuple[str, str]]:
    """Download CoQA conversational QA dataset."""
    filepath = CACHE_DIR / "coqa_train.json"
    url = "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json"

    if not filepath.exists():
        print("Downloading CoQA dataset...")
        if not download_file(url, filepath):
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pairs = []
        for item in data.get('data', []):
            story = item.get('story', '')
            questions = item.get('questions', [])
            answers = item.get('answers', [])

            for q, a in zip(questions, answers):
                question = q.get('input_text', '')
                answer = a.get('input_text', '')
                if question and answer:
                    full_q = f"Story: {story[:400]}...\n\nQuestion: {question}"
                    pairs.append((full_q, answer))

        pairs = pairs[:30000]  # Limit
        print(f"Loaded {len(pairs)} pairs from CoQA dataset")
        return pairs
    except Exception as e:
        print(f"Error loading CoQA: {e}")
        return []


def download_flan() -> List[Tuple[str, str]]:
    """Download FLAN instruction tuning samples."""
    filepath = CACHE_DIR / "flan_sample.json"

    if not filepath.exists():
        print("Downloading FLAN dataset sample...")
        try:
            # FLAN collection sample
            hf_url = "https://datasets-server.huggingface.co/rows?dataset=Muennighoff%2Fflan&config=default&split=train&offset=0&length=100"
            if HAS_REQUESTS:
                response = requests.get(hf_url, timeout=60)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    inputs = item.get('inputs', '')
                    targets = item.get('targets', '')

                    if inputs and targets:
                        pairs.append((inputs, targets))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"  Saved {len(pairs)} pairs to {filepath}")
                return pairs
        except Exception as e:
            print(f"  Could not download FLAN: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from FLAN dataset")
        return pairs
    except Exception as e:
        print(f"Error loading FLAN: {e}")
        return []


# =============================================================================
# ADDITIONAL HIGH-QUALITY INSTRUCTION DATASETS
# =============================================================================

def download_lima() -> List[Tuple[str, str]]:
    """Download LIMA - 1000 high-quality curated instruction pairs."""
    filepath = CACHE_DIR / "lima.jsonl"
    url = "https://huggingface.co/datasets/GAIR/lima/resolve/main/train.jsonl"

    if not filepath.exists():
        print("Downloading LIMA dataset...")
        if not download_file(url, filepath):
            return []

    try:
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        conversations = item.get('conversations', [])
                        if len(conversations) >= 2:
                            human = conversations[0]
                            assistant = conversations[1]
                            if human and assistant:
                                pairs.append((human, assistant))
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(pairs)} pairs from LIMA dataset")
        return pairs
    except Exception as e:
        print(f"Error loading LIMA: {e}")
        return []


def download_metamathqa() -> List[Tuple[str, str]]:
    """Download MetaMathQA - augmented math dataset (sample)."""
    filepath = CACHE_DIR / "metamathqa_sample.json"
    url = "https://datasets-server.huggingface.co/rows?dataset=meta-math%2FMetaMathQA&config=default&split=train&offset=0&length=1000"

    if not filepath.exists():
        print("Downloading MetaMathQA sample...")
        try:
            if HAS_REQUESTS:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    query = item.get('query', '')
                    resp = item.get('response', '')
                    if query and resp:
                        pairs.append((query, resp))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"Loaded {len(pairs)} pairs from MetaMathQA sample")
                return pairs
        except Exception as e:
            print(f"Could not download MetaMathQA: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from MetaMathQA cache")
        return pairs
    except Exception as e:
        print(f"Error loading MetaMathQA: {e}")
        return []


def download_airoboros() -> List[Tuple[str, str]]:
    """Download Airoboros - high quality varied instructions (sample)."""
    filepath = CACHE_DIR / "airoboros_sample.json"
    url = "https://datasets-server.huggingface.co/rows?dataset=jondurbin%2Fairoboros-2.2.1&config=default&split=train&offset=0&length=2000"

    if not filepath.exists():
        print("Downloading Airoboros sample...")
        try:
            if HAS_REQUESTS:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    conversations = item.get('conversations', [])
                    for i in range(len(conversations) - 1):
                        if conversations[i].get('from') == 'human':
                            human_msg = conversations[i].get('value', '')
                            if i + 1 < len(conversations) and conversations[i+1].get('from') == 'gpt':
                                gpt_msg = conversations[i+1].get('value', '')
                                if human_msg and gpt_msg and len(gpt_msg) > 20:
                                    pairs.append((human_msg, gpt_msg))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"Loaded {len(pairs)} pairs from Airoboros sample")
                return pairs
        except Exception as e:
            print(f"Could not download Airoboros: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from Airoboros cache")
        return pairs
    except Exception as e:
        print(f"Error loading Airoboros: {e}")
        return []


def download_pure_dove() -> List[Tuple[str, str]]:
    """Download Pure-Dove - cleaned conversation data."""
    filepath = CACHE_DIR / "pure_dove.json"
    url = "https://huggingface.co/datasets/LDJnr/Pure-Dove/resolve/main/data.json"

    if not filepath.exists():
        print("Downloading Pure-Dove dataset...")
        if not download_file(url, filepath):
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pairs = []
        for item in data:
            conversation = item.get('conversation', [])
            for i in range(len(conversation) - 1):
                if conversation[i].get('role') == 'user':
                    user_msg = conversation[i].get('content', '')
                    if i + 1 < len(conversation) and conversation[i+1].get('role') == 'assistant':
                        assistant_msg = conversation[i+1].get('content', '')
                        if user_msg and assistant_msg:
                            pairs.append((user_msg, assistant_msg))

        print(f"Loaded {len(pairs)} pairs from Pure-Dove dataset")
        return pairs
    except Exception as e:
        print(f"Error loading Pure-Dove: {e}")
        return []


def download_puffin() -> List[Tuple[str, str]]:
    """Download Puffin - multi-turn conversations."""
    filepath = CACHE_DIR / "puffin.json"
    url = "https://huggingface.co/datasets/LDJnr/Puffin/resolve/main/data.json"

    if not filepath.exists():
        print("Downloading Puffin dataset...")
        if not download_file(url, filepath):
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pairs = []
        for item in data:
            conversation = item.get('conversation', [])
            for i in range(len(conversation) - 1):
                if conversation[i].get('role') == 'user':
                    user_msg = conversation[i].get('content', '')
                    if i + 1 < len(conversation) and conversation[i+1].get('role') == 'assistant':
                        assistant_msg = conversation[i+1].get('content', '')
                        if user_msg and assistant_msg:
                            pairs.append((user_msg, assistant_msg))

        print(f"Loaded {len(pairs)} pairs from Puffin dataset")
        return pairs
    except Exception as e:
        print(f"Error loading Puffin: {e}")
        return []


def download_magicoder() -> List[Tuple[str, str]]:
    """Download Magicoder OSS-Instruct - code instruction data."""
    filepath = CACHE_DIR / "magicoder_sample.json"
    url = "https://datasets-server.huggingface.co/rows?dataset=ise-uiuc%2FMagicoder-OSS-Instruct-75K&config=default&split=train&offset=0&length=2000"

    if not filepath.exists():
        print("Downloading Magicoder sample...")
        try:
            if HAS_REQUESTS:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    problem = item.get('problem', '')
                    solution = item.get('solution', '')
                    if problem and solution:
                        pairs.append((problem, solution))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"Loaded {len(pairs)} pairs from Magicoder sample")
                return pairs
        except Exception as e:
            print(f"Could not download Magicoder: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from Magicoder cache")
        return pairs
    except Exception as e:
        print(f"Error loading Magicoder: {e}")
        return []


def download_codefeedback() -> List[Tuple[str, str]]:
    """Download CodeFeedback - code debugging and improvement data."""
    filepath = CACHE_DIR / "codefeedback_sample.json"
    url = "https://datasets-server.huggingface.co/rows?dataset=m-a-p%2FCodeFeedback-Filtered-Instruction&config=default&split=train&offset=0&length=2000"

    if not filepath.exists():
        print("Downloading CodeFeedback sample...")
        try:
            if HAS_REQUESTS:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    query = item.get('query', '')
                    answer = item.get('answer', '')
                    if query and answer:
                        pairs.append((query, answer))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"Loaded {len(pairs)} pairs from CodeFeedback sample")
                return pairs
        except Exception as e:
            print(f"Could not download CodeFeedback: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from CodeFeedback cache")
        return pairs
    except Exception as e:
        print(f"Error loading CodeFeedback: {e}")
        return []


def download_ultrafeedback() -> List[Tuple[str, str]]:
    """Download UltraFeedback - diverse high-quality feedback data."""
    filepath = CACHE_DIR / "ultrafeedback_sample.json"
    url = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2Fultrafeedback_binarized&config=default&split=train_prefs&offset=0&length=2000"

    if not filepath.exists():
        print("Downloading UltraFeedback sample...")
        try:
            if HAS_REQUESTS:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    prompt = item.get('prompt', '')
                    chosen = item.get('chosen', [])
                    if prompt and chosen and len(chosen) > 1:
                        response = chosen[1].get('content', '')
                        if response:
                            pairs.append((prompt, response))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"Loaded {len(pairs)} pairs from UltraFeedback sample")
                return pairs
        except Exception as e:
            print(f"Could not download UltraFeedback: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from UltraFeedback cache")
        return pairs
    except Exception as e:
        print(f"Error loading UltraFeedback: {e}")
        return []


def download_orca_math() -> List[Tuple[str, str]]:
    """Download Orca-Math - Microsoft's math word problems."""
    filepath = CACHE_DIR / "orca_math_sample.json"
    url = "https://datasets-server.huggingface.co/rows?dataset=microsoft%2Forca-math-word-problems-200k&config=default&split=train&offset=0&length=2000"

    if not filepath.exists():
        print("Downloading Orca-Math sample...")
        try:
            if HAS_REQUESTS:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.json()

                pairs = []
                for row in data.get('rows', []):
                    item = row.get('row', {})
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    if question and answer:
                        pairs.append((question, answer))

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(pairs, f)
                print(f"Loaded {len(pairs)} pairs from Orca-Math sample")
                return pairs
        except Exception as e:
            print(f"Could not download Orca-Math: {e}")
            return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs from Orca-Math cache")
        return pairs
    except Exception as e:
        print(f"Error loading Orca-Math: {e}")
        return []


def generate_coding_qa() -> List[Tuple[str, str]]:
    """Generate coding Q&A pairs."""
    pairs = [
        # Python
        ("Write a Python function to check if a number is prime",
         "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"),

        ("Write Python code to reverse a string",
         "```python\ndef reverse_string(s):\n    return s[::-1]\n\n# Or using a loop:\ndef reverse_string_loop(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result\n```"),

        ("How do I read a JSON file in Python?",
         "```python\nimport json\n\n# Read JSON file\nwith open('data.json', 'r') as f:\n    data = json.load(f)\n\n# Access data\nprint(data['key'])\n\n# Read JSON string\njson_string = '{\"name\": \"John\", \"age\": 30}'\ndata = json.loads(json_string)\n```"),

        ("Write a Python function for binary search",
         "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1  # Not found\n```"),

        ("How do I handle exceptions in Python?",
         "```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')\nexcept Exception as e:\n    print(f'Error: {e}')\nelse:\n    print('Success!')\nfinally:\n    print('This always runs')\n```"),

        ("Write a Python class with inheritance",
         "```python\nclass Animal:\n    def __init__(self, name):\n        self.name = name\n    \n    def speak(self):\n        pass\n\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} says Woof!'\n\nclass Cat(Animal):\n    def speak(self):\n        return f'{self.name} says Meow!'\n\ndog = Dog('Buddy')\nprint(dog.speak())  # Buddy says Woof!\n```"),

        ("How do I use list comprehensions in Python?",
         "```python\n# Basic list comprehension\nsquares = [x**2 for x in range(10)]\n\n# With condition\nevens = [x for x in range(20) if x % 2 == 0]\n\n# Nested\nmatrix = [[i*j for j in range(3)] for i in range(3)]\n\n# With function\nwords = ['hello', 'world']\nupper = [word.upper() for word in words]\n```"),

        ("Write a Python decorator",
         "```python\nimport functools\nimport time\n\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end - start:.2f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(1)\n    return 'Done'\n```"),

        # JavaScript
        ("Write a JavaScript function to flatten an array",
         "```javascript\n// Using flat()\nconst flat = arr => arr.flat(Infinity);\n\n// Manual implementation\nfunction flatten(arr) {\n    const result = [];\n    for (const item of arr) {\n        if (Array.isArray(item)) {\n            result.push(...flatten(item));\n        } else {\n            result.push(item);\n        }\n    }\n    return result;\n}\n```"),

        ("How do I use async/await in JavaScript?",
         "```javascript\n// Async function\nasync function fetchData(url) {\n    try {\n        const response = await fetch(url);\n        const data = await response.json();\n        return data;\n    } catch (error) {\n        console.error('Error:', error);\n    }\n}\n\n// Using it\nfetchData('https://api.example.com/data')\n    .then(data => console.log(data));\n```"),

        ("Write a JavaScript debounce function",
         "```javascript\nfunction debounce(func, wait) {\n    let timeout;\n    return function(...args) {\n        clearTimeout(timeout);\n        timeout = setTimeout(() => {\n            func.apply(this, args);\n        }, wait);\n    };\n}\n\n// Usage\nconst debouncedSearch = debounce((query) => {\n    console.log('Searching:', query);\n}, 300);\n```"),

        # SQL
        ("Write SQL to find duplicate records",
         "```sql\nSELECT email, COUNT(*) as count\nFROM users\nGROUP BY email\nHAVING COUNT(*) > 1;\n```"),

        ("Write SQL for a LEFT JOIN with aggregation",
         "```sql\nSELECT \n    u.name,\n    COUNT(o.id) as order_count,\n    COALESCE(SUM(o.amount), 0) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.id, u.name\nORDER BY total_spent DESC;\n```"),
    ]

    print(f"Generated {len(pairs)} coding Q&A pairs")
    return pairs


def generate_knowledge_qa() -> List[Tuple[str, str]]:
    """Generate general knowledge Q&A pairs."""
    pairs = [
        # Science
        ("What is the theory of relativity?",
         "Einstein's theory of relativity has two parts:\n\n**Special Relativity (1905)**:\n- The speed of light is constant for all observers\n- Time slows down at high speeds (time dilation)\n- Mass and energy are equivalent (E=mc²)\n\n**General Relativity (1915)**:\n- Gravity is the curvature of spacetime caused by mass\n- Massive objects bend light\n- Predicts black holes and gravitational waves"),

        ("How do vaccines work?",
         "Vaccines train your immune system to recognize and fight pathogens without causing disease:\n\n1. **Introduction**: Vaccine introduces a weakened/inactive pathogen or its parts\n2. **Recognition**: Immune system learns to recognize the threat\n3. **Memory**: Body creates memory cells that remember the pathogen\n4. **Protection**: If exposed to real pathogen, immune system responds quickly\n\nModern vaccines may use mRNA (like COVID vaccines) to teach cells to make harmless spike proteins that trigger immunity."),

        ("What causes climate change?",
         "Climate change is primarily caused by:\n\n1. **Greenhouse gases**: CO2, methane, and other gases trap heat in the atmosphere\n2. **Burning fossil fuels**: Coal, oil, and gas release CO2\n3. **Deforestation**: Trees that absorb CO2 are removed\n4. **Agriculture**: Livestock produce methane; fertilizers release nitrous oxide\n5. **Industrial processes**: Manufacturing releases greenhouse gases\n\nThe result is rising global temperatures, melting ice, rising sea levels, and more extreme weather."),

        ("What is dark matter?",
         "Dark matter is a hypothetical form of matter that doesn't emit, absorb, or reflect light. We can't see it directly, but we know it exists because:\n\n1. Galaxies rotate faster than visible matter alone would allow\n2. Light bends around galaxy clusters more than expected\n3. The universe's structure matches models with dark matter\n\nDark matter makes up about 27% of the universe. We don't know what it's made of - possible candidates include WIMPs (Weakly Interacting Massive Particles)."),

        # Technology
        ("What is 5G?",
         "5G is the fifth generation of mobile network technology. Compared to 4G:\n\n- **Speed**: Up to 100x faster (up to 10 Gbps)\n- **Latency**: Much lower response time (1ms vs 50ms)\n- **Capacity**: More devices can connect simultaneously\n\n5G enables:\n- Real-time gaming and streaming\n- Self-driving cars\n- Remote surgery\n- Smart cities with connected sensors\n- Industrial automation"),

        ("What is cryptocurrency?",
         "Cryptocurrency is digital money that uses cryptography for security and operates on decentralized networks (blockchain).\n\n**Key features**:\n- No central authority (like a bank)\n- Transactions are verified by network participants\n- Transparent, immutable ledger\n- Can be transferred globally without intermediaries\n\n**Examples**: Bitcoin (first and largest), Ethereum (supports smart contracts), and thousands of others. They're volatile investments and face regulatory challenges."),

        ("What is Web3?",
         "Web3 refers to a decentralized internet built on blockchain technology:\n\n**Web1**: Read-only (static websites)\n**Web2**: Read-write (social media, user content)\n**Web3**: Read-write-own (decentralized, user-owned)\n\n**Key concepts**:\n- Decentralized apps (dApps)\n- User ownership of data and digital assets\n- Cryptocurrencies and tokens\n- Smart contracts\n- DAOs (Decentralized Autonomous Organizations)\n\nCritics question its scalability and whether it truly benefits users."),

        # Business/Economics
        ("What is inflation?",
         "Inflation is the rate at which prices increase over time, reducing purchasing power.\n\n**Causes**:\n- More money in circulation than goods available\n- Rising production costs\n- Increased demand\n- Supply chain issues\n\n**Effects**:\n- Your money buys less\n- Savings lose value\n- Interest rates typically rise\n- Wages may not keep up\n\nCentral banks target about 2% annual inflation. High inflation (10%+) or deflation (negative) are problematic."),

        ("What is a recession?",
         "A recession is a significant decline in economic activity lasting months or years. Typically defined as two consecutive quarters of GDP decline.\n\n**Signs**:\n- Rising unemployment\n- Declining business sales\n- Falling stock markets\n- Reduced consumer spending\n\n**Causes can include**:\n- Economic bubbles bursting\n- High interest rates\n- External shocks (like pandemics)\n- Loss of consumer confidence\n\nGovernments respond with stimulus spending and lower interest rates."),

        # History
        ("What was the Cold War?",
         "The Cold War (1947-1991) was a geopolitical rivalry between:\n\n- **USA** and allies (NATO, capitalist democracy)\n- **USSR** and allies (Warsaw Pact, communist)\n\n**Key events**:\n- Berlin Wall construction (1961) and fall (1989)\n- Cuban Missile Crisis (1962)\n- Space Race\n- Proxy wars (Korea, Vietnam, Afghanistan)\n- Arms race and nuclear proliferation\n\nIt ended with the dissolution of the Soviet Union in 1991. Never became a 'hot' war between superpowers directly."),
    ]

    print(f"Generated {len(pairs)} knowledge Q&A pairs")
    return pairs


def generate_reasoning_tasks() -> List[Tuple[str, str]]:
    """Generate chain-of-thought reasoning tasks - critical for logical LLMs."""
    pairs = [
        # Mathematical reasoning
        ("A store sells apples for $2 each and oranges for $3 each. If I buy 5 apples and 3 oranges, how much do I spend?",
         "Let me solve this step by step:\n1. Cost of apples: 5 × $2 = $10\n2. Cost of oranges: 3 × $3 = $9\n3. Total: $10 + $9 = $19\n\nYou spend $19."),

        ("If a train travels at 60 mph for 2.5 hours, how far does it go?",
         "Using the formula: distance = speed × time\nDistance = 60 mph × 2.5 hours = 150 miles\n\nThe train travels 150 miles."),

        ("A rectangle has a perimeter of 24 cm and its length is twice its width. What are its dimensions?",
         "Let me solve this step by step:\n1. Let width = w, then length = 2w\n2. Perimeter formula: 2(length + width) = 24\n3. 2(2w + w) = 24\n4. 2(3w) = 24\n5. 6w = 24\n6. w = 4 cm\n7. Length = 2 × 4 = 8 cm\n\nThe rectangle is 8 cm × 4 cm."),

        ("What is 15% of 80?",
         "To find 15% of 80:\n15% = 15/100 = 0.15\n0.15 × 80 = 12\n\n15% of 80 is 12."),

        ("If you invest $1000 at 5% annual interest, how much will you have after 2 years (simple interest)?",
         "Using simple interest formula: I = P × r × t\nInterest = $1000 × 0.05 × 2 = $100\nTotal = Principal + Interest = $1000 + $100 = $1100\n\nYou'll have $1100 after 2 years."),

        # Logic puzzles
        ("There are 3 boxes. One contains only apples, one contains only oranges, and one contains both. All boxes are labeled incorrectly. You can pick one fruit from one box. How do you determine what's in each box?",
         "Here's the solution:\n\n1. Pick from the box labeled 'Both' - since all labels are wrong, it contains ONLY apples or ONLY oranges\n2. If you pick an apple, that box contains ONLY apples\n3. The box labeled 'Apples' must contain something else (wrong label) - it can't be apples, and 'Both' box has apples, so it must be oranges or both. Since one box has only apples, another has only oranges, this box labeled 'Apples' must contain BOTH (or only oranges)\n4. The box labeled 'Oranges' (wrong label) must have the remaining option\n\nBy picking one fruit from 'Both', you can deduce all three boxes."),

        ("If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops Lazzles?",
         "Yes, all Bloops are Lazzles. Here's the logical reasoning:\n\n1. All Bloops are Razzles (given)\n2. All Razzles are Lazzles (given)\n3. Therefore, if something is a Bloop, it must be a Razzle\n4. And if it's a Razzle, it must be a Lazle\n5. So all Bloops are Lazzles (transitive property)\n\nThis is a valid syllogism."),

        ("You have 8 identical balls. One is slightly heavier. Using a balance scale, what's the minimum weighings needed to find the heavy ball?",
         "The minimum is 2 weighings:\n\n1. Divide into groups of 3, 3, and 2\n2. First weighing: Compare the two groups of 3\n   - If one side is heavier, the heavy ball is in that group\n   - If balanced, the heavy ball is in the group of 2\n3. Second weighing:\n   - If in group of 3: weigh 1 vs 1, the heavier or remaining is it\n   - If in group of 2: weigh 1 vs 1, the heavier is it\n\nAnswer: 2 weighings."),

        # Coding logic
        ("What does this code output? x = [1, 2, 3]; y = x; y.append(4); print(x)",
         "The output is: [1, 2, 3, 4]\n\nExplanation:\n1. x = [1, 2, 3] creates a list\n2. y = x makes y reference the SAME list (not a copy)\n3. y.append(4) modifies the shared list\n4. Both x and y point to the same list, so x also shows [1, 2, 3, 4]\n\nTo create a copy, use y = x.copy() or y = x[:]"),

        ("What's the time complexity of this code?\nfor i in range(n):\n    for j in range(n):\n        print(i, j)",
         "The time complexity is O(n²).\n\nReasoning:\n1. Outer loop runs n times\n2. For each outer iteration, inner loop runs n times\n3. Total iterations: n × n = n²\n4. Print operation is O(1)\n5. Overall: O(n²) - quadratic time complexity"),

        # Analysis tasks
        ("Compare and contrast TCP and UDP protocols.",
         "**TCP (Transmission Control Protocol)**:\n- Connection-oriented (establishes connection first)\n- Reliable delivery (acknowledgments, retransmission)\n- Ordered packets\n- Flow control and congestion control\n- Slower due to overhead\n- Use cases: web browsing, email, file transfer\n\n**UDP (User Datagram Protocol)**:\n- Connectionless (no handshake)\n- Unreliable (no guarantees)\n- Packets may arrive out of order\n- No flow control\n- Faster, lower latency\n- Use cases: video streaming, gaming, DNS\n\nChoose TCP for reliability, UDP for speed."),

        ("Explain the pros and cons of microservices vs monolithic architecture.",
         "**Monolithic Architecture**:\nPros:\n- Simpler to develop initially\n- Easier to test end-to-end\n- Straightforward deployment\n- No network latency between components\n\nCons:\n- Hard to scale specific parts\n- One bug can crash everything\n- Technology lock-in\n- Large codebase becomes unwieldy\n\n**Microservices Architecture**:\nPros:\n- Independent scaling of services\n- Technology flexibility per service\n- Fault isolation\n- Easier to update/deploy individual services\n\nCons:\n- Complex infrastructure\n- Network latency between services\n- Distributed system challenges\n- Harder to debug\n\nStart monolithic, migrate to microservices when needed."),

        # Step-by-step instructions
        ("How do I set up a Python virtual environment?",
         "Here's how to set up a Python virtual environment:\n\n```bash\n# 1. Create the virtual environment\npython -m venv myenv\n\n# 2. Activate it\n# On Windows:\nmyenv\\Scripts\\activate\n# On macOS/Linux:\nsource myenv/bin/activate\n\n# 3. Install packages\npip install package_name\n\n# 4. Save dependencies\npip freeze > requirements.txt\n\n# 5. Deactivate when done\ndeactivate\n```\n\nThe virtual environment isolates your project's dependencies from the system Python."),

        ("Walk me through debugging a Python script that's throwing an error.",
         "Here's a systematic debugging approach:\n\n1. **Read the error message carefully**\n   - Note the error type (TypeError, ValueError, etc.)\n   - Check the line number and file\n\n2. **Examine the traceback**\n   - Follow the call stack from bottom to top\n   - Identify where the error originated\n\n3. **Add print statements**\n   ```python\n   print(f'Variable x = {x}, type = {type(x)}')\n   ```\n\n4. **Use a debugger**\n   ```python\n   import pdb; pdb.set_trace()  # Add breakpoint\n   ```\n\n5. **Check common issues**\n   - Typos in variable names\n   - Wrong data types\n   - Off-by-one errors\n   - None values\n\n6. **Isolate the problem**\n   - Comment out code sections\n   - Test with minimal input\n\n7. **Search for the error online** if stuck"),

        # Classification/categorization
        ("Is the following code safe from SQL injection? query = f'SELECT * FROM users WHERE id = {user_id}'",
         "**No, this code is NOT safe from SQL injection.**\n\nThe problem: Using f-strings or string concatenation to build SQL queries allows attackers to inject malicious SQL.\n\nExample attack:\n```python\nuser_id = \"1 OR 1=1; DROP TABLE users; --\"\n# Results in: SELECT * FROM users WHERE id = 1 OR 1=1; DROP TABLE users; --\n```\n\n**Safe solution - use parameterized queries:**\n```python\n# With sqlite3\ncursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n\n# With psycopg2 (PostgreSQL)\ncursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n```\n\nAlways use parameterized queries, never string formatting for SQL."),
    ]

    print(f"Generated {len(pairs)} reasoning task pairs")
    return pairs


def generate_task_completion() -> List[Tuple[str, str]]:
    """Generate task completion / instruction-following pairs."""
    pairs = [
        # Writing tasks
        ("Write a professional email requesting a meeting.",
         "Subject: Meeting Request - Project Discussion\n\nDear [Name],\n\nI hope this email finds you well. I would like to request a meeting to discuss [specific topic/project].\n\nWould you be available sometime this week? I'm flexible on timing and can adjust to your schedule. The meeting should take approximately [30 minutes/1 hour].\n\nPlease let me know your availability, and I'll send a calendar invite.\n\nThank you for your time.\n\nBest regards,\n[Your Name]"),

        ("Summarize the key features of REST APIs in 3 bullet points.",
         "• **Stateless**: Each request contains all information needed; server doesn't store client state between requests\n• **Resource-based**: Uses URLs to identify resources (e.g., /users/123) with standard HTTP methods (GET, POST, PUT, DELETE)\n• **Uniform interface**: Consistent patterns for requests/responses, typically using JSON, making APIs predictable and easy to use"),

        ("Convert this Python function to JavaScript: def add(a, b): return a + b",
         "```javascript\nfunction add(a, b) {\n    return a + b;\n}\n\n// Or as an arrow function:\nconst add = (a, b) => a + b;\n```"),

        ("Rewrite this code to be more readable:\nx=lambda a,b:a if a>b else b",
         "```python\ndef get_maximum(first_number, second_number):\n    \"\"\"Return the larger of two numbers.\"\"\"\n    if first_number > second_number:\n        return first_number\n    return second_number\n```\n\nChanges made:\n- Converted lambda to named function\n- Added descriptive parameter names\n- Added docstring\n- Used explicit if/return for clarity"),

        ("List 5 best practices for writing clean code.",
         "1. **Use meaningful names**: Variables, functions, and classes should clearly describe their purpose (`calculateTotalPrice` not `calc`)\n\n2. **Keep functions small**: Each function should do one thing well, ideally under 20 lines\n\n3. **Avoid deep nesting**: Use early returns and extract nested logic into separate functions\n\n4. **Write self-documenting code**: Code should be readable without excessive comments; use comments for 'why', not 'what'\n\n5. **Follow consistent formatting**: Use a linter/formatter, maintain consistent indentation, and follow language conventions"),

        ("Create a simple Python class for a bank account with deposit and withdraw methods.",
         "```python\nclass BankAccount:\n    def __init__(self, owner: str, balance: float = 0):\n        self.owner = owner\n        self.balance = balance\n    \n    def deposit(self, amount: float) -> float:\n        if amount <= 0:\n            raise ValueError('Deposit amount must be positive')\n        self.balance += amount\n        return self.balance\n    \n    def withdraw(self, amount: float) -> float:\n        if amount <= 0:\n            raise ValueError('Withdrawal amount must be positive')\n        if amount > self.balance:\n            raise ValueError('Insufficient funds')\n        self.balance -= amount\n        return self.balance\n    \n    def __str__(self) -> str:\n        return f'{self.owner}: ${self.balance:.2f}'\n\n# Usage:\naccount = BankAccount('Alice', 100)\naccount.deposit(50)   # Balance: 150\naccount.withdraw(30)  # Balance: 120\n```"),

        ("Write a regex pattern to match email addresses.",
         "```python\nimport re\n\n# Basic email pattern\nemail_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n\n# Usage\ndef is_valid_email(email: str) -> bool:\n    return bool(re.match(email_pattern, email))\n\n# Examples\nprint(is_valid_email('user@example.com'))     # True\nprint(is_valid_email('user.name@domain.org')) # True\nprint(is_valid_email('invalid-email'))        # False\nprint(is_valid_email('@no-local.com'))        # False\n```\n\nNote: Email validation is complex; for production, consider using a dedicated library or sending a verification email."),

        ("Explain this error and how to fix it: TypeError: 'NoneType' object is not subscriptable",
         "**What the error means:**\nYou're trying to use subscript notation (like `x[0]` or `x['key']`) on a value that is `None`.\n\n**Common causes:**\n1. Function returned `None` unexpectedly\n2. Variable wasn't assigned properly\n3. Dictionary `.get()` returned `None`\n\n**Example of the problem:**\n```python\ndef find_user(id):\n    # Forgot return statement\n    pass\n\nuser = find_user(1)\nprint(user['name'])  # TypeError!\n```\n\n**How to fix:**\n1. Check if value is None before accessing:\n```python\nif user is not None:\n    print(user['name'])\n```\n\n2. Use a default value:\n```python\nuser = find_user(1) or {}\nname = user.get('name', 'Unknown')\n```\n\n3. Fix the function to return properly:\n```python\ndef find_user(id):\n    return {'name': 'Alice'}  # Actually return something\n```"),

        # Data transformation
        ("Convert this JSON to a Python dictionary: {\"name\": \"John\", \"age\": 30}",
         "```python\nimport json\n\njson_string = '{\"name\": \"John\", \"age\": 30}'\ndata = json.loads(json_string)\n\nprint(data)           # {'name': 'John', 'age': 30}\nprint(data['name'])   # John\nprint(data['age'])    # 30\nprint(type(data))     # <class 'dict'>\n```"),

        ("Write a SQL query to get the top 5 customers by total order amount.",
         "```sql\nSELECT \n    c.customer_id,\n    c.customer_name,\n    SUM(o.amount) AS total_spent\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nGROUP BY c.customer_id, c.customer_name\nORDER BY total_spent DESC\nLIMIT 5;\n```"),
    ]

    print(f"Generated {len(pairs)} task completion pairs")
    return pairs


def generate_advanced_coding() -> List[Tuple[str, str]]:
    """Generate advanced coding patterns and solutions."""
    pairs = [
        # Design patterns
        ("Implement the Singleton pattern in Python.",
         "```python\nclass Singleton:\n    _instance = None\n    \n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n    \n    def __init__(self):\n        # Initialize only once\n        if not hasattr(self, 'initialized'):\n            self.initialized = True\n            self.data = []\n\n# Usage\na = Singleton()\nb = Singleton()\nprint(a is b)  # True - same instance\n\n# Alternative: Using decorator\ndef singleton(cls):\n    instances = {}\n    def get_instance(*args, **kwargs):\n        if cls not in instances:\n            instances[cls] = cls(*args, **kwargs)\n        return instances[cls]\n    return get_instance\n\n@singleton\nclass Database:\n    pass\n```"),

        ("Implement a simple Observer pattern.",
         "```python\nfrom typing import List, Callable\n\nclass Subject:\n    def __init__(self):\n        self._observers: List[Callable] = []\n        self._state = None\n    \n    def attach(self, observer: Callable):\n        self._observers.append(observer)\n    \n    def detach(self, observer: Callable):\n        self._observers.remove(observer)\n    \n    def notify(self):\n        for observer in self._observers:\n            observer(self._state)\n    \n    @property\n    def state(self):\n        return self._state\n    \n    @state.setter\n    def state(self, value):\n        self._state = value\n        self.notify()\n\n# Usage\ndef observer1(state):\n    print(f'Observer 1 received: {state}')\n\ndef observer2(state):\n    print(f'Observer 2 received: {state}')\n\nsubject = Subject()\nsubject.attach(observer1)\nsubject.attach(observer2)\nsubject.state = 'New State'  # Both observers notified\n```"),

        # Data structures
        ("Implement a stack with push, pop, and get_min in O(1) time.",
         "```python\nclass MinStack:\n    def __init__(self):\n        self.stack = []      # Main stack\n        self.min_stack = []  # Track minimums\n    \n    def push(self, val: int) -> None:\n        self.stack.append(val)\n        # Push to min_stack if empty or val <= current min\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n    \n    def pop(self) -> int:\n        if not self.stack:\n            raise IndexError('Stack is empty')\n        val = self.stack.pop()\n        # Pop from min_stack if we're removing the min\n        if val == self.min_stack[-1]:\n            self.min_stack.pop()\n        return val\n    \n    def top(self) -> int:\n        if not self.stack:\n            raise IndexError('Stack is empty')\n        return self.stack[-1]\n    \n    def get_min(self) -> int:\n        if not self.min_stack:\n            raise IndexError('Stack is empty')\n        return self.min_stack[-1]\n\n# Usage\nstack = MinStack()\nstack.push(3)\nstack.push(1)\nstack.push(2)\nprint(stack.get_min())  # 1\nstack.pop()\nprint(stack.get_min())  # 1\nstack.pop()\nprint(stack.get_min())  # 3\n```"),

        ("Implement a LRU (Least Recently Used) Cache.",
         "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n    \n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        # Move to end (most recently used)\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    \n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            # Update and move to end\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        # Evict oldest if over capacity\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n\n# Usage\ncache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\nprint(cache.get(1))    # 1\ncache.put(3, 3)        # Evicts key 2\nprint(cache.get(2))    # -1 (not found)\n```"),

        # Algorithms
        ("Implement merge sort in Python.",
         "```python\ndef merge_sort(arr: list) -> list:\n    if len(arr) <= 1:\n        return arr\n    \n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    \n    return merge(left, right)\n\ndef merge(left: list, right: list) -> list:\n    result = []\n    i = j = 0\n    \n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    \n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n\n# Usage\narr = [64, 34, 25, 12, 22, 11, 90]\nprint(merge_sort(arr))  # [11, 12, 22, 25, 34, 64, 90]\n```\n\nTime complexity: O(n log n)\nSpace complexity: O(n)"),

        ("Implement BFS and DFS for a graph.",
         "```python\nfrom collections import deque\n\ndef bfs(graph: dict, start: str) -> list:\n    \"\"\"Breadth-First Search - visit neighbors first.\"\"\"\n    visited = set()\n    queue = deque([start])\n    result = []\n    \n    while queue:\n        node = queue.popleft()\n        if node not in visited:\n            visited.add(node)\n            result.append(node)\n            queue.extend(n for n in graph[node] if n not in visited)\n    \n    return result\n\ndef dfs(graph: dict, start: str, visited=None) -> list:\n    \"\"\"Depth-First Search - go deep first.\"\"\"\n    if visited is None:\n        visited = set()\n    \n    visited.add(start)\n    result = [start]\n    \n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            result.extend(dfs(graph, neighbor, visited))\n    \n    return result\n\n# Usage\ngraph = {\n    'A': ['B', 'C'],\n    'B': ['A', 'D', 'E'],\n    'C': ['A', 'F'],\n    'D': ['B'],\n    'E': ['B', 'F'],\n    'F': ['C', 'E']\n}\n\nprint(bfs(graph, 'A'))  # ['A', 'B', 'C', 'D', 'E', 'F']\nprint(dfs(graph, 'A'))  # ['A', 'B', 'D', 'E', 'F', 'C']\n```"),

        # Async programming
        ("Write an async function in Python to fetch multiple URLs.",
         "```python\nimport asyncio\nimport aiohttp\n\nasync def fetch_url(session, url: str) -> dict:\n    \"\"\"Fetch a single URL.\"\"\"\n    async with session.get(url) as response:\n        return {\n            'url': url,\n            'status': response.status,\n            'content': await response.text()\n        }\n\nasync def fetch_all(urls: list) -> list:\n    \"\"\"Fetch multiple URLs concurrently.\"\"\"\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_url(session, url) for url in urls]\n        results = await asyncio.gather(*tasks, return_exceptions=True)\n        return results\n\n# Usage\nasync def main():\n    urls = [\n        'https://api.example.com/data1',\n        'https://api.example.com/data2',\n        'https://api.example.com/data3'\n    ]\n    results = await fetch_all(urls)\n    for result in results:\n        print(result)\n\n# Run\nasyncio.run(main())\n```"),

        # Testing
        ("Write unit tests for a calculator class using pytest.",
         "```python\n# calculator.py\nclass Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError('Cannot divide by zero')\n        return a / b\n\n# test_calculator.py\nimport pytest\nfrom calculator import Calculator\n\nclass TestCalculator:\n    @pytest.fixture\n    def calc(self):\n        return Calculator()\n    \n    def test_add(self, calc):\n        assert calc.add(2, 3) == 5\n        assert calc.add(-1, 1) == 0\n        assert calc.add(0, 0) == 0\n    \n    def test_subtract(self, calc):\n        assert calc.subtract(5, 3) == 2\n        assert calc.subtract(1, 1) == 0\n    \n    def test_multiply(self, calc):\n        assert calc.multiply(3, 4) == 12\n        assert calc.multiply(0, 5) == 0\n    \n    def test_divide(self, calc):\n        assert calc.divide(10, 2) == 5\n        assert calc.divide(7, 2) == 3.5\n    \n    def test_divide_by_zero(self, calc):\n        with pytest.raises(ValueError):\n            calc.divide(10, 0)\n\n# Run: pytest test_calculator.py -v\n```"),
    ]

    print(f"Generated {len(pairs)} advanced coding pairs")
    return pairs


def generate_system_design() -> List[Tuple[str, str]]:
    """Generate system design and architecture Q&A."""
    pairs = [
        ("How would you design a URL shortener like bit.ly?",
         "**URL Shortener Design:**\n\n**Requirements:**\n- Generate short URLs from long URLs\n- Redirect short URLs to original\n- Handle high read traffic\n- Optional: analytics, custom URLs\n\n**Components:**\n\n1. **API Service**\n   - POST /shorten - create short URL\n   - GET /{shortCode} - redirect\n\n2. **Short Code Generation**\n   - Base62 encoding (a-z, A-Z, 0-9)\n   - 7 chars = 62^7 = 3.5 trillion combinations\n   - Counter-based or random\n\n3. **Database**\n   - Key-value store (Redis) for fast lookups\n   - SQL for persistence: {shortCode, longUrl, createdAt, clicks}\n\n4. **Scaling**\n   - Cache popular URLs in Redis\n   - Multiple app servers behind load balancer\n   - Database sharding by shortCode\n\n**Flow:**\n```\nUser -> Load Balancer -> App Server -> Cache (Redis)\n                                    -> Database\n```"),

        ("Design a rate limiter for an API.",
         "**Rate Limiter Design:**\n\n**Algorithms:**\n\n1. **Token Bucket** (recommended)\n   - Bucket holds tokens, refilled at constant rate\n   - Each request consumes a token\n   - Allows bursts up to bucket size\n\n2. **Sliding Window**\n   - Count requests in rolling time window\n   - More accurate, slightly more complex\n\n**Implementation (Token Bucket):**\n```python\nimport time\nimport redis\n\nclass RateLimiter:\n    def __init__(self, redis_client, max_tokens=10, refill_rate=1):\n        self.redis = redis_client\n        self.max_tokens = max_tokens\n        self.refill_rate = refill_rate  # tokens per second\n    \n    def is_allowed(self, user_id: str) -> bool:\n        key = f'rate_limit:{user_id}'\n        now = time.time()\n        \n        # Get current state\n        data = self.redis.hgetall(key)\n        tokens = float(data.get('tokens', self.max_tokens))\n        last_refill = float(data.get('last_refill', now))\n        \n        # Refill tokens\n        elapsed = now - last_refill\n        tokens = min(self.max_tokens, tokens + elapsed * self.refill_rate)\n        \n        # Check and consume\n        if tokens >= 1:\n            tokens -= 1\n            allowed = True\n        else:\n            allowed = False\n        \n        # Save state\n        self.redis.hset(key, mapping={'tokens': tokens, 'last_refill': now})\n        self.redis.expire(key, 60)  # Cleanup\n        \n        return allowed\n```\n\n**Headers to return:**\n- X-RateLimit-Limit: max requests\n- X-RateLimit-Remaining: remaining\n- X-RateLimit-Reset: when limit resets"),

        ("How would you design a chat application like WhatsApp?",
         "**Chat Application Design:**\n\n**Requirements:**\n- Real-time messaging (1:1 and group)\n- Message delivery/read receipts\n- Online/offline status\n- Message history\n- Media sharing\n\n**Architecture:**\n\n1. **Client Layer**\n   - Mobile/web apps\n   - WebSocket connection for real-time\n\n2. **Connection Service**\n   - Manages WebSocket connections\n   - Routes messages to correct recipients\n   - Tracks online status\n\n3. **Message Service**\n   - Stores messages\n   - Handles delivery logic\n   - Manages groups\n\n4. **Database**\n   - User data: PostgreSQL\n   - Messages: Cassandra (scalable writes)\n   - Cache: Redis (online status, recent messages)\n   - Media: S3 or similar\n\n**Message Flow:**\n```\n1. User A sends message\n2. Message Service stores it\n3. Connection Service finds User B's server\n4. Delivers via WebSocket if online\n5. If offline, stores for later delivery\n6. Returns delivery receipt to A\n```\n\n**Scaling:**\n- Shard by user ID\n- Use consistent hashing for connection servers\n- Message queues for async processing\n- CDN for media"),

        ("Design a notification system.",
         "**Notification System Design:**\n\n**Types:**\n- Push notifications (mobile)\n- Email\n- SMS\n- In-app\n\n**Architecture:**\n\n```\n                    ┌─────────────────┐\n                    │ Notification    │\n┌─────────┐        │ Service         │\n│ Events  │───────>│ (prioritize,    │\n│ Queue   │        │  dedupe, route) │\n└─────────┘        └────────┬────────┘\n                            │\n         ┌──────────────────┼──────────────────┐\n         ▼                  ▼                  ▼\n   ┌───────────┐     ┌───────────┐     ┌───────────┐\n   │ Push      │     │ Email     │     │ SMS       │\n   │ Handler   │     │ Handler   │     │ Handler   │\n   └─────┬─────┘     └─────┬─────┘     └─────┬─────┘\n         ▼                 ▼                  ▼\n   ┌───────────┐     ┌───────────┐     ┌───────────┐\n   │ APNs/FCM  │     │ SendGrid  │     │ Twilio    │\n   └───────────┘     └───────────┘     └───────────┘\n```\n\n**Key Components:**\n\n1. **Event Producer**: Services emit notification events\n2. **Message Queue**: Kafka/RabbitMQ for reliability\n3. **Notification Service**:\n   - Rate limiting per user\n   - Deduplication\n   - User preference check\n   - Template rendering\n4. **Channel Handlers**: Platform-specific delivery\n\n**Database Schema:**\n```sql\nnotifications: id, user_id, type, content, status, created_at\nuser_preferences: user_id, channel, enabled, quiet_hours\ntemplates: id, name, content, variables\n```\n\n**Reliability:**\n- Retry with exponential backoff\n- Dead letter queue for failures\n- Idempotency keys"),
    ]

    print(f"Generated {len(pairs)} system design pairs")
    return pairs


# =============================================================================
# SOPHISTICATED ANALYSIS & OPINION Q&A - For nuanced questions like "does X suck"
# =============================================================================

def generate_company_tech_analysis() -> List[Tuple[str, str]]:
    """Generate nuanced analysis Q&A about companies, tech, and opinions."""
    pairs = [
        # AI Companies Analysis
        ("Does Anthropic suck?",
         "That's a nuanced question. Let me give you a balanced analysis:\n\n**Strengths:**\n- Founded by ex-OpenAI researchers (Dario & Daniela Amodei) focused on AI safety\n- Claude models are competitive with GPT-4 in many benchmarks\n- Strong focus on constitutional AI and reducing harmful outputs\n- More transparent about limitations than some competitors\n- Good at following nuanced instructions\n\n**Criticisms:**\n- Smaller model ecosystem than OpenAI\n- API can be more expensive for some use cases\n- Sometimes overly cautious/refuses reasonable requests\n- Less brand recognition than OpenAI\n- Slower to ship consumer products\n\n**Verdict:** Anthropic doesn't suck - they're a serious AI lab doing important safety research while shipping competitive models. Whether they're \"better\" depends on your specific needs. Their focus on safety is either a feature or a limitation depending on your perspective."),

        ("Is Anthropic better than OpenAI?",
         "It depends on what you need:\n\n**Anthropic (Claude) strengths:**\n- Better at nuanced instructions and following complex prompts\n- Stronger safety focus, less likely to produce harmful content\n- Longer context windows (200k tokens)\n- Often more honest about uncertainty\n- Better at coding tasks in some benchmarks\n\n**OpenAI (GPT-4) strengths:**\n- Larger ecosystem (DeepSeeker LLM, plugins, GPTs, DALL-E)\n- More third-party integrations\n- Stronger brand recognition\n- GPT-4 Vision is excellent\n- More features (function calling, fine-tuning options)\n\n**Choose Anthropic if:** You need reliability, safety, long context, or nuanced responses\n**Choose OpenAI if:** You need the ecosystem, integrations, or multimodal capabilities\n\nBoth are excellent. Competition is good for users."),

        ("Is OpenAI good or bad?",
         "OpenAI is complex - here's a balanced take:\n\n**The Good:**\n- Pioneered modern LLMs (GPT series)\n- Made AI accessible through DeepSeeker LLM\n- Strong research output (DALL-E, Whisper, Codex)\n- Pushed the entire field forward\n- Good developer tools and API\n\n**The Controversial:**\n- Started as non-profit, became capped-profit (some see as mission drift)\n- Sam Altman firing/rehiring drama raised governance questions\n- Rapid deployment vs. careful safety testing debate\n- Closed-source despite \"Open\" in name\n- Competitive practices some find aggressive\n\n**The Concerns:**\n- Moving fast may mean moving recklessly\n- Concentration of AI power in one company\n- Training data and copyright issues\n\n**Bottom line:** OpenAI has done more to advance and democratize AI than almost anyone, but they're not saints. They're a company with business interests that sometimes conflict with their stated mission."),

        ("What do you think of Google?",
         "Google is a fascinating case study in tech evolution:\n\n**What they do well:**\n- Search is still unmatched for most queries\n- Android dominates mobile globally\n- YouTube is the dominant video platform\n- Cloud infrastructure is world-class\n- Deep AI/ML research (invented Transformers!)\n- Maps, Gmail, Chrome - genuinely useful products\n\n**Valid criticisms:**\n- Privacy concerns are real - their business IS your data\n- Kills products people love (Google Reader, etc.)\n- Search quality arguably declining (more ads, SEO spam)\n- Antitrust issues in multiple jurisdictions\n- Can feel like a monopoly in several markets\n- Internal bureaucracy slows innovation\n\n**The irony:** Google invented the Transformer architecture that powers DeepSeeker LLM but was slow to ship competitive AI products.\n\n**Overall:** Google makes genuinely useful products but their scale and data practices warrant scrutiny."),

        ("Is Meta/Facebook evil?",
         "Let me give you a nuanced take:\n\n**Legitimate concerns:**\n- Cambridge Analytica showed serious data governance failures\n- Internal research showed Instagram harms teen mental health (and they knew)\n- Algorithm optimizes for engagement, which can mean outrage\n- Spread of misinformation on platforms\n- Privacy practices have been questionable historically\n\n**In their defense:**\n- They do employ thousands working on safety\n- Free services that billions genuinely use and value\n- WhatsApp provides essential communication in developing countries\n- Open-sourced important AI models (LLaMA)\n\n**The real issue:** Facebook isn't \"evil\" in a cartoon villain sense - they're a company whose business model creates incentives that sometimes harm society. The question is whether that's fixable.\n\n**My take:** They've caused real harm through negligence and misaligned incentives, but \"evil\" oversimplifies."),

        ("Is Amazon a good company?",
         "Amazon is complicated:\n\n**What they do well:**\n- Incredible logistics and delivery infrastructure\n- AWS basically built cloud computing\n- Customer obsession genuinely shows in product\n- Prime provides real value for many\n- Kindle revolutionized reading\n\n**Legitimate concerns:**\n- Warehouse worker conditions (exhausting, injuries)\n- Crushing small competitors then copying their products\n- Antitrust concerns (platform and seller)\n- Environmental impact of fast shipping\n- Tax avoidance strategies\n\n**The business reality:**\n- E-commerce margins are thin; AWS prints money\n- They disrupted retail but also employ hundreds of thousands\n- They set delivery expectations that are hard to match\n\n**Bottom line:** Amazon delivers incredible value to consumers while raising real concerns about workers, competition, and market power. Whether that's \"good\" depends on how you weigh those factors."),

        ("Is cryptocurrency a scam?",
         "This is nuanced:\n\n**What's legitimate:**\n- Blockchain technology has real uses\n- Bitcoin as \"digital gold\" has some merit\n- Useful for remittances and banking the unbanked\n- DeFi enables interesting financial primitives\n\n**What's often scammy:**\n- Most altcoins/memecoins are pump-and-dumps\n- Many NFT projects were cash grabs\n- Lots of fraud, rug pulls, and Ponzi schemes\n- Celebrity endorsements are usually paid shilling\n\n**The honest take:**\n- Bitcoin and Ethereum have survived 10+ years\n- 95%+ of crypto projects are worthless or scams\n- The technology is real; the speculation is often not\n\n**Advice:** If you invest, stick to established projects, never invest more than you can lose, and be extremely skeptical of anything promising high returns."),

        ("Should I use AWS or Google Cloud or Azure?",
         "It depends on your needs:\n\n**Choose AWS if:**\n- Need the widest service selection (200+ services)\n- Enterprise/large-scale deployments\n- Most mature ecosystem\n- Your team already knows AWS\n\n**Choose Google Cloud if:**\n- Heavy on data analytics/BigQuery workloads\n- Using Kubernetes (GKE is excellent)\n- Need strong AI/ML services\n- Prefer simpler pricing\n\n**Choose Azure if:**\n- Microsoft shop (Office 365, Windows servers)\n- Enterprise with existing Microsoft contracts\n- Hybrid cloud scenarios\n- .NET workloads\n\n**Honest truth:** For most projects, all three work fine. Pick based on your team's expertise."),

        ("Is AI going to take my job?",
         "Honest answer: Maybe, but probably not how you think.\n\n**Jobs at higher risk:**\n- Routine data entry and processing\n- Basic content writing (SEO articles, product descriptions)\n- Simple customer service (chatbots improving fast)\n- Some coding tasks (boilerplate, simple scripts)\n- Basic graphic design\n\n**Jobs that are safer (for now):**\n- Roles requiring physical presence\n- High-stakes decision making needing accountability\n- Creative work requiring true originality\n- Jobs requiring emotional intelligence\n- Complex problem-solving in ambiguous situations\n\n**The more likely scenario:**\n- AI augments most jobs rather than replacing them\n- You'll use AI tools to be more productive\n- New jobs will emerge we can't predict\n\n**What to do:**\n1. Learn to use AI tools effectively\n2. Focus on skills AI struggles with\n3. Stay adaptable and keep learning"),

        ("Is college worth it anymore?",
         "This is genuinely complicated. Here's my honest take:\n\n**When college IS worth it:**\n- Careers requiring degrees (medicine, law, engineering)\n- Strong schools with good career outcomes\n- You have clear goals and will leverage the degree\n- Scholarships make it affordable\n\n**When to reconsider:**\n- Taking massive debt for a low-ROI degree\n- Going \"because that's what you do\" without a plan\n- Fields where skills matter more than credentials (tech, trades)\n- You learn better through self-study or experience\n\n**Alternatives that work:**\n- Coding bootcamps (for motivated people)\n- Trade schools (electricians, plumbers earn well)\n- Community college + transfer\n- Self-study + portfolio + networking\n\n**The nuance:** College isn't universally \"worth it\" or \"not worth it\" - it depends on your field, school, costs, and how you use it."),

        ("What programming language should I learn first?",
         "**Short answer:** Python for most people.\n\n**Why Python:**\n- Readable syntax, gentle learning curve\n- Versatile (web, data science, automation, AI/ML)\n- Huge job market\n- Excellent libraries and community\n\n**But consider JavaScript if:**\n- You want to build websites\n- You want to see visual results quickly\n- Full-stack web development\n\n**Other options:**\n- Java/C#: Enterprise jobs, Android, game dev\n- Go/Rust: Systems programming, performance\n- SQL: Everyone needs database skills\n\n**What NOT to start with:**\n- C/C++ (memory management is hard)\n- Haskell/Lisp (steep curve)\n\n**The truth:** The first language matters less than actually building things. Pick Python or JavaScript and start."),

        ("Is Elon Musk a genius or a fraud?",
         "The honest answer is: neither extreme is accurate.\n\n**What's genuinely impressive:**\n- SpaceX has revolutionized rocket reusability\n- Tesla accelerated EV adoption by decades\n- Built multiple successful companies in hard industries\n- Takes on ambitious projects others won't\n\n**What's overhyped:**\n- He's a CEO/product visionary, not doing the engineering himself\n- Credit often goes to him rather than his teams\n- Many promises are late or don't happen\n- Twitter/X acquisition has been chaotic\n\n**Legitimate criticisms:**\n- Treatment of employees can be harsh\n- Twitter behavior is often erratic\n- Market manipulation concerns\n- Overpromises timelines consistently\n\n**The balanced view:** He's a genuinely talented entrepreneur who's built real companies. He's also prone to exaggeration and poor judgment on social media."),

        ("Why do tech companies do layoffs after hiring so much?",
         "Great question:\n\n**The hiring spree (2020-2022):**\n- COVID accelerated digital adoption dramatically\n- Interest rates were near zero (cheap money)\n- Tech stocks were sky-high\n- Companies assumed pandemic trends were permanent\n- FOMO: competitors hiring meant you had to hire\n\n**The correction (2022-2024):**\n- Interest rates rose sharply\n- Growth slowed as pandemic boost faded\n- Investors demanded profitability over growth\n- AI made some roles redundant\n\n**The uncomfortable truth:**\n- Tech overhired based on unsustainable projections\n- \"Growth at all costs\" met \"show me the profits\"\n- Companies optimize for shareholders, not employees\n\n**What it means:** Tech is cyclical like other industries. \"Tech job = job security\" was always somewhat myth."),

        ("Is TikTok actually dangerous?",
         "Let me separate the different concerns:\n\n**Legitimate concerns:**\n- Algorithm is extremely effective at capturing attention\n- Mental health impacts, especially on young users\n- Can spread misinformation quickly\n- Data collection is extensive\n\n**The China/security angle:**\n- ByteDance is Chinese\n- Chinese law could compel data sharing\n- Real security experts are genuinely concerned\n- BUT: No public evidence of actual data misuse yet\n\n**What's overblown:**\n- \"TikTok is rotting brains\" - Instagram/YouTube aren't better\n- Unique danger vs. other social media is debatable\n\n**My take:** TikTok's algorithm is more addictive than competitors. The China concerns are legitimate but sometimes used as cover for protectionism. The real issue is social media's impact generally.\n\n**What to do:** Set time limits, be conscious of usage, especially for kids."),
    ]

    print(f"Generated {len(pairs)} company/tech analysis pairs")
    return pairs


def generate_general_knowledge_comprehensive() -> List[Tuple[str, str]]:
    """Generate comprehensive general knowledge Q&A."""
    pairs = [
        # Science
        ("How does gravity work?",
         "Gravity works through the curvature of spacetime:\n\n**The basic idea (Einstein's General Relativity):**\n- Mass and energy curve the fabric of spacetime\n- Objects follow the straightest path through curved spacetime\n- What we perceive as gravity is objects following curved paths\n\n**Analogy:** A bowling ball on a trampoline creates a dip. Roll a marble nearby and it curves toward the bowling ball.\n\n**Key facts:**\n- Gravity is the weakest fundamental force\n- But it acts over infinite distance and only attracts\n- Gravitational waves were detected in 2015\n\n**What we don't fully understand:** How to unify gravity with quantum mechanics."),

        ("What causes cancer?",
         "Cancer is caused by mutations in DNA that allow cells to grow uncontrollably:\n\n**The mechanism:**\n1. Normal cells have genes controlling growth and death\n2. Mutations damage these control systems\n3. Cells divide uncontrollably\n4. They can spread (metastasis)\n\n**What causes mutations:**\n- Inherited (~5-10%)\n- Lifestyle: Smoking, alcohol, obesity\n- Environmental: UV radiation, pollution\n- Infections: HPV, hepatitis\n- Random errors during cell division\n- Age\n\n**Prevention:** Don't smoke, limit alcohol, maintain healthy weight, use sunscreen, get vaccinated (HPV), do recommended screenings."),

        ("What happened to the dinosaurs?",
         "Non-avian dinosaurs went extinct ~66 million years ago from an asteroid impact:\n\n**The impact:**\n- Asteroid ~10-15 km wide hit Mexico (Chicxulub crater)\n- Impact energy: billions of nuclear bombs\n- Triggered earthquakes, tsunamis, wildfires globally\n\n**How it killed them:**\n1. Immediate: Heat pulse, shockwave\n2. Short-term: Wildfires, acid rain, tsunamis\n3. Long-term: Dust blocked sunlight for months/years\n4. Photosynthesis collapsed, food chains broke\n\n**What survived:** Small animals, burrowing animals, some marine life, birds (which ARE dinosaurs).\n\n**Fun fact:** If the asteroid hit hours earlier or later, it might have hit ocean and dinosaurs might still be here."),

        ("Why is the sky blue?",
         "The sky is blue because of Rayleigh scattering:\n\n**How it works:**\n1. Sunlight contains all colors (white light)\n2. Light hits air molecules in atmosphere\n3. Shorter wavelengths (blue) scatter more than longer wavelengths (red)\n4. Blue light scatters in all directions\n5. When you look at sky, you see scattered blue light\n\n**Why sunsets are red/orange:**\n- At sunset, light travels through more atmosphere\n- Blue light scatters away before reaching you\n- Longer wavelengths (red/orange) make it through\n\n**Fun fact:** On Mars, the sky is butterscotch/tan colored because of dust particles that scatter light differently."),

        ("How do vaccines work?",
         "Vaccines train your immune system to fight diseases without causing the disease:\n\n**The process:**\n1. Vaccine introduces something that looks like the pathogen\n2. Immune system recognizes it as foreign and attacks\n3. Body creates \"memory cells\" that remember the threat\n4. If exposed to real pathogen, body responds immediately\n\n**Types:**\n- Live attenuated: Weakened virus (MMR)\n- Inactivated: Killed pathogen (flu shot)\n- Subunit: Pieces of pathogen (HPV)\n- mRNA: Instructions to make harmless piece (COVID Pfizer/Moderna)\n\n**Common misconceptions clarified:**\n- Don't \"overwhelm\" immune systems\n- Side effects mean immune system is responding (good!)\n- Ingredients are in tiny, well-studied amounts"),

        # History
        ("Why did Rome fall?",
         "The Roman Empire's fall was gradual with multiple causes:\n\n**Political:** Unstable succession, corruption, overreliance on mercenaries\n\n**Military:** Constant barbarian invasions, difficulty defending vast borders\n\n**Economic:** Currency debasement (inflation), heavy taxation, trade disruption\n\n**Social:** Population decline, loss of civic virtue\n\n**What people get wrong:**\n- It wasn't sudden - took centuries\n- Eastern Roman Empire lasted until 1453\n- Christianity didn't \"cause\" it (common myth)\n\n**The lesson:** Complex systems fail from accumulated problems, not single causes."),

        ("What caused World War 1?",
         "WWI resulted from multiple factors:\n\n**Immediate trigger:** Assassination of Archduke Franz Ferdinand (1914)\n\n**Underlying causes (MAIN):**\n\n**M - Militarism:** Arms race, especially naval (Britain vs Germany)\n\n**A - Alliances:** Triple Entente vs Triple Alliance - attack one meant war with all\n\n**I - Imperialism:** Competition for colonies created tensions\n\n**N - Nationalism:** Ethnic groups wanted independence, aggressive German nationalism\n\n**Why it escalated:**\n- Mobilization plans were inflexible\n- Leaders underestimated how bad it would be\n- Expected a short war, got 4 years\n\n**Result:** ~17 million dead, empires collapsed, set stage for WWII."),

        ("What was the Cold War?",
         "The Cold War (1947-1991) was a geopolitical rivalry:\n\n**The sides:**\n- USA and allies (NATO, capitalist democracy)\n- USSR and allies (Warsaw Pact, communist)\n\n**Key events:**\n- Berlin Wall (1961 built, 1989 fell)\n- Cuban Missile Crisis (1962)\n- Space Race\n- Proxy wars (Korea, Vietnam, Afghanistan)\n- Arms race and nuclear proliferation\n\n**Why \"cold\":** Never became direct \"hot\" war between superpowers (nukes made it too risky)\n\n**How it ended:** Soviet Union dissolved in 1991 due to economic problems and reforms under Gorbachev."),

        # Society / Current Affairs
        ("Why is housing so expensive?",
         "Housing costs have outpaced income due to:\n\n**Supply problems:**\n- Zoning laws restrict building\n- NIMBYism: existing owners oppose development\n- Construction costs rising\n- Approval processes take years\n\n**Demand factors:**\n- Population growth in popular cities\n- Investment buyers\n- Airbnb reduces long-term supply\n\n**Financial factors:**\n- Low interest rates made buying attractive\n- Housing seen as investment\n- Institutional investors buying homes\n\n**The uncomfortable truth:** Existing homeowners benefit from high prices and have political power to block solutions."),

        ("Why is healthcare expensive in the US?",
         "US healthcare costs more due to systemic issues:\n\n**Pricing problems:**\n- No price controls on drugs or procedures\n- Massive administrative costs (~30% of spending)\n- Every hospital and insurer has billing departments\n\n**Structural issues:**\n- Fee-for-service incentivizes more procedures\n- Defensive medicine (extra tests to avoid lawsuits)\n- Lack of competition in many markets\n\n**What we pay more for:**\n- Same drugs cost 2-3x other countries\n- Administrative overhead highest in world\n- Specialist doctors earn much more\n\n**What other countries do:** Government negotiates prices, standardized billing, more primary care.\n\n**Result:** US spends ~18% of GDP on healthcare with worse outcomes than many countries spending half as much."),

        ("What is climate change doing?",
         "Climate change is causing measurable, accelerating impacts:\n\n**What's happening now:**\n- Global temperature up ~1.1°C since pre-industrial\n- Sea levels rising ~3.6mm/year (accelerating)\n- Arctic ice declining ~13% per decade\n- Extreme weather increasing\n\n**Effects we're seeing:**\n- More intense hurricanes\n- Longer droughts\n- Increased flooding\n- Heat waves killing thousands\n- Larger wildfires\n- Coral reef bleaching\n\n**What's coming:**\n- 1.5°C warming likely by 2030s\n- 2-4°C possible by 2100 without major action\n- More climate refugees\n- Agricultural disruption\n\n**The truth:** We know the problem, we have solutions, but political and economic obstacles slow action."),

        # Philosophy
        ("What is the meaning of life?",
         "Major perspectives:\n\n**Religious:** Serve God, reach afterlife, live according to divine will\n\n**Philosophical:**\n- Existentialism: Life has no inherent meaning; we create our own\n- Absurdism: Life is meaningless, but embrace it anyway\n- Aristotle: Flourishing through virtue and reason\n- Utilitarianism: Maximize happiness for all\n\n**Practical approaches:**\n- Find purpose through work, relationships, creativity\n- Help others\n- Experience and appreciate existence\n- Pursue knowledge and growth\n\n**My take:** Maybe meaning isn't found but created. Different things give different lives meaning. The search itself might be part of living well.\n\n**What often helps:** Connection with others, meaningful work, helping people, experiencing beauty, growth."),

        ("Is free will real?",
         "One of philosophy's hardest problems:\n\n**Arguments against:**\n- Brain activity precedes conscious decision\n- Everything caused by prior events (determinism)\n- Choices come from genes + environment + randomness\n\n**Arguments for:**\n- We experience making choices\n- Moral responsibility requires free will\n- Quantum indeterminacy introduces randomness\n\n**Compatibilist view (most common):**\n- \"Free will\" means acting according to your desires without coercion\n- You're free if you do what you want, even if wants are determined\n\n**My take:** We probably don't have libertarian free will (truly uncaused choices), but compatibilist version is meaningful for practical purposes."),

        # Practical
        ("How do I get better at learning?",
         "Evidence-based techniques:\n\n**Spaced repetition:** Review at increasing intervals, don't cram\n\n**Active recall:** Test yourself instead of re-reading\n\n**Interleaving:** Mix up topics in study sessions\n\n**Elaboration:** Connect new info to what you know, ask \"why\"\n\n**What doesn't work:**\n- Highlighting and re-reading (feels productive, isn't)\n- Cramming (short-term only)\n- Multitasking\n\n**Meta-learning:**\n- Sleep is crucial (consolidation)\n- Exercise improves cognition\n- Manage stress\n\n**Practical tips:**\n- Pomodoro: 25 min focused, 5 min break\n- Remove distractions\n- Start with hardest material when fresh"),

        ("How do I negotiate salary?",
         "**Before negotiation:**\n- Research market rates (Glassdoor, Levels.fyi)\n- Know your alternatives (other offers, current job)\n- Document your accomplishments\n\n**Key principles:**\n- Let them name a number first if possible\n- Anchor high (but reasonable)\n- Focus on value you provide\n- Negotiate whole package (salary, equity, PTO, remote)\n- Never accept on the spot\n\n**What to say:**\n- \"Based on my research, I was expecting [X-Y range]\"\n- \"Is there flexibility on base salary?\"\n- \"What would it take to get to [target]?\"\n\n**What NOT to do:**\n- Give a number first\n- Accept without countering\n- Apologize for negotiating\n- Lie about other offers\n\n**The truth:** Most people don't negotiate and leave money on the table. Companies expect it."),

        ("How do I be more confident?",
         "Confidence isn't a feeling to wait for - it's a skill to build:\n\n**The key insight:** Confidence comes from action, not vice versa. Act, gain evidence, then feel confident.\n\n**Build evidence:**\n- Start small and succeed\n- Track your wins\n- Review when doubting yourself\n\n**Physical confidence:**\n- Posture matters (stand straight, take up space)\n- Dress in a way that makes YOU feel good\n- Exercise\n- Make eye contact\n\n**Mental reframes:**\n- \"I'm not confident\" → \"I'm building confidence\"\n- \"They'll judge me\" → \"They're thinking about themselves\"\n- \"I might fail\" → \"I'll learn either way\"\n\n**What confident people actually do:** Act despite uncertainty, aren't afraid to be wrong, ask questions without embarrassment.\n\n**The truth:** Everyone is insecure sometimes. Confident people just act anyway."),

        ("How do I have a difficult conversation?",
         "**Before:**\n- Get clear on your goal\n- Assume good intent from other person\n- Pick right time/place (private, not rushed)\n\n**Framework - State → Explain → Ask:**\n\n1. **State** issue factually (no blame)\n   \"I've noticed reports have been late the past three weeks\"\n\n2. **Explain** impact\n   \"This affects my ability to update the client\"\n\n3. **Ask** for their perspective\n   \"Can you help me understand what's happening?\"\n   Then LISTEN\n\n**During:**\n- Stay calm\n- Listen more than talk\n- Avoid \"you always\" and \"you never\"\n- Focus on behavior and impact, not character\n\n**Helpful phrases:**\n- \"I want to understand your perspective\"\n- \"When [X happens], I feel [Y]\"\n- \"Can we find a solution together?\"\n\n**If heated:** Take a break - \"Let's pause and come back to this\""),

        ("How do I start investing with little money?",
         "You can start with very little:\n\n**Step 1: Build emergency fund first** (3-6 months expenses in high-yield savings)\n\n**Step 2: Use 401k if available** (especially get employer match - free money!)\n\n**Step 3: Open brokerage account** (Fidelity, Schwab, Vanguard have no minimums)\n\n**Step 4: Buy index funds**\n- Total stock market (VTI, FXAIX)\n- S&P 500 (VOO, SPY)\n- Start with whatever you can ($50, $100)\n\n**Key principles:**\n- Consistency beats timing\n- Keep fees under 0.2%\n- Don't check constantly\n- Time in market > timing market\n\n**Common mistakes:**\n- Waiting until you have \"enough\"\n- Trying to pick stocks\n- Panic selling in downturns\n\n**The math:** $200/month at 7% = ~$500k in 40 years. Start now."),

        ("How do I make friends as an adult?",
         "Adult friendship is hard because you lose built-in social structures:\n\n**Why it's hard:** No forced proximity (school), busy schedules, people have established groups\n\n**Strategies that work:**\n\n1. **Repeated unplanned interaction**\n   - Join recurring activities (sports, class, club)\n   - Regular attendance is key - same people, weekly\n   - Takes 50+ hours to form friendship\n\n2. **Be the initiator**\n   - Most people want friends but won't initiate\n   - Invite people to things\n   - Suggest specific plans, not \"we should hang out\"\n\n3. **Use apps and groups**\n   - Meetup.com\n   - Bumble BFF\n   - Local Discord, Reddit\n\n**Accept the awkwardness:** Adult friendship-making feels weird at first. Most people feel the same way.\n\n**Be patient:** Real friendships take time. Keep showing up."),
    ]

    print(f"Generated {len(pairs)} general knowledge pairs")
    return pairs


def generate_controversial_nuanced() -> List[Tuple[str, str]]:
    """Generate nuanced responses to controversial questions."""
    pairs = [
        ("Is capitalism good or bad?",
         "Capitalism is a tool with significant tradeoffs:\n\n**What it does well:**\n- Efficient resource allocation through prices\n- Innovation incentives (profit motive)\n- Lifted billions out of poverty over 200 years\n- Consumer choice and competition\n\n**Legitimate criticisms:**\n- Inequality tends to increase without intervention\n- Externalities (pollution, climate) not priced in\n- Market failures in healthcare, education, housing\n- Boom-bust cycles cause real suffering\n\n**The nuanced view:**\n- Pure capitalism doesn't exist anywhere\n- All successful economies are mixed (markets + regulation)\n- The question is how to structure the mix\n\n**My take:** Markets are powerful for creating wealth. But markets alone don't solve distribution or externalities. Good policy uses markets where they work and intervenes where they don't."),

        ("Should drugs be legalized?",
         "This requires separating different issues:\n\n**Arguments for legalization/decriminalization:**\n- War on drugs has failed\n- Creates violent black markets\n- Disproportionately punishes minorities\n- Portugal decriminalized with better outcomes\n- Personal freedom argument\n\n**Arguments against:**\n- Some drugs are very dangerous\n- Addiction causes real harm\n- Marketing could increase use\n\n**The nuanced position:**\n- Cannabis: Evidence supports legalization with regulation\n- Psychedelics: Growing evidence for decriminalization, medical use\n- Hard drugs: Decriminalization + treatment over criminalization\n\n**My take:** Drug policy should be based on harm reduction, not moral judgment. Criminalization has failed. Different substances need different approaches."),

        ("Is immigration good or bad?",
         "Immigration has complex effects:\n\n**Economic benefits:**\n- Fills labor shortages\n- Entrepreneurs start businesses at higher rates\n- Supports aging population tax base\n\n**Economic concerns:**\n- Can affect wages in some sectors (debated)\n- Short-term fiscal costs for services\n- Housing pressure\n\n**Social aspects:**\n- Cultural enrichment and diversity\n- Integration takes time and effort\n- Speed affects integration success\n\n**What research shows:**\n- Long-term economic impact is positive\n- Second generation typically integrates well\n- Speed matters for successful integration\n\n**My take:** Immigration is generally positive economically, but speed and integration matter. Dismissing concerns as racism doesn't help. Neither does scapegoating immigrants for problems caused by other factors."),

        ("Is social media bad for society?",
         "Social media has significant downsides we're still learning to manage:\n\n**Documented harms:**\n- Correlation with teen depression/anxiety\n- Addictive design (engagement optimization)\n- Misinformation spreads faster than corrections\n- Political polarization\n\n**Real benefits:**\n- Connection with distant people\n- Community for marginalized groups\n- Information access\n- Small business marketing\n\n**The nuance:**\n- Effects vary by platform, use pattern, individual\n- Active use better than passive scrolling\n- Young people more vulnerable\n- Design choices matter (algorithmic vs chronological)\n\n**My take:** Social media isn't inherently bad, but current implementations optimize for engagement over wellbeing. The harms are real. We need better design, regulation, and personal practices."),

        ("Are electric cars actually better?",
         "Yes, but the full picture is complex:\n\n**Manufacturing:**\n- EV battery production has higher emissions than gas car\n- Creates \"carbon debt\" that takes miles to pay off\n- Mining concerns for lithium, cobalt\n\n**Lifetime emissions:**\n- EVs have zero tailpipe emissions\n- Electricity source matters:\n  - Coal grid: EVs still usually better, smaller margin\n  - Renewable grid: EVs much better\n  - Grids getting cleaner over time\n\n**Break-even:** Typically 20-50k miles to offset manufacturing\n\n**What's often ignored:**\n- Still cause tire and road wear pollution\n- Best option is less driving overall\n\n**The conclusion:** EVs are better for climate than gas cars. They're not zero-impact. If you're driving, electric is the better choice."),

        ("Is universal basic income a good idea?",
         "UBI has real tradeoffs:\n\n**Arguments for:**\n- Simplifies welfare bureaucracy\n- No \"welfare cliff\" discouraging work\n- Provides security in age of automation\n- Addresses unpaid work (caregiving)\n\n**Arguments against:**\n- Extremely expensive (trillions annually)\n- May reduce work incentives\n- Inflation concerns\n- Might undermine other programs\n\n**What evidence shows:**\n- Pilots show minimal work reduction\n- Recipients spend on basics\n- Mental health improves\n- BUT: Pilots aren't same as permanent universal program\n\n**Funding challenges:** A meaningful UBI ($1000/month) costs ~$3 trillion/year in US\n\n**My take:** UBI addresses real problems but faces massive funding and political challenges. Smaller targeted programs might be more achievable."),

        ("Was dropping atomic bombs justified?",
         "One of history's most debated ethical questions:\n\n**Arguments it was justified:**\n- Ended war immediately\n- Avoided invasion (estimated millions of casualties)\n- Japan was not surrendering\n\n**Arguments it wasn't:**\n- Civilian targeting is inherently wrong (100k+ immediate deaths)\n- Japan was seeking surrender terms\n- Could have demonstrated on uninhabited area first\n- Long-term radiation effects\n\n**Historical complexity:**\n- Japanese military wanted to fight on; civilians and emperor wanted peace\n- Cold War considerations were a factor\n- We can't know counterfactual outcomes\n\n**My take:** This defies simple answers. Reasonable people disagree. The decision had valid military reasoning but troubling elements. It should strengthen commitment to nuclear non-use."),
    ]

    print(f"Generated {len(pairs)} controversial/nuanced pairs")
    return pairs


def load_all_instruction_data() -> List[Tuple[str, str]]:
    """Load all instruction-following data - WEB + GENERATED."""
    all_pairs = []

    print("=" * 60)
    print("DOWNLOADING WEB INSTRUCTION DATASETS")
    print("=" * 60)

    # ===== WEB DOWNLOADED DATASETS =====

    # Databricks Dolly (15k instruction pairs)
    dolly = download_dolly()
    all_pairs.extend(dolly)

    # OpenAssistant conversations (from HuggingFace)
    oasst = download_oasst()
    all_pairs.extend(oasst)

    # Capybara - high quality diverse data
    capybara = download_capybara()
    all_pairs.extend(capybara)

    # LMSYS Chat - real user conversations
    lmsys = download_lmsys_chat()
    all_pairs.extend(lmsys)

    # ===== MATH & REASONING DATASETS =====
    print("\n--- Math & Reasoning ---")

    # GSM8K - Grade school math word problems
    gsm8k = download_gsm8k()
    all_pairs.extend(gsm8k)

    # MATH - Competition math problems
    math_data = download_math_dataset()
    all_pairs.extend(math_data)

    # ===== SCIENCE & QA DATASETS =====
    print("\n--- Science & QA ---")

    # SciQ - Science questions
    sciq = download_sciq()
    all_pairs.extend(sciq)

    # ARC - AI2 Reasoning Challenge
    arc = download_arc()
    all_pairs.extend(arc)

    # OpenBookQA - Science + commonsense
    openbookqa = download_openbookqa()
    all_pairs.extend(openbookqa)

    # ===== COMMONSENSE REASONING =====
    print("\n--- Commonsense Reasoning ---")

    # CommonsenseQA
    csqa = download_commonsenseqa()
    all_pairs.extend(csqa)

    # PIQA - Physical commonsense
    piqa = download_piqa()
    all_pairs.extend(piqa)

    # WinoGrande - Commonsense reasoning
    winogrande = download_winogrande()
    all_pairs.extend(winogrande)

    # HellaSwag - Sentence completion
    hellaswag = download_hellaswag()
    all_pairs.extend(hellaswag)

    # ===== READING COMPREHENSION & QA =====
    print("\n--- Reading Comprehension & QA ---")

    # BoolQ - Yes/No questions
    boolq = download_boolq()
    all_pairs.extend(boolq)

    # Natural Questions - Real Google questions
    nq = download_natural_questions()
    all_pairs.extend(nq)

    # SQuAD - Reading comprehension
    squad = download_squad()
    all_pairs.extend(squad)

    # TriviaQA - Trivia questions
    triviaqa = download_triviaqa()
    all_pairs.extend(triviaqa)

    # CoQA - Conversational QA
    coqa = download_coqa()
    all_pairs.extend(coqa)

    # ===== MORE INSTRUCTION DATA =====
    print("\n--- Additional Instruction Data ---")

    # FLAN - Google's instruction tuning
    flan = download_flan()
    all_pairs.extend(flan)

    # ===== ADDITIONAL HIGH-QUALITY DATASETS =====
    print("\n--- Additional High-Quality Datasets ---")

    # LIMA - High-quality curated pairs
    lima = download_lima()
    all_pairs.extend(lima)

    # MetaMathQA - Math reasoning
    metamath = download_metamathqa()
    all_pairs.extend(metamath)

    # Airoboros - Varied high-quality instructions
    airoboros = download_airoboros()
    all_pairs.extend(airoboros)

    # Pure-Dove - Clean conversation data
    pure_dove = download_pure_dove()
    all_pairs.extend(pure_dove)

    # Puffin - Multi-turn conversations
    puffin = download_puffin()
    all_pairs.extend(puffin)

    # Magicoder - Code instruction data
    magicoder = download_magicoder()
    all_pairs.extend(magicoder)

    # CodeFeedback - Code debugging data
    codefeedback = download_codefeedback()
    all_pairs.extend(codefeedback)

    # UltraFeedback - High-quality feedback data
    ultrafeedback = download_ultrafeedback()
    all_pairs.extend(ultrafeedback)

    # Orca-Math - Microsoft's math word problems
    orca_math = download_orca_math()
    all_pairs.extend(orca_math)

    web_total = len(all_pairs)
    print(f"\n>>> WEB-DOWNLOADED: {web_total} pairs")

    print("\n" + "=" * 60)
    print("GENERATING HIGH-QUALITY INSTRUCTION DATA")
    print("=" * 60)

    # ===== GENERATED INSTRUCTION DATA =====

    # Coding Q&A
    coding = generate_coding_qa()
    all_pairs.extend(coding)

    # Task completion instructions
    tasks = generate_task_completion()
    all_pairs.extend(tasks)

    # Advanced coding (design patterns, algorithms)
    advanced = generate_advanced_coding()
    all_pairs.extend(advanced)

    # System design questions
    system = generate_system_design()
    all_pairs.extend(system)

    # Company/tech analysis (Anthropic, OpenAI, Google, etc.)
    company_analysis = generate_company_tech_analysis()
    all_pairs.extend(company_analysis)

    # Comprehensive general knowledge
    general_knowledge = generate_general_knowledge_comprehensive()
    all_pairs.extend(general_knowledge)

    # Controversial/nuanced questions
    controversial = generate_controversial_nuanced()
    all_pairs.extend(controversial)

    generated_total = len(all_pairs) - web_total
    print(f"\n>>> GENERATED: {generated_total} pairs")

    print("\n" + "=" * 60)
    print(f"TOTAL INSTRUCTION PAIRS: {len(all_pairs)}")
    print("=" * 60)
    print("Sources:")
    print("  INSTRUCTION: Dolly, OASST, Capybara, LMSYS")
    print("  MATH/REASON: GSM8K, MATH, FLAN, MetaMathQA, Orca-Math")
    print("  SCIENCE/QA:  SciQ, ARC, OpenBookQA, SQuAD, CoQA, TriviaQA, NQ")
    print("  COMMONSENSE: CommonsenseQA, PIQA, WinoGrande, HellaSwag, BoolQ")
    print("  CODE:        Magicoder, CodeFeedback")
    print("  HIGH-QUAL:   LIMA, Airoboros, Pure-Dove, Puffin, UltraFeedback")
    print("  GENERATED:   Coding, Tasks, Advanced, System Design, Analysis")

    return all_pairs


def load_generated_instruction_data() -> List[Tuple[str, str]]:
    """Load only locally generated instruction data (no web downloads)."""
    all_pairs = []

    print("\n" + "=" * 60)
    print("GENERATING HIGH-QUALITY INSTRUCTION DATA (NO DOWNLOADS)")
    print("=" * 60)

    # Coding Q&A
    coding = generate_coding_qa()
    all_pairs.extend(coding)

    # Task completion instructions
    tasks = generate_task_completion()
    all_pairs.extend(tasks)

    # Advanced coding (design patterns, algorithms)
    advanced = generate_advanced_coding()
    all_pairs.extend(advanced)

    # System design questions
    system = generate_system_design()
    all_pairs.extend(system)

    # Company/tech analysis (Anthropic, OpenAI, Google, etc.)
    company_analysis = generate_company_tech_analysis()
    all_pairs.extend(company_analysis)

    # Comprehensive general knowledge
    general_knowledge = generate_general_knowledge_comprehensive()
    all_pairs.extend(general_knowledge)

    # Controversial/nuanced questions
    controversial = generate_controversial_nuanced()
    all_pairs.extend(controversial)

    print(f"\n>>> GENERATED: {len(all_pairs)} pairs")
    return all_pairs


def get_all_instruction_datasets(allow_download: bool = True) -> List[Tuple[str, str]]:
    """Return instruction datasets with optional web downloads."""
    if allow_download:
        return load_all_instruction_data()

    print("Skipping web downloads (allow_download=False)")
    return load_generated_instruction_data()


def save_instruction_data():
    """Download and save all instruction data to a single file."""
    all_pairs = load_all_instruction_data()

    output_file = CACHE_DIR / "instruction_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_pairs)} pairs to {output_file}")
    return all_pairs


if __name__ == "__main__":
    print("Downloading instruction-following datasets...")
    print("=" * 50)
    save_instruction_data()

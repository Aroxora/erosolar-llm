#!/usr/bin/env python3
"""
Retrieval-based kids Q&A system.

Instead of generating answers, this finds the most similar question
in our curated KIDS_QA dataset and returns that answer directly.

This GUARANTEES accurate, on-topic responses for known question types.
"""

import re
from difflib import SequenceMatcher
from data import KIDS_QA, QA_PAIRS, GREETINGS

# Build lookup database
KNOWLEDGE_BASE = []

# Add all curated Q&A pairs
for q, a in KIDS_QA:
    KNOWLEDGE_BASE.append((q.lower().strip(), a))

for q, a in QA_PAIRS:
    KNOWLEDGE_BASE.append((q.lower().strip(), a))

for q, a in GREETINGS:
    KNOWLEDGE_BASE.append((q.lower().strip(), a))

print(f"Loaded {len(KNOWLEDGE_BASE)} Q&A pairs into knowledge base")


def normalize(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower().strip()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def get_keywords(text: str) -> set:
    """Extract important keywords from text."""
    text = normalize(text)
    # Remove common words
    stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
                 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'what', 'which', 'who',
                 'whom', 'this', 'that', 'these', 'those', 'am', 'it', 'its',
                 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your',
                 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
                 'and', 'but', 'if', 'or', 'because', 'so', 'than', 'too',
                 'very', 'just', 'about', 'like', 'really', 'actually', 'pls',
                 'please', 'help', 'tell', 'explain', 'give', 'show', 'make'}
    words = text.split()
    return {w for w in words if w not in stopwords and len(w) > 2}


def similarity_score(query: str, candidate: str) -> float:
    """Calculate similarity between query and candidate question."""
    # Normalize both
    q_norm = normalize(query)
    c_norm = normalize(candidate)

    # Method 1: Sequence matching
    seq_score = SequenceMatcher(None, q_norm, c_norm).ratio()

    # Method 2: Keyword overlap (Jaccard)
    q_keywords = get_keywords(query)
    c_keywords = get_keywords(candidate)

    if q_keywords and c_keywords:
        intersection = len(q_keywords & c_keywords)
        union = len(q_keywords | c_keywords)
        keyword_score = intersection / union if union > 0 else 0
    else:
        keyword_score = 0

    # Method 3: Check for key topic words
    topic_boost = 0
    topic_words = {
        'dinosaur': ['dinosaur', 'extinct', 't-rex', 'triceratops', 'fossil'],
        'president': ['president', 'washington', 'lincoln', 'white house'],
        'planet': ['planet', 'solar system', 'mercury', 'mars', 'jupiter'],
        'gravity': ['gravity', 'fall', 'weight', 'newton'],
        'constitution': ['constitution', 'bill of rights', 'amendment', 'founding'],
        'thanksgiving': ['thanksgiving', 'turkey', 'pilgrim', 'november'],
        'july': ['july', 'independence', 'fireworks', 'fourth'],
        'math': ['plus', 'minus', 'times', 'divided', 'add', 'subtract', 'multiply'],
        'animal': ['dog', 'cat', 'bird', 'fish', 'animal', 'pet'],
        'space': ['moon', 'sun', 'star', 'rocket', 'astronaut', 'space'],
        'friend': ['friend', 'bully', 'school', 'lunch', 'talk'],
        'joke': ['joke', 'funny', 'laugh', 'riddle', 'knock knock'],
        'game': ['minecraft', 'fortnite', 'game', 'video game', 'pokemon'],
        'superhero': ['superhero', 'batman', 'superman', 'spider-man', 'avenger'],
    }

    q_lower = query.lower()
    c_lower = candidate.lower()
    for topic, words in topic_words.items():
        q_has = any(w in q_lower for w in words)
        c_has = any(w in c_lower for w in words)
        if q_has and c_has:
            topic_boost = 0.3
            break

    # Combine scores
    final_score = (seq_score * 0.3) + (keyword_score * 0.5) + topic_boost
    return min(final_score, 1.0)


def find_best_answer(query: str, threshold: float = 0.25) -> tuple:
    """Find the best matching answer from knowledge base."""
    best_score = 0
    best_answer = None
    best_question = None

    for question, answer in KNOWLEDGE_BASE:
        score = similarity_score(query, question)
        if score > best_score:
            best_score = score
            best_answer = answer
            best_question = question

    if best_score >= threshold:
        return best_answer, best_score, best_question
    return None, best_score, best_question


def answer(query: str, fallback_fn=None) -> str:
    """
    Get answer for a query.

    Args:
        query: The question to answer
        fallback_fn: Optional function to call if no good match found
                     Should take (query) and return str

    Returns:
        The answer string
    """
    result, score, matched_q = find_best_answer(query)

    if result:
        # Clean up answer (remove generation tags if present)
        result = re.sub(r'\[Generation:.*?\]', '', result).strip()
        return result

    # No good match - use fallback if provided
    if fallback_fn:
        return fallback_fn(query)

    # Default fallback
    return "That's a great question! I'd need to think about that more. Can you ask me something else, or rephrase your question?"


# Test
if __name__ == "__main__":
    test_questions = [
        "Why did dinosaurs go extinct?",
        "What are the three branches of government?",
        "Who was the first president?",
        "Tell me a funny joke",
        "How do I make friends at school?",
        "What is 47 + 18?",
        "Why is the sky blue?",
        "What is Thanksgiving about?",
        "How many planets are there?",
        "Who is the strongest superhero?",
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL-BASED KIDS Q&A TEST")
    print("=" * 60)

    for q in test_questions:
        ans, score, matched = find_best_answer(q)
        print(f"\n📝 Q: {q}")
        print(f"🔍 Match: {matched} (score: {score:.2f})")
        if ans:
            print(f"✅ A: {ans[:150]}...")
        else:
            print(f"❌ No match above threshold")

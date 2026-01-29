"""
Word-level Tokenizer for clean, readable output.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
from dataclasses import dataclass


@dataclass
class SpecialTokens:
    """Special tokens configuration."""
    pad: str = "<|pad|>"
    unk: str = "<|unk|>"
    bos: str = "<|bos|>"
    eos: str = "<|eos|>"
    # User/Assistant turn tokens
    user: str = "<|user|>"
    assistant: str = "<|assistant|>"
    end_turn: str = "<|end_turn|>"
    # Reasoning tokens for chain-of-thought
    think_start: str = "<|think_start|>"
    think_end: str = "<|think_end|>"
    step: str = "<|step|>"
    answer: str = "<|answer|>"

    def to_dict(self) -> Dict[str, int]:
        return {
            self.pad: 0, self.unk: 1, self.bos: 2, self.eos: 3,
            self.user: 4, self.assistant: 5, self.end_turn: 6,
            self.think_start: 7, self.think_end: 8, self.step: 9, self.answer: 10
        }


class BPETokenizer:
    """Word-level tokenizer (renamed for compatibility).

    Uses whole words as tokens for clean, readable output.
    Falls back to character-level for unknown words.
    """

    def __init__(self, special_tokens: Optional[SpecialTokens] = None):
        self.special_tokens = special_tokens or SpecialTokens()
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List = []  # Kept for compatibility

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.special_tokens.pad]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.special_tokens.bos]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.special_tokens.eos]

    @property
    def user_token_id(self) -> int:
        return self.token_to_id.get(self.special_tokens.user, -1)

    @property
    def assistant_token_id(self) -> int:
        return self.token_to_id.get(self.special_tokens.assistant, -1)

    @property
    def end_turn_id(self) -> int:
        return self.token_to_id.get(self.special_tokens.end_turn, -1)

    @property
    def think_start_id(self) -> int:
        return self.token_to_id.get(self.special_tokens.think_start, -1)

    @property
    def think_end_id(self) -> int:
        return self.token_to_id.get(self.special_tokens.think_end, -1)

    @property
    def step_id(self) -> int:
        return self.token_to_id.get(self.special_tokens.step, -1)

    @property
    def answer_id(self) -> int:
        return self.token_to_id.get(self.special_tokens.answer, -1)

    def train(self, text: str, vocab_size: int):
        """Train word-level tokenizer."""
        print("\nTraining word tokenizer...")

        # Start with special tokens
        self.token_to_id = self.special_tokens.to_dict()
        next_id = len(self.token_to_id)

        # Count word frequencies
        words = text.split()
        word_freqs = Counter(words)

        # Add most common words up to vocab size
        # Include words appearing only once to reduce <|unk|> tokens
        for word, freq in word_freqs.most_common(vocab_size - next_id):
            if word not in self.token_to_id:  # Include all words regardless of frequency
                self.token_to_id[word] = next_id
                next_id += 1
                if next_id >= vocab_size:
                    break

        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        print(f"  Vocab size: {len(self.token_to_id)}")

    def _encode_word(self, word: str) -> List[int]:
        """Encode a single word, with character-level fallback."""
        # Direct lookup
        if word in self.token_to_id:
            return [self.token_to_id[word]]

        # Try lowercase
        if word.lower() in self.token_to_id:
            return [self.token_to_id[word.lower()]]

        # Character-level fallback: try to split into known substrings
        result = []
        i = 0
        while i < len(word):
            # Try longest match first
            matched = False
            for length in range(min(len(word) - i, 10), 0, -1):
                substr = word[i:i+length]
                if substr in self.token_to_id:
                    result.append(self.token_to_id[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                # Single character not found, use UNK
                result.append(self.token_to_id[self.special_tokens.unk])
                i += 1

        return result if result else [self.token_to_id[self.special_tokens.unk]]

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs with character-level fallback."""
        ids = []
        if add_special:
            ids.append(self.bos_token_id)

        for word in text.split():
            ids.extend(self._encode_word(word))

        if add_special:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text with proper spacing."""
        special_ids = set(self.special_tokens.to_dict().values())
        tokens = []

        for id in ids:
            if id in self.id_to_token:
                if skip_special and id in special_ids:
                    continue
                tokens.append(self.id_to_token[id])

        # Join with spaces, then clean up
        text = ' '.join(tokens)

        # Fix spacing around punctuation and special chars
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' :', ':')
        text = text.replace(' ;', ';')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        text = text.replace(' )', ')')
        text = text.replace('( ', '(')
        text = text.replace(' ]', ']')
        text = text.replace('[ ', '[')
        text = text.replace(' }', '}')
        text = text.replace('{ ', '{')
        text = text.replace(" '", "'")
        text = text.replace(' "', '"')
        text = text.replace('\n ', '\n')
        text = text.replace(' \n', '\n')

        return text.strip()

    def save(self, path: Path):
        """Save tokenizer to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'tokenizer.json', 'w') as f:
            json.dump({
                'type': 'word',
                'vocab': self.token_to_id,
                'merges': self.merges
            }, f, indent=2)

    def load(self, path: Path):
        """Load tokenizer from disk."""
        with open(Path(path) / 'tokenizer.json', 'r') as f:
            data = json.load(f)
        self.special_tokens = SpecialTokens()
        self.token_to_id = data['vocab']
        self.merges = data.get('merges', [])
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

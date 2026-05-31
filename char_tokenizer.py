# SPDX-License-Identifier: AGPL-3.0-only
"""char_tokenizer.py — a character-level tokenizer mirroring the BPETokenizer interface
used by model.generate and the registry.

Why it exists: with word-level tokens, a held-out quality word has its own (untrainable,
held-out) embedding, so the model literally cannot emit it — zero-shot generalization is
pinned at 0%. With CHARACTER tokens, every quality is a sequence of characters that are all
seen during training, so the model CAN emit (and copy) an unseen quality. This is the lever
for real zero-shot generalization; char-level transformers form copy/induction heads readily.
"""
from __future__ import annotations

import json
from pathlib import Path

EOT = "<|endoftext|>"  # corpus sample separator -> mapped to the EOS id


class CharTokenizer:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    _N_SPECIAL = 4

    def __init__(self, special_tokens=None):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    # ── training ──
    def train(self, text: str, vocab_size: int = 256) -> None:
        chars = sorted(set(text.replace(EOT, "")))
        self.char_to_id = {}
        nid = self._N_SPECIAL
        for c in chars:
            self.char_to_id[c] = nid
            nid += 1
        self._rebuild()
        print(f"Training char tokenizer...\n  Vocab size: {self.vocab_size} ({len(self.char_to_id)} chars + 4 special)")

    def _rebuild(self) -> None:
        self.id_to_char = {self.PAD: "", self.UNK: "�", self.BOS: "", self.EOS: ""}
        for c, i in self.char_to_id.items():
            self.id_to_char[i] = c

    # ── interface used by model.generate / honest_pipeline / registry ──
    @property
    def vocab_size(self) -> int:
        return self._N_SPECIAL + len(self.char_to_id)

    @property
    def bos_token_id(self) -> int:
        return self.BOS

    @property
    def eos_token_id(self) -> int:
        return self.EOS

    @property
    def pad_token_id(self) -> int:
        return self.PAD

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(self.BOS)
        parts = text.split(EOT)
        for k, part in enumerate(parts):
            ids.extend(self.char_to_id.get(ch, self.UNK) for ch in part)
            if k < len(parts) - 1:
                ids.append(self.EOS)  # the <|endoftext|> marker
        if add_special:
            ids.append(self.EOS)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if skip_special and i in (self.PAD, self.BOS, self.EOS):
                continue
            out.append(self.id_to_char.get(i, ""))
        return "".join(out)

    # ── persistence (registry.save_model / load_model) ──
    def save(self, path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "char_tokenizer.json").write_text(json.dumps({"char_to_id": self.char_to_id}))

    def load(self, path) -> None:
        d = json.loads((Path(path) / "char_tokenizer.json").read_text())
        self.char_to_id = {k: int(v) for k, v in d["char_to_id"].items()}
        self._rebuild()

# SPDX-License-Identifier: AGPL-3.0-only
"""erosolar inference API — serves the REAL trained appreciation model.

Runs on Cloud Run behind the site's /api/** rewrite. This is genuine neural
inference (the tiny PyTorch transformer's own generate loop), not the in-browser
template fallback. Honesty rules still hold: no capability-class claims; the
model is what it is — a small, honest appreciation generator.
"""
from __future__ import annotations

import os
import re

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from registry import load_model
from honest_pipeline import APPRECIATION_QUALITIES, CURATED_QUALITIES

QUALITIES = list(APPRECIATION_QUALITIES)  # full ~230-quality vocabulary
_QUALITY_SET = set(QUALITIES)


def sanitize_quality(q: str) -> str:
    """Accept any single lowercase word; default to a known quality if empty."""
    q = (q or "").strip().lower().split()
    return q[0] if q else "clarity"

DEVICE = torch.device("cpu")
MODEL, TOK, CFG, INFO = load_model(os.environ.get("MODEL_NAME", "erosolar-v0.01"), device=DEVICE)
MODEL.eval()

app = FastAPI(title="erosolar inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)


def prettify(raw: str) -> str:
    s = re.sub(r"\s+\.", ".", raw)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    s = re.sub(r"\bi\b", "I", s)
    return s


def answer_of(out: str) -> str:
    """Pull the first generated answer (robust to 'Answer:'/'Answer :' decoder spacing)."""
    parts = re.split(r"Answer\s*:", out)
    body = parts[1] if len(parts) > 1 else out
    body = re.split(r"<\|endoftext\|>|Topic\s*:|Question\s*:", body)[0]
    return body.strip()


def run(prompt: str, max_tokens: int, temperature: float) -> str:
    # Wider sampling (top_k 60, top_p 0.97) measurably reduces byte-identical repeats
    # without hurting validity — see the temperature/variety sweep recorded in the README.
    return MODEL.generate(
        TOK, prompt=prompt, max_tokens=max_tokens,
        temperature=float(temperature), top_k=60, top_p=0.97, device=DEVICE,
    )


class AppReq(BaseModel):
    quality: str = "clarity"
    temperature: float = 0.95


class GenReq(BaseModel):
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.95


@app.get("/api/health")
def health():
    return {"status": "ok", "model": INFO.name, "params": INFO.params, "device": str(DEVICE)}


@app.get("/api/qualities")
def qualities():
    return {"qualities": QUALITIES}


@app.post("/api/appreciation")
def appreciation(req: AppReq):
    q = sanitize_quality(req.quality)
    out = run(f"Topic : {q} . Write appreciation . Answer :", 32, req.temperature)
    body = answer_of(out)
    return {"quality": q, "raw": body, "display": prettify(body),
            "model": INFO.name, "source": "neural", "in_vocab": q in _QUALITY_SET}


@app.post("/api/generate")
def generate(req: GenReq):
    out = run(req.prompt, min(int(req.max_tokens), 64), req.temperature)
    return {"prompt": req.prompt, "completion": out, "model": INFO.name}

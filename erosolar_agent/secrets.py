"""Secret/config loading for the erosolar agent runtime.

Secrets live in a gitignored ``.env`` at the repo root (never committed). This
module loads them lazily and exposes typed accessors. Real environment variables
always take precedence over ``.env`` values, so production/CI can inject secrets
without a file present.

Never log a raw secret — use :func:`redact` for any diagnostic output.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"


def _parse_env_file(path: Path) -> dict:
    """Minimal, dependency-free .env parser (KEY=VALUE, # comments, quotes)."""
    values: dict = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


@lru_cache(maxsize=1)
def _dotenv() -> dict:
    return _parse_env_file(_ENV_PATH)


def get_secret(name, default=None, *, required: bool = False):
    """Return a secret by name.

    Resolution order: real environment variable -> .env file -> ``default``.
    Raises ``RuntimeError`` if ``required`` and the value is empty/unset.
    """
    val = os.environ.get(name)
    if val is None or val == "":
        val = _dotenv().get(name)
    if val is None or val == "":
        val = default
    if required and not val:
        raise RuntimeError(
            f"Missing required secret {name!r}. Set it as an environment variable "
            f"or add it to {_ENV_PATH} (see .env.example)."
        )
    return val


def redact(secret) -> str:
    """Render a secret safe for logs (short, irreversible fingerprint)."""
    if not secret:
        return "<unset>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}…{secret[-4:]} (len={len(secret)})"


# -- Convenience accessors -------------------------------------------------

def image_video_gen_api_key(required: bool = False):
    """API key for the image/video generation provider."""
    return get_secret("IMAGE_VIDEO_GEN_API_KEY", required=required)


def image_video_gen_base_url(default=None):
    """Base URL/endpoint for the image/video generation provider."""
    return get_secret("IMAGE_VIDEO_GEN_BASE_URL", default=default)


def lambda_api_key(required: bool = False):
    """Lambda Cloud API key — can launch/terminate GPU instances. Keep it off any
    box/endpoint that doesn't strictly need it."""
    return get_secret("LAMBDA_API_KEY", required=required)


def tavily_api_key():
    return get_secret("TAVILY_API_KEY")


def deepseek_api_key():
    return get_secret("DEEPSEEK_API_KEY")


def hf_token():
    return get_secret("HF_TOKEN")


def wandb_api_key():
    return get_secret("WANDB_API_KEY")

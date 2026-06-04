"""Resolve registry entries to a servable source (path/HF repo).

Reads the repo-root `registry` module (same import style as serve.py), so the
new backend fields added to ModelInfo are available without duplicating state.
"""

from __future__ import annotations


def _registry():
    from registry import get_registry  # repo-root module
    return get_registry()


def get_entry(name: str):
    return _registry().get(name)


def hf_models() -> list:
    return [m for m in _registry().list_models()
            if getattr(m, "backend", "minigpt") == "hf-vllm"]


def resolve_source(name: str) -> str:
    """Return the HF repo id or local path to serve for a registered model."""
    info = get_entry(name)
    if not info:
        raise ValueError(f"model {name!r} not found in registry")
    src = getattr(info, "hf_model_id", "") or getattr(info, "adapter_path", "")
    if not src:
        raise ValueError(
            f"model {name!r} has no hf_model_id/adapter_path — run merge_and_export "
            f"with --register, or set it in models/registry.json"
        )
    return src

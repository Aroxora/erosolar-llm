"""TRL version-compatibility shims.

TRL renamed two things around v1.0 that our SFT/DPO scripts touch:
  * SFTConfig/DPOConfig length field: ``max_seq_length`` (<1.0) -> ``max_length`` (>=1.0)
  * SFT/DPO Trainer tokenizer arg: ``tokenizer=`` (<1.0) -> ``processing_class=`` (>=1.0)

Unsloth pins whichever TRL it's compatible with (which can be either generation),
so we detect the right names at runtime instead of hard-coding one.
"""

from __future__ import annotations

import dataclasses


def _config_fields(cls):
    try:
        return {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        return set()


def make_config(cls, length_value=None, **kwargs):
    """Build a TRL *Config, passing the length under whatever field this version
    exposes and dropping any kwargs this version doesn't recognize.

    Returns (config, field_names)."""
    fields = _config_fields(cls)
    kw = {k: v for k, v in kwargs.items() if (not fields) or k in fields}
    if length_value is not None:
        if "max_length" in fields:
            kw["max_length"] = length_value
        elif "max_seq_length" in fields:
            kw["max_seq_length"] = length_value
        elif not fields:  # introspection failed; assume modern name
            kw["max_length"] = length_value
    return cls(**kw), fields


def make_trainer(cls, tokenizer=None, **kwargs):
    """Construct a TRL Trainer, passing the tokenizer under ``processing_class``
    (TRL>=1.0) or falling back to ``tokenizer`` (older)."""
    if tokenizer is None:
        return cls(**kwargs)
    try:
        return cls(processing_class=tokenizer, **kwargs)
    except TypeError:
        return cls(tokenizer=tokenizer, **kwargs)

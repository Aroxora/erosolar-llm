"""
Simplified configuration for training.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any
import yaml
from datetime import datetime
import hashlib
import json


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.05
    log_every_n_steps: int = 100
    save_every_n_steps: int = 500
    sample_max_tokens: int = 80


@dataclass
class DataConfig:
    """Data configuration."""
    vocab_size: int = 2048
    stride_ratio: float = 0.5
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"


@dataclass
class Config:
    """Master configuration."""
    # Model params (duplicated from model.py for convenience)
    vocab_size: int = 2048
    max_seq_len: int = 128
    embed_dim: int = 192
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 384
    dropout: float = 0.1

    # Infini-attention settings (for infinite context)
    use_infini_attention: bool = False
    segment_size: int = 128  # Local attention window size
    use_delta_rule: bool = True  # Use delta rule for memory updates
    use_rope: bool = False  # Use Rotary Position Embeddings

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment
    name: str = "mini-gpt"
    output_dir: str = "experiments"
    seed: int = 42

    def __post_init__(self):
        self.data.vocab_size = self.vocab_size

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        # Handle nested configs
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        return cls(**data)

    def to_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_hash(self) -> str:
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]

    def get_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{self.get_hash()}_{timestamp}"


# Presets
PRESETS = {
    # Tiny models - ~1-2M params (for quick testing)
    "tiny-1": Config(vocab_size=8000, max_seq_len=128, embed_dim=192, num_heads=4, num_layers=4, ff_dim=384,
                     training=TrainingConfig(epochs=1)),
    "tiny-2": Config(vocab_size=8000, max_seq_len=128, embed_dim=192, num_heads=4, num_layers=4, ff_dim=384,
                     training=TrainingConfig(epochs=2)),

    # Small models - ~10M params
    "small-1": Config(vocab_size=8000, max_seq_len=256, embed_dim=384, num_heads=6, num_layers=6, ff_dim=1024,
                      training=TrainingConfig(epochs=1, batch_size=16)),
    "small-2": Config(vocab_size=8000, max_seq_len=256, embed_dim=384, num_heads=6, num_layers=6, ff_dim=1024,
                      training=TrainingConfig(epochs=2, batch_size=16)),

    # Medium models - ~25-50M params (general purpose)
    "medium-1": Config(vocab_size=12000, max_seq_len=384, embed_dim=512, num_heads=8, num_layers=8, ff_dim=2048,
                       training=TrainingConfig(epochs=1, batch_size=8, learning_rate=3e-4)),
    "medium-2": Config(vocab_size=12000, max_seq_len=384, embed_dim=512, num_heads=8, num_layers=8, ff_dim=2048,
                       training=TrainingConfig(epochs=2, batch_size=8, learning_rate=3e-4)),

    # Medium+ models - ~50M params (enhanced)
    "medium-plus": Config(vocab_size=16000, max_seq_len=512, embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072,
                          training=TrainingConfig(epochs=2, batch_size=4, learning_rate=3e-4)),

    # Large model - ~100M params (requires more memory)
    "large-1": Config(vocab_size=16000, max_seq_len=512, embed_dim=1024, num_heads=16, num_layers=16, ff_dim=4096,
                      training=TrainingConfig(epochs=1, batch_size=2, learning_rate=2e-4)),

    # ============================================================================
    # Infini-attention presets (infinite context with compressive memory)
    # These models can handle arbitrarily long sequences with bounded memory
    # Compatible with: Huawei Ascend NPU, NVIDIA CUDA, Apple MPS, CPU
    # ============================================================================

    # Infini-tiny: ~2M params, 128-token segments, infinite effective context
    "infini-tiny": Config(
        vocab_size=8000, max_seq_len=512, embed_dim=192, num_heads=4, num_layers=4, ff_dim=384,
        use_infini_attention=True, segment_size=128, use_delta_rule=True, use_rope=True,
        training=TrainingConfig(epochs=2, batch_size=16)
    ),

    # Infini-small: ~10M params, 256-token segments
    "infini-small": Config(
        vocab_size=8000, max_seq_len=1024, embed_dim=384, num_heads=6, num_layers=6, ff_dim=1024,
        use_infini_attention=True, segment_size=256, use_delta_rule=True, use_rope=True,
        training=TrainingConfig(epochs=2, batch_size=8, learning_rate=3e-4)
    ),

    # Infini-medium: ~50M params, 512-token segments, up to 1M effective context
    "infini-medium": Config(
        vocab_size=12000, max_seq_len=2048, embed_dim=512, num_heads=8, num_layers=8, ff_dim=2048,
        use_infini_attention=True, segment_size=512, use_delta_rule=True, use_rope=True,
        training=TrainingConfig(epochs=2, batch_size=4, learning_rate=3e-4)
    ),

    # Infini-large: ~100M params, 1024-token segments, designed for very long context
    "infini-large": Config(
        vocab_size=16000, max_seq_len=4096, embed_dim=1024, num_heads=16, num_layers=16, ff_dim=4096,
        use_infini_attention=True, segment_size=1024, use_delta_rule=True, use_rope=True,
        training=TrainingConfig(epochs=1, batch_size=2, learning_rate=2e-4)
    ),

    # Infini-huawei: Optimized for Huawei Ascend NPU (910/310 series)
    # Uses configurations that work well with CANN and torch_npu
    "infini-huawei": Config(
        vocab_size=12000, max_seq_len=2048, embed_dim=512, num_heads=8, num_layers=8, ff_dim=2048,
        use_infini_attention=True, segment_size=512, use_delta_rule=True, use_rope=True,
        training=TrainingConfig(epochs=2, batch_size=8, learning_rate=3e-4)
    ),
}


def get_preset(name: str) -> Config:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]

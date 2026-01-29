"""
Model Registry - Manage trained models with metadata.

Provides:
- Model catalog with names, descriptions, metrics
- Easy listing, loading, and deletion
- Automatic checkpoint discovery
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
import torch

from model import MiniGPT, InfiniGPT, ModelConfig
from tokenizer import BPETokenizer
from config import Config


# Default models directory
MODELS_DIR = Path("models")
REGISTRY_FILE = MODELS_DIR / "registry.json"


@dataclass
class ModelInfo:
    """Metadata for a trained model."""
    name: str
    description: str = ""
    created: str = ""
    preset: str = "tiny"

    # Architecture
    params: int = 0
    vocab_size: int = 2048
    embed_dim: int = 192
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 128

    # Training
    epochs_trained: int = 0
    final_loss: float = 0.0
    training_time_mins: float = 0.0

    # Paths (relative to MODELS_DIR)
    checkpoint_path: str = ""
    tokenizer_path: str = ""
    config_path: str = ""

    # Tags for organization
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        # Handle missing fields gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class ModelRegistry:
    """Registry for managing trained models."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self._registry: Dict[str, ModelInfo] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)
                self._registry = {
                    name: ModelInfo.from_dict(info)
                    for name, info in data.items()
                }
            except (json.JSONDecodeError, KeyError):
                self._registry = {}
        else:
            self._registry = {}

    def _save_registry(self):
        """Save registry to disk."""
        data = {name: info.to_dict() for name, info in self._registry.items()}
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self._registry.values())

    def get(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self._registry.get(name)

    def exists(self, name: str) -> bool:
        """Check if model exists."""
        return name in self._registry

    def register(self, info: ModelInfo) -> bool:
        """Register a new model."""
        if not info.name:
            raise ValueError("Model name is required")

        self._registry[info.name] = info
        self._save_registry()
        return True

    def update(self, name: str, **kwargs) -> bool:
        """Update model metadata."""
        if name not in self._registry:
            return False

        info = self._registry[name]
        for key, value in kwargs.items():
            if hasattr(info, key):
                setattr(info, key, value)

        self._save_registry()
        return True

    def delete(self, name: str, delete_files: bool = True) -> bool:
        """Delete a model from registry and optionally from disk."""
        if name not in self._registry:
            return False

        info = self._registry[name]

        if delete_files:
            # Delete model directory
            model_dir = self.models_dir / name
            if model_dir.exists():
                shutil.rmtree(model_dir)

        del self._registry[name]
        self._save_registry()
        return True

    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the checkpoint path for a model."""
        info = self.get(name)
        if info and info.checkpoint_path:
            return self.models_dir / info.checkpoint_path
        return None

    def get_tokenizer_path(self, name: str) -> Optional[Path]:
        """Get the tokenizer path for a model."""
        info = self.get(name)
        if info and info.tokenizer_path:
            return self.models_dir / info.tokenizer_path
        return None

    def save_model(self, name: str, description: str, model: MiniGPT,
                   tokenizer: BPETokenizer, config: Config,
                   epochs: int = 0, loss: float = 0.0,
                   training_time: float = 0.0, tags: List[str] = None,
                   preset: str = "custom") -> ModelInfo:
        """Save a trained model to the registry."""

        # Create model directory
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        checkpoint_dir = model_dir / "checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'config': config.to_dict(),
        }, checkpoint_dir / "model.pt")

        # Save tokenizer
        tokenizer_dir = model_dir / "tokenizer"
        tokenizer.save(tokenizer_dir)

        # Save config
        config.to_yaml(str(model_dir / "config.yaml"))

        # Create model info
        info = ModelInfo(
            name=name,
            description=description,
            created=datetime.now().isoformat(),
            preset=preset,
            params=model.get_num_params(),
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            epochs_trained=epochs,
            final_loss=loss,
            training_time_mins=training_time,
            checkpoint_path=f"{name}/checkpoint",
            tokenizer_path=f"{name}/tokenizer",
            config_path=f"{name}/config.yaml",
            tags=tags or []
        )

        self.register(info)
        return info

    def load_model(self, name: str, device: torch.device = None) -> tuple:
        """Load a model, tokenizer, and config by name.

        Returns: (model, tokenizer, config, info)
        """
        info = self.get(name)
        if not info:
            raise ValueError(f"Model '{name}' not found")

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        # Load config
        config_path = self.models_dir / info.config_path
        if config_path.exists():
            config = Config.from_yaml(str(config_path))
        else:
            config = Config()

        # Load tokenizer
        tokenizer = BPETokenizer()
        tokenizer_path = self.models_dir / info.tokenizer_path
        tokenizer.load(tokenizer_path)

        # Load model
        # Use ff_dim from config if available, otherwise default to embed_dim * 2
        ff_dim = config.ff_dim if hasattr(config, 'ff_dim') else info.embed_dim * 2

        # Check for Infini-attention flags
        use_infini = getattr(config, 'use_infini_attention', False)
        use_rope = getattr(config, 'use_rope', False)
        segment_size = getattr(config, 'segment_size', 256)
        use_delta_rule = getattr(config, 'use_delta_rule', True)

        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=info.max_seq_len,
            embed_dim=info.embed_dim,
            num_heads=info.num_heads,
            num_layers=info.num_layers,
            ff_dim=ff_dim,
            dropout=0.0,
            use_rope=use_rope,
            use_infini_attention=use_infini,
            segment_size=segment_size,
            use_delta_rule=use_delta_rule
        )

        # Choose model class based on config flags
        if use_infini:
            model = InfiniGPT(model_config).to(device)
        else:
            model = MiniGPT(model_config).to(device)

        # Load weights
        checkpoint_path = self.models_dir / info.checkpoint_path / "model.pt"
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        return model, tokenizer, config, info


# Global registry instance
_registry = None

def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def list_models() -> List[ModelInfo]:
    """List all models in the registry."""
    return get_registry().list_models()


def get_model(name: str) -> Optional[ModelInfo]:
    """Get model info by name."""
    return get_registry().get(name)


def load_model(name: str, device: torch.device = None) -> tuple:
    """Load a model by name."""
    return get_registry().load_model(name, device)


def delete_model(name: str) -> bool:
    """Delete a model."""
    return get_registry().delete(name)

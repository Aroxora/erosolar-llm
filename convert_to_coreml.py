#!/usr/bin/env python3
"""
Convert erosolar PyTorch model to CoreML format for iOS deployment.
Author: Bo Shang <bo@shang.software>
"""

import torch
import coremltools as ct
from pathlib import Path
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MiniGPT, ModelConfig

def convert_to_coreml(model_path: Path, output_path: Path, config_path: Path):
    """Convert PyTorch model to CoreML."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize model config
    model_config = ModelConfig(
        vocab_size=config.get('vocab_size', 32000),
        embed_dim=config.get('embed_dim', 768),
        num_layers=config.get('num_layers', 12),
        num_heads=config.get('num_heads', 12),
        max_seq_len=config.get('max_seq_len', 512),
        ff_dim=config.get('ff_dim', 3072)
    )

    # Initialize model
    model = MiniGPT(model_config)

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Update config from checkpoint if available
        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            model_config = ModelConfig(
                vocab_size=ckpt_config.get('vocab_size', model_config.vocab_size),
                embed_dim=ckpt_config.get('embed_dim', model_config.embed_dim),
                num_layers=ckpt_config.get('num_layers', model_config.num_layers),
                num_heads=ckpt_config.get('num_heads', model_config.num_heads),
                max_seq_len=ckpt_config.get('max_seq_len', model_config.max_seq_len),
                ff_dim=ckpt_config.get('ff_dim', model_config.ff_dim)
            )
            model = MiniGPT(model_config)
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Create example input
    example_input = torch.randint(0, model_config.vocab_size, (1, 128))

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)))],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram"
    )

    # Save
    mlmodel.save(str(output_path))
    print(f"Model saved to {output_path}")

def main():
    model_path = Path("models/erosolar-v0.04/checkpoint/model.pt")
    config_path = Path("models/erosolar-v0.04/config.yaml")
    output_path = Path("ios_app/DeepSeekerChat/DeepSeekerChat/erosolar-v0.04.mlpackage")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    convert_to_coreml(model_path, output_path, config_path)

if __name__ == "__main__":
    main()

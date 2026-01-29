#!/usr/bin/env python3
"""
Export erosolar model weights for iOS deployment.
Saves weights as simple binary format that Swift can load.
Author: Bo Shang <bo@shang.software>
"""

import torch
import json
import numpy as np
from pathlib import Path
import struct

def export_weights(model_path: Path, output_dir: Path):
    """Export model weights to iOS-compatible format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}

    # Save config as JSON
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Export each weight tensor as binary file
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    manifest = {}
    for name, tensor in state_dict.items():
        # Convert to float16 for smaller size
        arr = tensor.cpu().numpy().astype(np.float16)

        # Save shape and data
        safe_name = name.replace('.', '_')
        weight_path = weights_dir / f"{safe_name}.bin"

        with open(weight_path, 'wb') as f:
            # Write shape info
            f.write(struct.pack('I', len(arr.shape)))  # num dimensions
            for dim in arr.shape:
                f.write(struct.pack('I', dim))
            # Write data
            f.write(arr.tobytes())

        manifest[name] = {
            "file": f"weights/{safe_name}.bin",
            "shape": list(arr.shape),
            "dtype": "float16"
        }
        print(f"  Exported {name}: {arr.shape}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    print(f"\nTotal weights: {len(manifest)}")
    print(f"Output directory: {output_dir}")

def main():
    model_path = Path("models/erosolar-v0.04/checkpoint/model.pt")
    output_dir = Path("ios_app/DeepSeekerChat/DeepSeekerChat/Model")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    export_weights(model_path, output_dir)

if __name__ == "__main__":
    main()

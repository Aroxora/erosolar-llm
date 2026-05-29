#!/usr/bin/env python3
"""
Erosolar LLM - Unified Python Module for AGI Training

Features:
- Infini-Attention: O(n) time, O(1) memory transformers
- Master scalar CoT tracking (thinking-only; answers excluded)
- Cloud Run: Angular frontend deployment
- Training: Multi-JSONL data loading from data_store

Installation:
    pip install erosolar-llm              # Core only
    pip install erosolar-llm[full]        # All features
    pip install erosolar-llm[cloud]       # Cloud Run support
    pip install erosolar-llm[huawei]      # Huawei NPU support

CLI Entry Points:
    erosolar           - Main CLI orchestrator
    erosolar-train     - Training script
    erosolar-generate  - Text generation
    erosolar-cloud     - Cloud Run management
    erosolar-cot       - CoT attention calculations
    erosolar-data      - Training data generation

PyPI Upload:
    python -m build
    twine upload dist/*
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="erosolar",
    version="0.1.0",
    author="Bo Shang",
    author_email="bo@shang.software",
    description="erosolar — an honest, small Chain-of-Thought LLM pipeline (Infini-Attention; license-clean training data; no hallucinated performance). Dedicated to Samantha Briasco-Stewart.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aroxora/erosolar-llm",
    project_urls={
        "Documentation": "https://github.com/Aroxora/erosolar-llm#readme",
        "Issues": "https://github.com/Aroxora/erosolar-llm/issues",
    },
    license="AGPL-3.0-only",
    python_requires=">=3.9",
    py_modules=[
        # Core model
        "model",
        "infini_attention",
        "config",
        "tokenizer",
        # Training
        "train_v001",
        "data",
        "auto_attention",
        "mini_cot_optimizer",
        "master_scalar",
        # Data generation
        "generate_all_training_data",
        "generate_coding_only",
        "generate",
        # CLI and orchestration
        "mini_the_agentic_cli",
        # Cloud and deployment
        "cloud_run",
        # Device support
        "huawei_npu",
        # Utilities
        "atomic_save",
        "registry",
        "concepts",
        "training_upgrade_pipeline",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "aiohttp>=3.8.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "full": [
            "openai>=1.0.0",
            "transformers>=4.36.0",
            "datasets>=2.14.0",
            "accelerate>=0.25.0",
            "wandb>=0.16.0",
            "huggingface_hub>=0.19.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "build>=1.0.0",
            "twine>=4.0.0",
        ],
        "cloud": [
            "google-cloud-run>=0.10.0",
            "google-auth>=2.0.0",
        ],
        "huawei": [
            # torch_npu - installed separately for Huawei Ascend NPUs
        ],
    },
    entry_points={
        "console_scripts": [
            # Primary entry point - Mini AI Manager
            "mini-ai-manager=__main__:main",
            # Alternative entry points
            "erosolar=mini_the_agentic_cli:main",
            "erosolar-train=train_v001:main",
            "erosolar-generate=generate:main",
            "erosolar-cloud=cloud_run:main",
            "erosolar-cot=mini_cot_optimizer:main",
            "erosolar-data=generate_all_training_data:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "transformer",
        "attention",
        "infini-attention",
        "linear-attention",
        "long-context",
        "language-model",
        "pytorch",
        "machine-learning",
        "cloud-run",
        "angular",
        "chain-of-thought",
        "master-scalar",
        "deepseek",
    ],
    include_package_data=True,
)

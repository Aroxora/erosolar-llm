"""
Erosolar LLM - Unified Python Module for AGI Training

This package implements:
1. Infini-attention for O(n) time and O(1) memory complexity
2. Master scalar optimization for CoT (thinking-only; answers excluded)
3. Real CoT attention calculations (embedding-based + structural)
4. Training data loading from ALL JSONL files in data_store
5. Cloud Run deployment for Angular frontend
6. Provably correct adversarial verification

Master scalar is the primary optimization target.

Architecture:
- Core: model.py, infini_attention.py, config.py
- Training: train_v001.py, data.py, auto_attention.py
- CoT Attention Calculations: master_scalar.py, mini_cot_optimizer.py
- Data Generation: generate_all_training_data.py, generate_coding_only.py
- Cloud & Frontend: cloud_run.py, angular-chat/
- CLI: mini_the_agentic_cli.py (main orchestrator)

Mathematical Foundation:
- Infini-attention: O(nd² + nsd) time, O(d²) memory
- Standard attention: O(n²d) time, O(nd) memory
- For n > d + s, Infini-attention strictly dominates

Usage:
    import erosolar
    # Pipeline is automatically initialized

    # Access components
    from erosolar import (
        MiniGPT, InfiniGPT,
        Config, get_preset,
        CloudRunManager, CloudRunConfig
    )

    # Deploy to Cloud Run
    from erosolar import deploy_angular
    deploy_angular()
"""

import sys
import warnings
from pathlib import Path

# Ensure the package directory is in path
_package_dir = Path(__file__).parent
if str(_package_dir) not in sys.path:
    sys.path.insert(0, str(_package_dir))

# Version info
__version__ = "0.1.0"
__author__ = "Erosolar AI"

# =============================================================================
# REQUIRED INITIALIZATION: Training Data Upgrade Pipeline
# =============================================================================
# This pipeline is MANDATORY for all training operations.
# It enables iterative improvement through GPT-5.1-codex-mini enhancement.

try:
    from training_upgrade_pipeline import (
        TrainingDataUpgradePipeline,
        PipelineConfig,
        GapTargetingConfig,
        EnhancedExample,
        RepoDataSource,
        GapTask,
        GapProbeResult,
        GPT52Client,
        IterativeTrainer,
        get_pipeline,
        initialize_pipeline as _init_pipeline
    )

    # Initialize pipeline on import (lazy - actual API calls deferred)
    _PIPELINE_INITIALIZED = False

    def _ensure_pipeline():
        """Ensure pipeline is initialized before any training."""
        global _PIPELINE_INITIALIZED
        if not _PIPELINE_INITIALIZED:
            try:
                _init_pipeline()
                _PIPELINE_INITIALIZED = True
            except Exception as e:
                warnings.warn(
                    f"Pipeline initialization deferred (API key may not be set): {e}\n"
                    "Set OPENAI_API_KEY to enable GPT-5.1-codex-mini enhancement."
                )

    # Register initialization hook
    _ensure_pipeline()

except ImportError as e:
    warnings.warn(
        f"Training upgrade pipeline not available: {e}\n"
        "Some features will be limited."
    )
    TrainingDataUpgradePipeline = None
    get_pipeline = lambda: None

# =============================================================================
# Core Model Components
# =============================================================================

try:
    from model import (
        MiniGPT,
        InfiniGPT,
        ModelConfig,
        create_model,
        TransformerBlock,
        CausalSelfAttention,
        RotaryPositionalEmbedding,
        apply_rotary_emb
    )
except ImportError:
    MiniGPT = None
    InfiniGPT = None

# =============================================================================
# Infini-Attention Components
# =============================================================================

try:
    from infini_attention import (
        InfiniAttention,
        InfiniAttentionConfig,
        InfiniTransformerBlock,
        InfiniMemoryManager,
        CompressiveMemory
    )
except ImportError:
    InfiniAttention = None

# =============================================================================
# Configuration
# =============================================================================

try:
    from config import (
        Config,
        TrainingConfig,
        DataConfig,
        get_preset,
        PRESETS
    )
except ImportError:
    Config = None
    get_preset = None

# =============================================================================
# Tokenizer
# =============================================================================

try:
    from tokenizer import (
        BPETokenizer,
        SpecialTokens
    )
except ImportError:
    BPETokenizer = None

# =============================================================================
# Training
# =============================================================================

try:
    from train import (
        Trainer,
        TextDataset,
        create_datasets,
        setup_device
    )
except ImportError:
    Trainer = None

# =============================================================================
# Data
# =============================================================================

try:
    from data import create_training_corpus
except ImportError:
    create_training_corpus = None

# =============================================================================
# Registry
# =============================================================================

try:
    from registry import (
        ModelRegistry,
        ModelInfo,
        get_registry,
        list_models
    )
except ImportError:
    get_registry = None

# =============================================================================
# Device Support (including Huawei NPU)
# =============================================================================

try:
    from huawei_npu import (
        get_device,
        get_device_info,
        elu_plus_one,
        safe_divide,
        DeviceAwareModule
    )
except ImportError:
    get_device = None

# =============================================================================
# Provable Adversarial Training
# =============================================================================

try:
    from provable_adversarial_training import (
        AdversarialVerifier,
        verify_training_example
    )
except ImportError:
    AdversarialVerifier = None

# =============================================================================
# Cloud Run & Angular Frontend Management
# =============================================================================

try:
    from cloud_run import (
        CloudRunManager,
        CloudRunConfig,
    )
except ImportError:
    CloudRunManager = None
    CloudRunConfig = None

# =============================================================================
# CoT Attention Calculations & Training Data Generation (master scalar target)
# =============================================================================
#
# MANDATORY RULE: Master Scalar Pre-Computation
# =============================================
# Before combining ANY deepseek-reasoner generated data with existing training
# data, the master scalar MUST be pre-computed from all existing JSONL files.
#
# This ensures:
# 1. Baseline coherence measurement before new data integration
# 2. Accurate tracking of master scalar improvements per generation
# 3. Quality control - new data should improve, not degrade, the master scalar
#
# Usage:
#   python master_scalar.py --max-samples 500  # Pre-compute before generation
#   python generate_cot_optimization_data.py   # Generate with deepseek-reasoner
#   python master_scalar.py --max-samples 500  # Verify improvement
#
# See generate_cot_optimization_data.py for fully automated generation with
# master scalar tracking.
# =============================================================================

try:
    from mini_cot_optimizer import CoTOptimizer
except ImportError:
    CoTOptimizer = None

try:
    from auto_attention import AutoAttention
except ImportError:
    AutoAttention = None

# =============================================================================
# Master Scalar Pre-Computation (MANDATORY before deepseek-reasoner)
# =============================================================================

try:
    from master_scalar import (
        compute_master_scalar_from_file,
        compute_master_scalar_sync,
        MasterScalarTracker,
        get_tracker,
        extract_cot_texts,
    )
    MASTER_SCALAR_AVAILABLE = True
except ImportError:
    compute_master_scalar_from_file = None
    compute_master_scalar_sync = None
    MasterScalarTracker = None
    get_tracker = None
    extract_cot_texts = None
    MASTER_SCALAR_AVAILABLE = False


def pre_compute_master_scalar(max_samples: int = 500) -> dict:
    """
    MANDATORY: Pre-compute master scalar before combining with deepseek-reasoner data.

    This function MUST be called before any deepseek-reasoner generation to establish
    a baseline and verify improvements after generation.

    Args:
        max_samples: Maximum samples to analyze (for efficiency)

    Returns:
        dict with master_scalar, coherence_scalar, safety_score, sample_count

    Example:
        >>> import erosolar
        >>> baseline = erosolar.pre_compute_master_scalar()
        >>> print(f"Baseline Master Scalar: {baseline['master_scalar']:.4f}")
        >>> # Now run deepseek-reasoner generation
        >>> # After generation, call again to verify improvement
    """
    if not MASTER_SCALAR_AVAILABLE:
        raise ImportError("master_scalar module not available")

    import asyncio

    async def _compute():
        result = await compute_master_scalar_from_file(max_samples=max_samples)
        return {
            "master_scalar": result.master_scalar,
            "coherence_scalar": result.coherence_scalar,
            "safety_score": result.safety_score,
            "raw_dot_product": result.raw_dot_product,
            "sample_count": result.sample_count,
            "sampled_count": result.sampled_count,
        }

    return asyncio.run(_compute())

# =============================================================================
# MINI - The AI Brain (Core Orchestrator)
# =============================================================================
# Mini is the central AI agent that manages:
# - Master scalar tracking (primary optimization target)
# - CoT attention calculations
# - Training data generation orchestration
# - DeepSeek reasoner integration
# - Tool execution and context management

try:
    from mini_the_agentic_cli import (
        MiniAgent,
        MiniShell,
        ToolExecutor,
        ContextWindow,
        SecretsManager,
        Colors,
        # Core functions
        load_erosolar_version,
        save_erosolar_version,
        get_action_count,
        reset_action_counter,
    )
    MINI_AVAILABLE = True
except ImportError as e:
    MiniAgent = None
    MiniShell = None
    ToolExecutor = None
    ContextWindow = None
    SecretsManager = None
    Colors = None
    load_erosolar_version = None
    save_erosolar_version = None
    get_action_count = None
    reset_action_counter = None
    MINI_AVAILABLE = False
    warnings.warn(f"Mini AI brain not available: {e}")

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",

    # Pipeline (REQUIRED)
    "TrainingDataUpgradePipeline",
    "PipelineConfig",
    "GapTargetingConfig",
    "EnhancedExample",
    "RepoDataSource",
    "GapTask",
    "GapProbeResult",
    "GPT52Client",
    "IterativeTrainer",
    "get_pipeline",

    # Models
    "MiniGPT",
    "InfiniGPT",
    "ModelConfig",
    "create_model",

    # Infini-Attention
    "InfiniAttention",
    "InfiniAttentionConfig",
    "InfiniTransformerBlock",
    "InfiniMemoryManager",
    "CompressiveMemory",

    # Configuration
    "Config",
    "TrainingConfig",
    "DataConfig",
    "get_preset",
    "PRESETS",

    # Tokenizer
    "BPETokenizer",
    "SpecialTokens",

    # Training
    "Trainer",
    "TextDataset",
    "create_datasets",
    "setup_device",

    # Data
    "create_training_corpus",

    # Registry
    "ModelRegistry",
    "ModelInfo",
    "get_registry",
    "list_models",

    # Device
    "get_device",
    "get_device_info",

    # Verification
    "AdversarialVerifier",

    # Cloud Run & Frontend
    "CloudRunManager",
    "CloudRunConfig",
    "deploy_angular",
    "get_cloud_status",

    # CoT Attention Calculations
    "CoTOptimizer",
    "AutoAttention",

    # MINI - The AI Brain
    "MiniAgent",
    "MiniShell",
    "ToolExecutor",
    "ContextWindow",
    "SecretsManager",
    "MINI_AVAILABLE",
    "get_mini",
    "run_mini",
    "mini_generate",

    # Core Comparison Functions
    "compare_enhancement_models",
    "run_upgrade_demonstration",
    "info",
]


# =============================================================================
# Global Mini Instance (Singleton)
# =============================================================================

_MINI_INSTANCE = None


def get_mini(working_dir: Path = None) -> "MiniAgent":
    """
    Get the global Mini AI brain instance.

    Mini is the central orchestrator for:
    - Master scalar tracking (primary optimization target)
    - CoT attention calculations
    - Training data generation
    - DeepSeek reasoner API integration

    Args:
        working_dir: Optional working directory override

    Returns:
        MiniAgent instance

    Example:
        >>> import erosolar
        >>> mini = erosolar.get_mini()
        >>> response = await mini.send("Generate training data for coding")
    """
    global _MINI_INSTANCE

    if MiniAgent is None:
        raise ImportError("Mini AI brain not available - check dependencies")

    if _MINI_INSTANCE is None:
        if working_dir:
            _MINI_INSTANCE = MiniAgent(working_dir)
        else:
            _MINI_INSTANCE = MiniAgent()

    return _MINI_INSTANCE


def run_mini():
    """
    Start the Mini interactive shell.

    This launches the full Mini AI brain interface for:
    - Interactive training data generation
    - Master scalar calculations
    - Model training orchestration
    - Cloud Run deployment

    Example:
        >>> import erosolar
        >>> erosolar.run_mini()  # Starts interactive shell
    """
    if MiniShell is None:
        raise ImportError("Mini AI brain not available - check dependencies")

    import asyncio
    shell = MiniShell()
    asyncio.run(shell.run())


async def mini_generate(prompt: str, working_dir: Path = None) -> str:
    """
    Generate a response from Mini AI brain.

    This is a convenience function for single-shot generation.

    Args:
        prompt: The prompt to send to Mini
        working_dir: Optional working directory

    Returns:
        Mini's response string

    Example:
        >>> import erosolar
        >>> import asyncio
        >>> response = asyncio.run(erosolar.mini_generate("What is the current training status?"))
        >>> print(response)
    """
    mini = get_mini(working_dir)
    return await mini.send(prompt)


def info():
    """Print package information and status."""
    print("=" * 60)
    print("Erosolar LLM - Unified Python Module for AGI Training")
    print("=" * 60)
    print(f"Version: {__version__}")
    print()

    # Component status - organized by category
    print("CORE COMPONENTS:")
    core_components = [
        ("Mini AI Brain", MINI_AVAILABLE),
        ("MiniGPT Model", MiniGPT is not None),
        ("InfiniGPT Model", InfiniGPT is not None),
        ("Infini-Attention", InfiniAttention is not None),
    ]
    for name, available in core_components:
        status = "✓" if available else "✗"
        print(f"  {status} {name}")

    print("\nTRAINING & TOOLING (MASTER SCALAR + ATTENTION CALCULATIONS):")
    training_components = [
        ("Training Upgrade Pipeline", TrainingDataUpgradePipeline is not None),
        ("CoT Attention Calculations", CoTOptimizer is not None),
        ("Attention Tooling", AutoAttention is not None),
        ("Trainer", Trainer is not None),
    ]
    for name, available in training_components:
        status = "✓" if available else "✗"
        print(f"  {status} {name}")

    print("\nINFRASTRUCTURE:")
    infra_components = [
        ("Cloud Run Manager", CloudRunManager is not None),
        ("Configuration", Config is not None),
        ("Tokenizer", BPETokenizer is not None),
        ("Registry", get_registry is not None),
        ("Device Support", get_device is not None),
    ]
    for name, available in infra_components:
        status = "✓" if available else "✗"
        print(f"  {status} {name}")

    print()
    print("Mathematical Guarantees:")
    print("  Time Complexity:   O(nd² + nsd) vs O(n²d) standard")
    print("  Memory Complexity: O(d²) vs O(nd) standard")
    print("  Efficiency Gain:   n/(s+d) for n > s+d")
    print()
    print("For detailed math, see README.md")
    print("=" * 60)


# =============================================================================
# Cloud Run & Angular Deployment Functions
# =============================================================================

def deploy_angular(
    project_id: str = "erosolar-prod",
    region: str = "us-central1",
    service_name: str = "erosolar-chat",
    build: bool = True
) -> str:
    """
    Deploy Angular frontend to Cloud Run.

    Args:
        project_id: GCP project ID
        region: Cloud Run region
        service_name: Cloud Run service name
        build: Whether to build Angular first

    Returns:
        Service URL if successful

    Example:
        >>> import erosolar
        >>> url = erosolar.deploy_angular()
        >>> print(f"Deployed to: {url}")
    """
    if CloudRunManager is None:
        raise ImportError("cloud_run module not available")

    config = CloudRunConfig(
        project_id=project_id,
        region=region,
        service_name=service_name
    )
    manager = CloudRunManager(config)
    return manager.deploy_angular(build=build)


def get_cloud_status(
    project_id: str = "erosolar-prod",
    region: str = "us-central1",
    service_name: str = "erosolar-chat"
) -> dict:
    """
    Get Cloud Run service status.

    Args:
        project_id: GCP project ID
        region: Cloud Run region
        service_name: Cloud Run service name

    Returns:
        Dict with service status information
    """
    if CloudRunManager is None:
        return {"error": "cloud_run module not available"}

    config = CloudRunConfig(
        project_id=project_id,
        region=region,
        service_name=service_name
    )
    manager = CloudRunManager(config)
    return manager.get_service_status()


# =============================================================================
# Enhancement Model Comparison (CORE FUNCTIONALITY)
# =============================================================================

def compare_enhancement_models(
    training_data: list = None,
    models_to_test: list = None,
    epochs: int = 200,
    verbose: bool = True
):
    """
    Compare how different GPT enhancement models affect training quality.

    This is the CORE PURPOSE of this module: demonstrating that training on
    GPT-5.1-codex-mini enhanced data produces better models than training on original data.

    Args:
        training_data: List of (prompt, response) tuples. Uses default ML examples if None.
        models_to_test: List of model names to compare. Default: ["gpt-5.1-codex-mini", "gpt-5.1"]
        epochs: Training epochs per model
        verbose: Print detailed progress

    Returns:
        dict: {model_name: {"loss": float, "generation": str, "enhanced_data": list}}

    Example:
        >>> import erosolar
        >>> results = erosolar.compare_enhancement_models()
        >>> print(results["gpt-5.1-codex-mini"]["loss"])  # Should be lower than base
    """
    import torch

    if training_data is None:
        training_data = [
            ("What is machine learning?", "ML is when computers learn from data."),
            ("Explain neural networks", "Neural networks are layers of nodes."),
            ("What is backpropagation?", "Its how neural nets learn."),
            ("Define gradient descent", "An optimization method."),
            ("What is attention?", "Attention lets models focus on parts."),
        ]

    if models_to_test is None:
        models_to_test = ["gpt-5.1-codex-mini", "gpt-5.1"]

    def train_model(data, epochs_inner):
        """Train InfiniGPT on given data."""
        config = ModelConfig(
            vocab_size=500, embed_dim=64, num_layers=2, num_heads=4,
            ff_dim=128, segment_size=16, max_seq_len=64, use_infini_attention=True
        )

        tokenizer = BPETokenizer()
        all_text = " ".join([p + " " + r for p, r in data])
        tokenizer.train(all_text, vocab_size=500)

        model = InfiniGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        pairs = []
        for prompt, response in data:
            tokens = tokenizer.encode(f"{prompt} {response}")[:64]
            if len(tokens) >= 4:
                pairs.append(tokens)

        model.train()
        final_loss = 0.0
        for _ in range(epochs_inner):
            epoch_loss = 0.0
            for token_list in pairs:
                optimizer.zero_grad()
                model.reset_memory()
                tokens = torch.tensor([token_list], dtype=torch.long)
                out = model(tokens[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    out.view(-1, config.vocab_size), tokens[:, 1:].view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(pairs)

        return model, tokenizer, final_loss

    def generate(model, tok, prompt, n=20):
        """Generate from model."""
        model.eval()
        model.reset_memory()
        ids = torch.tensor([tok.encode(prompt)])
        with torch.no_grad():
            for _ in range(n):
                out = model(ids)
                probs = torch.softmax(out[0, -1] / 0.8, dim=-1)
                ids = torch.cat([ids, torch.multinomial(probs, 1).unsqueeze(0)], dim=1)
        return tok.decode(ids[0].tolist())

    results = {}

    # Train base model
    if verbose:
        print("\n" + "="*70)
        print("ENHANCEMENT MODEL COMPARISON")
        print("="*70)
        print("\n[BASE] Training on original data...")

    base_m, base_t, base_l = train_model(training_data, epochs)
    base_gen = generate(base_m, base_t, "What is")
    results["base"] = {
        "loss": base_l,
        "generation": base_gen,
        "enhanced_data": training_data
    }

    if verbose:
        print(f"    Loss: {base_l:.4f}")
        print(f"    Generation: {base_gen[:60]}")

    # Test each enhancement model
    for model_name in models_to_test:
        if verbose:
            print(f"\n[{model_name.upper()}] Enhancing with {model_name}...")

        try:
            client = GPT52Client(PipelineConfig(api_model=model_name, backup_models=()))
            enhanced_data = []

            for p, o in training_data:
                e, t, s = client.enhance_response(p, o)
                enhanced_data.append((p, e))
                if verbose:
                    print(f"    {p[:25]}... -> {t} (score: {s:.2f})")

            if verbose:
                print(f"\n    Training {model_name}-enhanced model...")

            m, t, l = train_model(enhanced_data, epochs)
            gen_output = generate(m, t, "What is")

            results[model_name] = {
                "loss": l,
                "generation": gen_output,
                "enhanced_data": enhanced_data
            }

            if verbose:
                print(f"    Loss: {l:.4f}")
                print(f"    Generation: {gen_output[:60]}")

        except Exception as e:
            if verbose:
                print(f"    ERROR: {str(e)[:50]}")
            results[model_name] = {"error": str(e)}

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("REAL RESULTS SUMMARY")
        print("="*70)
        print(f"| Model         | Loss   | Improvement |")
        print(f"|---------------|--------|-------------|")
        for name, data in results.items():
            if "error" not in data:
                improvement = ((base_l - data["loss"]) / base_l * 100) if name != "base" else 0
                print(f"| {name:13s} | {data['loss']:.4f} | {improvement:+.1f}%        |")
        print("="*70)

    return results


def run_upgrade_demonstration(cycles: int = 2, verbose: bool = True):
    """
    Run a full demonstration of the upgrade pipeline.

    This demonstrates the core thesis: iteratively enhancing training data
    with GPT-5.1-codex-mini produces progressively better models.

    Args:
        cycles: Number of enhance-train cycles
        verbose: Print detailed output

    Returns:
        list: Results from each cycle
    """
    results = []
    current_data = [
        ("What is ML?", "Computers learning from data."),
        ("Explain AI", "Artificial intelligence."),
        ("What is deep learning?", "Neural nets with many layers."),
    ]

    if verbose:
        print("\n" + "="*70)
        print(f"UPGRADE PIPELINE DEMONSTRATION ({cycles} cycles)")
        print("="*70)

    for cycle in range(cycles):
        if verbose:
            print(f"\n--- Cycle {cycle + 1} ---")

        cycle_result = compare_enhancement_models(
            training_data=current_data,
            models_to_test=["gpt-5.1-codex-mini"],
            epochs=100,
            verbose=verbose
        )

        results.append(cycle_result)

        # Use enhanced data for next cycle
        if "gpt-5.1-codex-mini" in cycle_result and "enhanced_data" in cycle_result["gpt-5.1-codex-mini"]:
            current_data = cycle_result["gpt-5.1-codex-mini"]["enhanced_data"]

    return results


# =============================================================================
# DEEPSEEK API KEY CHECK & INITIALIZATION
# =============================================================================

import os

def _check_deepseek_api_key() -> bool:
    """Check if DEEPSEEK_API_KEY is available."""
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _prompt_for_api_key() -> str:
    """Prompt user for DEEPSEEK_API_KEY if not set."""
    import getpass
    print("\n" + "=" * 60)
    print("DEEPSEEK API KEY REQUIRED")
    print("=" * 60)
    print("Mini AI Brain requires a DeepSeek API key to function.")
    print("Get your key at: https://platform.deepseek.com/")
    print()
    key = getpass.getpass("Enter DEEPSEEK_API_KEY: ").strip()
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
        print("API key set for this session.")
        print("To persist, add to ~/.bashrc: export DEEPSEEK_API_KEY='your-key'")
    return key


def ensure_api_key() -> bool:
    """Ensure DEEPSEEK_API_KEY is available, prompting if needed."""
    if _check_deepseek_api_key():
        return True
    key = _prompt_for_api_key()
    return bool(key)


# Print initialization message
_api_status = "✓ READY" if _check_deepseek_api_key() else "✗ KEY NEEDED"
print(f"Erosolar LLM v{__version__} initialized")
print(f"Mini AI Brain: {'AVAILABLE' if MINI_AVAILABLE else 'NOT AVAILABLE'} | API: {_api_status}")
print("Launch: mini-ai-manager | python -m erosolar")

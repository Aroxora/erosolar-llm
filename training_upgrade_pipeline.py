"""
Training Data Upgrade Pipeline

This module implements an iterative training data enhancement loop:
1. Trained model generates candidate responses
2. GPT-5.1-codex-mini (via OpenAI Responses API) enhances/corrects the data
3. Enhanced data trains the next generation model

The pipeline creates a self-improving cycle where each model generation
is trained on data verified and enhanced by a more capable model.

Mathematical Foundation:
- Let M_n be model at iteration n with capability C(M_n)
- Let G be GPT-5.1-codex-mini with capability C(G) >> C(M_0)
- Training data D_n = Enhance(Generate(M_{n-1}), G)
- Then: C(M_n) >= C(M_{n-1}) with high probability

This is required at initialization to ensure all training benefits from
the upgrade pipeline infrastructure.
"""

import os
import sys
import json
import hashlib
import time
import importlib
import inspect
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime
import random
import threading


class ProgressTracker:
    """Real-time progress tracker for GPT-5.1-codex-mini upgrades."""

    def __init__(self):
        self.total_upgrades = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.current_batch = 0
        self.current_batch_total = 0
        self.start_time = None
        self._lock = threading.Lock()

    def start(self):
        self.start_time = time.time()

    def increment(self, accepted: bool = True):
        with self._lock:
            self.total_upgrades += 1
            if accepted:
                self.total_accepted += 1
            else:
                self.total_rejected += 1
            self._print_status()

    def set_batch(self, current: int, total: int):
        with self._lock:
            self.current_batch = current
            self.current_batch_total = total

    def _print_status(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.total_upgrades / elapsed if elapsed > 0 else 0

        # Build status line
        status = (
            f"\r  \033[96m[GPT-5.1-codex-mini UPGRADES]\033[0m "
            f"\033[92m✓ {self.total_upgrades}\033[0m total | "
            f"\033[92m{self.total_accepted}\033[0m accepted | "
            f"\033[93m{self.total_rejected}\033[0m rejected | "
            f"\033[94m{rate:.1f}/s\033[0m"
        )

        if self.current_batch_total > 0:
            status += f" | batch {self.current_batch}/{self.current_batch_total}"

        # Clear line and print
        sys.stdout.write("\033[K" + status)
        sys.stdout.flush()

    def finish(self):
        print()  # Newline after progress


# Global progress tracker
_progress_tracker = ProgressTracker()

# OpenAI Responses API client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

@dataclass
class PipelineConfig:
    """Configuration for the training data upgrade pipeline."""
    # Model settings - tries in order until one works
    api_model: str = "gpt-5.1-codex-mini"  # Primary model (OpenAI Responses API)
    backup_models: tuple = (
        "gpt-5.1-mini",      # GPT-5.1 mini fallback
        "gpt-5-nano",        # Very cheap fallback
        "gpt-4o",            # GPT-4o - strong fallback
        "gpt-4o-mini",       # GPT-4o mini - cost-effective
        "gpt-4-turbo",       # GPT-4 turbo - reliable
    )
    api_base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7

    # Pipeline settings
    batch_size: int = 10
    enhancement_ratio: float = 0.3  # Fraction of data to enhance
    verification_threshold: float = 0.8  # Minimum quality score
    cache_dir: str = "cache/pipeline"

    # Generation settings
    prompts_per_category: int = 50
    max_response_length: int = 512

    # Quality control
    require_verification: bool = True
    log_rejections: bool = True
    rejection_log_file: str = "cache/pipeline/rejections.jsonl"

    # Repository data integration
    repo_data_allow_downloads: bool = False
    repo_data_dedupe: bool = True
    repo_data_shuffle: bool = True
    repo_data_max_pairs: Optional[int] = None
    repo_data_optimize: bool = False

    # Persistence
    persist_dir: str = "cache/pipeline/persist"
    persist_config: bool = True
    persist_stats: bool = True
    persist_runs: bool = True
    persist_generated: bool = True
    persist_enhanced_data: bool = True
    persist_enhanced_batches: bool = True
    load_persisted_state: bool = True

    # Full source gating (for review-only builds)
    require_full_source: bool = False
    full_source_env_var: str = "EROSOLAR_FULL_SOURCE"
    full_source_marker: Optional[str] = None

    # Gap-targeting (focus on GPT-5.1-codex-mini weakness probes)
    gap_targeting: "GapTargetingConfig" = field(default_factory=lambda: GapTargetingConfig())


@dataclass
class GapTargetingConfig:
    """Configuration for gap-targeted task generation and probing."""
    enabled: bool = True
    seed: int = 42
    categories: Tuple[str, ...] = (
        "format_strict",
        "multi_step_math",
        "string_transform",
        "table_reasoning",
        "tool_schema"
    )
    max_tasks: int = 200
    max_failures: int = 100
    require_probe: bool = True
    general_ratio: float = 0.2
    probe_temperature: float = 0.2
    probe_max_tokens: int = 512
    probe_model: Optional[str] = None
    persist_probe_results: bool = True


@dataclass
class EnhancedExample:
    """A training example that has been enhanced by GPT-5.1-codex-mini."""
    original_prompt: str
    original_response: str
    enhanced_response: str
    enhancement_type: str  # "correction", "expansion", "simplification", "verified"
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_training_pair(self) -> Tuple[str, str]:
        """Convert to (prompt, response) training pair."""
        return (self.original_prompt, self.enhanced_response)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedExample":
        return cls(**data)


@dataclass
class RepoDataSource:
    """Repository data source metadata."""
    name: str
    loader: Callable[[bool], List[Tuple[str, str]]]
    description: str = ""
    requires_network: bool = False


@dataclass
class GapTask:
    """A task with an oracle response and validator."""
    task_id: str
    prompt: str
    oracle_response: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GapProbeResult:
    """Result of probing GPT-5.1-codex-mini on a gap task."""
    task_id: str
    category: str
    prompt: str
    oracle_response: str
    model_response: str
    passed: bool
    failure_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GPT52Client:
    """
    Client for GPT-5.1-codex-mini Responses API.

    Uses OpenAI's Responses API format for structured enhancement requests.
    Falls back to standard completions if Responses API unavailable.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None
        self._initialized = False

    def _ensure_client(self):
        """Lazy initialization of OpenAI client."""
        if self._initialized:
            return

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your API key from https://platform.openai.com/api-keys"
            )

        kwargs = {"api_key": api_key}
        if self.config.api_base_url:
            kwargs["base_url"] = self.config.api_base_url

        self._client = OpenAI(**kwargs)
        self._initialized = True

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None
    ) -> str:
        """Generate a response using OpenAI Responses API."""
        self._ensure_client()

        model = model_override or self.config.api_model
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens

        # Build input for Responses API
        input_content = prompt
        if system_prompt:
            input_content = f"{system_prompt}\n\n{prompt}"

        response = self._client.responses.create(
            model=model,
            input=input_content,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        return response.output_text.strip()

    def enhance_response(
        self,
        prompt: str,
        original_response: str,
        enhancement_instructions: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Enhance a response using GPT-5.1-codex-mini with fallback to backup models.

        Args:
            prompt: The original user prompt
            original_response: The model's original response
            enhancement_instructions: Optional specific enhancement guidance

        Returns:
            Tuple of (enhanced_response, enhancement_type, quality_score)
        """
        self._ensure_client()

        system_prompt = """You are an expert training data curator. Your task is to enhance
and verify AI assistant responses for training a new language model.

For each response, you should:
1. Fix any factual errors
2. Improve clarity and completeness
3. Ensure the response is helpful and accurate
4. Rate the quality of the enhanced response (0.0-1.0)

Respond in JSON format:
{
    "enhanced_response": "your improved response",
    "enhancement_type": "correction|expansion|simplification|verified",
    "quality_score": 0.95,
    "changes_made": "brief description of changes"
}

If the original response is already excellent, use "verified" as the enhancement_type
and return it unchanged with a high quality score."""

        user_message = f"""Original prompt: {prompt}

Original response: {original_response}

{enhancement_instructions or "Please enhance this response for training data."}"""

        # Try primary model, then fallbacks
        models_to_try = [self.config.api_model] + list(self.config.backup_models)
        last_error = None

        full_prompt = f"{system_prompt}\n\n{user_message}"

        for model in models_to_try:
            try:
                response = self._client.responses.create(
                    model=model,
                    input=full_prompt,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    text={"format": {"type": "json_object"}}
                )
                result = json.loads(response.output_text)
                if model != self.config.api_model:
                    print(f"  [Fallback to {model} succeeded]")
                return (
                    result.get("enhanced_response", original_response),
                    result.get("enhancement_type", "unknown"),
                    float(result.get("quality_score", 0.5))
                )

            except Exception as e:
                error_str = str(e)
                last_error = e
                # Check for 401 (auth) or 404 (model not found) errors
                if "401" in error_str or "invalid_api_key" in error_str:
                    print(f"  [Auth failed for {model}, trying backup...]")
                    continue
                elif "404" in error_str or "model_not_found" in error_str:
                    print(f"  [Model {model} not found, trying backup...]")
                    continue
                else:
                    # Other error, don't retry
                    print(f"Enhancement failed ({model}): {e}")
                    break

        # All models failed
        print(f"All models failed. Last error: {last_error}")
        return (original_response, "failed", 0.3)

    def batch_enhance(
        self,
        examples: List[Tuple[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[EnhancedExample]:
        """
        Enhance a batch of examples using MAXIMAL threading.

        Args:
            examples: List of (prompt, response) tuples
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of EnhancedExample objects
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(examples)
        if total == 0:
            return []

        results = [None] * total  # Pre-allocate for ordered results
        completed_count = [0]  # Use list for mutable counter in closure
        progress_lock = threading.Lock()

        # Start progress tracking
        _progress_tracker.start()
        _progress_tracker.set_batch(0, total)

        def enhance_single(idx_and_example):
            idx, (prompt, response) = idx_and_example
            try:
                enhanced, etype, score = self.enhance_response(prompt, response)
                result = EnhancedExample(
                    original_prompt=prompt,
                    original_response=response,
                    enhanced_response=enhanced,
                    enhancement_type=etype,
                    quality_score=score
                )
                accepted = score >= self.config.verification_threshold
            except Exception as e:
                result = EnhancedExample(
                    original_prompt=prompt,
                    original_response=response,
                    enhanced_response=response,
                    enhancement_type="failed",
                    quality_score=0.3
                )
                accepted = False
            return idx, result, accepted

        # MAXIMAL THREADING: Up to 100 concurrent API calls
        max_workers = min(100, total)
        print(f"\n  \033[96m[GPT-5.1-codex-mini Enhance]\033[0m Starting {total} upgrades with {max_workers} threads...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(enhance_single, (i, ex)): i for i, ex in enumerate(examples)}

            for future in as_completed(futures):
                try:
                    idx, result, accepted = future.result()
                    results[idx] = result

                    with progress_lock:
                        completed_count[0] += 1
                        _progress_tracker.set_batch(completed_count[0], total)
                        _progress_tracker.increment(accepted=accepted)

                        if progress_callback:
                            progress_callback(completed_count[0], total)
                except Exception as e:
                    _progress_tracker.increment(accepted=False)

        _progress_tracker.finish()
        print(f"  \033[92m[GPT-5.1-codex-mini Enhance]\033[0m Complete: {_progress_tracker.total_accepted} accepted, {_progress_tracker.total_rejected} rejected")

        # Filter out any None results (shouldn't happen but safety check)
        return [r for r in results if r is not None]

    def generate_training_prompt(self, category: str) -> str:
        """Generate a new training prompt using GPT-5.1-codex-mini Responses API."""
        self._ensure_client()

        instruction = f"""Generate a diverse, educational prompt for the category: {category}

The prompt should be:
- Clear and unambiguous
- Appropriate for training an AI assistant
- Varied in complexity and style
- Not repetitive with common examples

Return ONLY the prompt text, nothing else."""

        response = self._client.responses.create(
            model=self.config.api_model,
            input=instruction,
            temperature=0.9,
            max_output_tokens=200
        )
        return response.output_text.strip()


def _supports_param(func: Callable[..., Any], name: str) -> bool:
    """Check whether a callable supports a specific parameter name."""
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _lazy_loader(
    module_name: str,
    func_name: str,
    extra_kwargs: Optional[Dict[str, Any]] = None
) -> Callable[[bool], List[Tuple[str, str]]]:
    """Create a lazy loader for repo data sources."""
    def _loader(allow_download: bool) -> List[Tuple[str, str]]:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        kwargs = dict(extra_kwargs or {})

        if _supports_param(func, "allow_download"):
            kwargs["allow_download"] = allow_download
        elif _supports_param(func, "allow_downloads"):
            kwargs["allow_downloads"] = allow_download
        elif _supports_param(func, "allow_network"):
            kwargs["allow_network"] = allow_download

        return func(**kwargs)

    return _loader


class TrainingDataUpgradePipeline:
    """
    Main pipeline for upgrading training data through model generation + GPT-5.1-codex-mini enhancement.

    This creates a virtuous cycle:
    1. Current model generates candidate responses
    2. GPT-5.1-codex-mini enhances and verifies the responses
    3. Enhanced data trains the next model iteration

    Each iteration produces higher quality training data, leading to
    capability improvements bounded only by GPT-5.1-codex-mini's verification accuracy.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.gpt_client = GPT52Client(self.config)
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = Path(self.config.persist_dir or self.config.cache_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Repository data sources (registered lazily)
        self.repo_data_sources: Dict[str, RepoDataSource] = {}
        self._repo_sources_registered = False

        # Statistics
        self.stats = {
            "total_generated": 0,
            "total_enhanced": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "enhancement_types": {},
            "avg_quality_score": 0.0
        }

        if self.config.load_persisted_state:
            self._load_persisted_state()
        if self.config.persist_config:
            self._persist_config()

    def _ensure_repo_sources_registered(self):
        """Register repository data sources once."""
        if self._repo_sources_registered:
            return
        _register_repo_data_sources(self)
        self._repo_sources_registered = True

    def _persist_path(self, filename: str) -> Path:
        return self.persist_dir / filename

    def _write_json(self, path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        tmp_path.replace(path)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _persist_config(self):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config)
        }
        self._write_json(self._persist_path("config.json"), payload)

    def _persist_stats(self):
        if not self.config.persist_stats:
            return
        payload = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats
        }
        self._write_json(self._persist_path("stats.json"), payload)

    def _persist_run_event(self, event: str, payload: Optional[Dict[str, Any]] = None):
        if not self.config.persist_runs:
            return
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "payload": payload or {}
        }
        self._append_jsonl(self._persist_path("runs.jsonl"), record)

    def _load_persisted_state(self):
        stats_path = self._persist_path("stats.json")
        if not stats_path.exists():
            return
        try:
            with open(stats_path, "r") as f:
                payload = json.load(f)
            loaded = payload.get("stats", {})
            if not isinstance(loaded, dict):
                return
            # Preserve existing keys and update with persisted values
            for key in ("total_generated", "total_enhanced", "total_accepted",
                        "total_rejected", "avg_quality_score"):
                if key in loaded:
                    self.stats[key] = loaded[key]
            if isinstance(loaded.get("enhancement_types"), dict):
                self.stats["enhancement_types"] = loaded["enhancement_types"]
        except Exception as e:
            warnings.warn(f"Failed to load persisted stats: {e}")

    def _persist_generated_batch(
        self,
        generated: List[Tuple[str, str]],
        iteration: Optional[int] = None,
        label: str = "generated"
    ):
        if not self.config.persist_generated:
            return
        filename = f"{label}_iter_{iteration}.jsonl" if iteration is not None else f"{label}_batches.jsonl"
        path = self._persist_path(filename)
        timestamp = datetime.now().isoformat()
        for prompt, response in generated:
            self._append_jsonl(path, {
                "timestamp": timestamp,
                "iteration": iteration,
                "prompt": prompt,
                "response": response
            })

    def _persist_enhanced_batch(
        self,
        enhanced: List[EnhancedExample],
        batch_id: str,
        source: str,
        iteration: Optional[int] = None
    ):
        if not self.config.persist_enhanced_batches:
            return
        path = self._persist_path("enhanced_batches.jsonl")
        for ex in enhanced:
            payload = ex.to_dict()
            payload["batch_id"] = batch_id
            payload["source"] = source
            payload["iteration"] = iteration
            self._append_jsonl(path, payload)

    def _persist_repo_manifest(self, manifest: Dict[str, Any]):
        if not self.config.persist_runs:
            return
        self._append_jsonl(self._persist_path("repo_manifests.jsonl"), manifest)

    def _is_full_source_available(self) -> bool:
        env_val = os.environ.get(self.config.full_source_env_var)
        if env_val and env_val.strip().lower() in ("1", "true", "yes", "on"):
            return True
        marker = self.config.full_source_marker
        if marker:
            marker_path = Path(marker)
            if not marker_path.is_absolute():
                marker_path = Path(__file__).resolve().parent / marker_path
            if marker_path.exists():
                return True
        return False

    def persist_iteration_history(self, history: List[Dict[str, Any]]):
        """Persist iterative trainer history for review."""
        if not self.config.persist_runs:
            return
        payload = {
            "timestamp": datetime.now().isoformat(),
            "history": history
        }
        self._write_json(self._persist_path("iteration_history.json"), payload)

    def register_repo_data_source(
        self,
        name: str,
        loader: Callable[[bool], List[Tuple[str, str]]],
        description: str = "",
        requires_network: bool = False
    ) -> bool:
        """Register a repository data source."""
        if name in self.repo_data_sources:
            return False
        self.repo_data_sources[name] = RepoDataSource(
            name=name,
            loader=loader,
            description=description,
            requires_network=requires_network
        )
        return True

    def list_repo_data_sources(self) -> List[RepoDataSource]:
        """List registered repository data sources."""
        self._ensure_repo_sources_registered()
        return list(self.repo_data_sources.values())

    def _dedupe_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Remove exact duplicate (prompt, response) pairs while preserving order."""
        seen = set()
        deduped = []
        for prompt, response in pairs:
            key = (prompt, response)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((prompt, response))
        return deduped

    def _optimize_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Basic in-pipeline optimization (trim, filter empties)."""
        cleaned = []
        for prompt, response in pairs:
            prompt = (prompt or "").strip()
            response = (response or "").strip()
            if not prompt or not response:
                continue
            cleaned.append((prompt, response))
        return cleaned

    @staticmethod
    def _format_training_pairs(pairs: List[Tuple[str, str]]) -> str:
        """Format pairs into a corpus with special tokens for turn boundaries."""
        formatted = [f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n{response}\n<|end_turn|>" for prompt, response in pairs]
        return "\n\n\n".join(formatted)

    @staticmethod
    def _write_pairs_jsonl(pairs: List[Tuple[str, str]], output_path: str):
        """Write (prompt, response) pairs to JSONL messages format."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for prompt, response in pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def load_repository_data(
        self,
        allow_downloads: Optional[bool] = None,
        max_pairs: Optional[int] = None,
        dedupe: Optional[bool] = None,
        shuffle: Optional[bool] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        optimize: Optional[bool] = None
    ) -> List[Tuple[str, str]]:
        """Load training pairs from all registered repository data sources.

        Honors download gating, optional full-source requirements, and persists
        a manifest of source counts and skip reasons.
        """
        self._ensure_repo_sources_registered()

        allow_downloads = (
            self.config.repo_data_allow_downloads
            if allow_downloads is None else allow_downloads
        )
        max_pairs = self.config.repo_data_max_pairs if max_pairs is None else max_pairs
        dedupe = self.config.repo_data_dedupe if dedupe is None else dedupe
        shuffle = self.config.repo_data_shuffle if shuffle is None else shuffle
        optimize = self.config.repo_data_optimize if optimize is None else optimize

        if self.config.require_full_source and not self._is_full_source_available():
            reason = (
                "Full source required but marker/env not found. "
                "Set EROSOLAR_FULL_SOURCE=1 or provide full_source_marker."
            )
            warnings.warn(reason)
            self._persist_repo_manifest({
                "timestamp": datetime.now().isoformat(),
                "status": "skipped",
                "reason": reason,
                "allow_downloads": allow_downloads,
                "sources": []
            })
            return []

        pairs: List[Tuple[str, str]] = []
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "allow_downloads": allow_downloads,
            "sources": [],
            "total_pairs_before": 0,
            "total_pairs_after": 0,
            "dedupe": dedupe,
            "shuffle": shuffle,
            "optimize": optimize,
            "max_pairs": max_pairs
        }
        print(f"  Loading {len(self.repo_data_sources)} data sources...")
        for idx, source in enumerate(self.repo_data_sources.values()):
            if include_sources and source.name not in include_sources:
                print(f"    [{idx+1}] {source.name}: skipped (not in include list)")
                manifest["sources"].append({
                    "name": source.name,
                    "status": "skipped",
                    "reason": "not_in_include_sources"
                })
                continue
            if exclude_sources and source.name in exclude_sources:
                print(f"    [{idx+1}] {source.name}: skipped (excluded)")
                manifest["sources"].append({
                    "name": source.name,
                    "status": "skipped",
                    "reason": "excluded"
                })
                continue
            if source.requires_network and not allow_downloads:
                print(f"    [{idx+1}] {source.name}: skipped (network disabled)")
                manifest["sources"].append({
                    "name": source.name,
                    "status": "skipped",
                    "reason": "network_disabled"
                })
                continue
            try:
                print(f"    [{idx+1}] {source.name}: loading...", end="", flush=True)
                source_pairs = source.loader(allow_downloads)
                pairs.extend(source_pairs)
                print(f" \033[92m{len(source_pairs)} pairs\033[0m")
                manifest["sources"].append({
                    "name": source.name,
                    "status": "loaded",
                    "pairs": len(source_pairs)
                })
            except Exception as e:
                print(f" \033[91mERROR: {e}\033[0m")
                manifest["sources"].append({
                    "name": source.name,
                    "status": "error",
                    "error": str(e)
                })
                warnings.warn(f"Repo data source '{source.name}' failed: {e}")

        manifest["total_pairs_before"] = len(pairs)
        print(f"  Total raw pairs: {len(pairs)}")
        if optimize:
            print(f"  Optimizing pairs...", end="", flush=True)
            pairs = self._optimize_pairs(pairs)
            print(f" \033[92m{len(pairs)} pairs\033[0m")
        if dedupe:
            print(f"  Deduplicating...", end="", flush=True)
            before = len(pairs)
            pairs = self._dedupe_pairs(pairs)
            print(f" \033[92m{before} -> {len(pairs)} pairs\033[0m")
        if shuffle:
            print(f"  Shuffling {len(pairs)} pairs...")
            random.shuffle(pairs)
        if max_pairs:
            print(f"  Limiting to {max_pairs} pairs...")
            pairs = pairs[:max_pairs]

        manifest["total_pairs_after"] = len(pairs)
        print(f"  \033[92mFinal: {len(pairs)} training pairs ready\033[0m")
        self._persist_repo_manifest(manifest)

        return pairs

    def export_repository_corpus(
        self,
        output_file: Optional[str] = None,
        **load_kwargs: Any
    ) -> str:
        """Load repository data and export as a training corpus."""
        pairs = self.load_repository_data(**load_kwargs)
        corpus = self._format_training_pairs(pairs)
        if output_file:
            with open(output_file, "w") as f:
                f.write(corpus)
        self._persist_run_event("export_repository_corpus", {
            "pairs": len(pairs),
            "output_file": output_file
        })
        return corpus

    def _generate_gap_tasks(self) -> List[GapTask]:
        """Generate candidate gap-targeted tasks."""
        gap_cfg = self.config.gap_targeting
        if not gap_cfg.enabled:
            return []

        rng = random.Random(gap_cfg.seed)
        tasks: List[GapTask] = []
        category_generators = {
            "format_strict": self._generate_format_strict_tasks,
            "multi_step_math": self._generate_multi_step_math_tasks,
            "string_transform": self._generate_string_transform_tasks,
            "table_reasoning": self._generate_table_reasoning_tasks,
            "tool_schema": self._generate_tool_schema_tasks,
        }

        per_category = max(1, gap_cfg.max_tasks // max(len(gap_cfg.categories), 1))
        for category in gap_cfg.categories:
            generator = category_generators.get(category)
            if not generator:
                continue
            tasks.extend(generator(per_category, rng))

        rng.shuffle(tasks)
        return tasks[:gap_cfg.max_tasks]

    def _generate_format_strict_tasks(self, count: int, rng: random.Random) -> List[GapTask]:
        tasks = []
        for idx in range(count):
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            word = rng.choice(["alpha", "bravo", "charlie", "delta", "echo"])
            prompt = (
                "Return ONLY a JSON object with keys alpha, beta, gamma.\n"
                f"alpha = {a} + {b}\n"
                f"beta = reverse of '{word}'\n"
                "gamma = list of first 3 prime numbers\n"
                "No extra keys. No commentary."
            )
            payload = {
                "alpha": a + b,
                "beta": word[::-1],
                "gamma": [2, 3, 5]
            }
            oracle = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            tasks.append(GapTask(
                task_id=f"format_strict_{idx}",
                prompt=prompt,
                oracle_response=oracle,
                category="format_strict"
            ))
        return tasks

    def _generate_multi_step_math_tasks(self, count: int, rng: random.Random) -> List[GapTask]:
        tasks = []
        ops = ["add", "sub", "mul"]
        for idx in range(count):
            start = rng.randint(5, 50)
            steps = []
            value = start
            for _ in range(3):
                op = rng.choice(ops)
                num = rng.randint(2, 15)
                if op == "add":
                    value += num
                    steps.append(f"add {num}")
                elif op == "sub":
                    value -= num
                    steps.append(f"subtract {num}")
                else:
                    value *= num
                    steps.append(f"multiply by {num}")
            prompt = (
                f"Start with {start}. Then {', '.join(steps)}. "
                "Return ONLY the final integer."
            )
            oracle = str(value)
            tasks.append(GapTask(
                task_id=f"multi_step_math_{idx}",
                prompt=prompt,
                oracle_response=oracle,
                category="multi_step_math"
            ))
        return tasks

    def _generate_string_transform_tasks(self, count: int, rng: random.Random) -> List[GapTask]:
        tasks = []
        samples = ["pipeline", "upgrade", "proprietary", "context", "dataset"]
        for idx in range(count):
            text = rng.choice(samples)
            shift = rng.randint(1, 4)
            prompt = (
                f"Transform the string '{text}' by:\n"
                "1) reverse it\n"
                f"2) shift each letter forward by {shift} (a->b, z->a)\n"
                "Return ONLY the final string."
            )
            reversed_text = text[::-1]
            shifted = []
            for ch in reversed_text:
                if "a" <= ch <= "z":
                    offset = (ord(ch) - ord("a") + shift) % 26
                    shifted.append(chr(ord("a") + offset))
                else:
                    shifted.append(ch)
            oracle = "".join(shifted)
            tasks.append(GapTask(
                task_id=f"string_transform_{idx}",
                prompt=prompt,
                oracle_response=oracle,
                category="string_transform"
            ))
        return tasks

    def _generate_table_reasoning_tasks(self, count: int, rng: random.Random) -> List[GapTask]:
        tasks = []
        for idx in range(count):
            rows = []
            total = 0
            for row_id in range(4):
                score = rng.randint(10, 99)
                flag = rng.choice(["Y", "N"])
                rows.append((row_id + 1, score, flag))
                if flag == "Y":
                    total += score
            table = "id,score,flag\n" + "\n".join(
                f"{r[0]},{r[1]},{r[2]}" for r in rows
            )
            prompt = (
                "Given the CSV table below, sum the score for rows with flag=Y.\n"
                "Return ONLY the integer sum.\n\n"
                f"{table}"
            )
            tasks.append(GapTask(
                task_id=f"table_reasoning_{idx}",
                prompt=prompt,
                oracle_response=str(total),
                category="table_reasoning"
            ))
        return tasks

    def _generate_tool_schema_tasks(self, count: int, rng: random.Random) -> List[GapTask]:
        tasks = []
        for idx in range(count):
            a = rng.randint(3, 12)
            b = rng.randint(3, 12)
            prompt = (
                "Return ONLY a JSON object that matches this schema:\n"
                "{ \"tool\": \"calculator\", \"arguments\": {\"x\": int, \"y\": int, \"op\": \"add\"} }\n"
                f"Use x={a}, y={b}, op=add. No extra keys."
            )
            payload = {"tool": "calculator", "arguments": {"x": a, "y": b, "op": "add"}}
            oracle = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            tasks.append(GapTask(
                task_id=f"tool_schema_{idx}",
                prompt=prompt,
                oracle_response=oracle,
                category="tool_schema"
            ))
        return tasks

    def _validate_gap_response(self, task: GapTask, response: str) -> Tuple[bool, str]:
        """Validate model response for a gap task."""
        response = response.strip()

        if task.category in ("format_strict", "tool_schema"):
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                return False, "invalid_json"
            try:
                oracle_parsed = json.loads(task.oracle_response)
            except json.JSONDecodeError:
                return False, "oracle_invalid_json"
            if parsed != oracle_parsed:
                return False, "json_mismatch"
            return True, ""

        if task.category in ("multi_step_math", "table_reasoning"):
            token = response.split()[0] if response else ""
            if not token.lstrip("-").isdigit():
                return False, "not_integer"
            if token != task.oracle_response:
                return False, "numeric_mismatch"
            return True, ""

        if task.category == "string_transform":
            if response != task.oracle_response:
                return False, "string_mismatch"
            return True, ""

        return False, "unknown_category"

    def probe_gap_tasks(self, tasks: List[GapTask]) -> List[GapProbeResult]:
        """Probe GPT-5.1-codex-mini and keep only tasks it fails."""
        gap_cfg = self.config.gap_targeting
        if not tasks:
            return []

        if gap_cfg.require_probe and not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed for gap probing.")

        model_override = gap_cfg.probe_model
        system_prompt = (
            "You are a precise assistant. Follow instructions exactly. "
            "Return only the requested format."
        )
        results: List[GapProbeResult] = []
        failures = 0

        for task in tasks:
            if failures >= gap_cfg.max_failures:
                break
            try:
                response = self.gpt_client.generate_response(
                    task.prompt,
                    system_prompt=system_prompt,
                    temperature=gap_cfg.probe_temperature,
                    max_tokens=gap_cfg.probe_max_tokens,
                    model_override=model_override
                )
            except Exception as e:
                response = ""
                failure_reason = f"probe_error:{e}"
                results.append(GapProbeResult(
                    task_id=task.task_id,
                    category=task.category,
                    prompt=task.prompt,
                    oracle_response=task.oracle_response,
                    model_response=response,
                    passed=False,
                    failure_reason=failure_reason
                ))
                failures += 1
                continue

            passed, reason = self._validate_gap_response(task, response)
            results.append(GapProbeResult(
                task_id=task.task_id,
                category=task.category,
                prompt=task.prompt,
                oracle_response=task.oracle_response,
                model_response=response,
                passed=passed,
                failure_reason=reason
            ))
            if not passed:
                failures += 1

        if gap_cfg.persist_probe_results:
            self._persist_gap_probe_results(results)

        return results

    def _persist_gap_probe_results(self, results: List[GapProbeResult]):
        path = self._persist_path("gap_probe_results.jsonl")
        for result in results:
            self._append_jsonl(path, asdict(result))

    def build_gap_training_pairs(self) -> List[Tuple[str, str]]:
        """Generate and filter gap tasks to only those GPT-5.1-codex-mini fails."""
        print(f"  Building gap training pairs...")
        gap_cfg = self.config.gap_targeting
        if not gap_cfg.enabled:
            print(f"    Gap targeting disabled")
            return []

        print(f"    Generating gap tasks...", end="", flush=True)
        tasks = self._generate_gap_tasks()
        print(f" \033[92m{len(tasks)} tasks\033[0m")
        if not tasks:
            return []

        try:
            print(f"    Probing GPT-5.1-codex-mini on gap tasks...", end="", flush=True)
            probe_results = self.probe_gap_tasks(tasks)
            print(f" \033[92mdone\033[0m")
        except Exception as e:
            print(f" \033[91mERROR: {e}\033[0m")
            warnings.warn(f"Gap probing failed: {e}")
            return []
        failed = [r for r in probe_results if not r.passed]
        print(f"    GPT-5.1-codex-mini failed on \033[93m{len(failed)}\033[0m/{len(probe_results)} tasks")

        pairs = [(r.prompt, r.oracle_response) for r in failed]
        return pairs[:gap_cfg.max_failures]

    def build_targeted_training_pairs(
        self,
        allow_downloads: Optional[bool] = None,
        include_sources: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """Combine gap-targeted failures with general proprietary data."""
        print(f"\n\033[96m{'─'*60}\033[0m")
        print(f"\033[96m  Building Targeted Training Pairs\033[0m")
        print(f"\033[96m{'─'*60}\033[0m")

        gap_pairs = self.build_gap_training_pairs()
        print(f"\n  Loading general repository data...")
        general_pairs = self.load_repository_data(
            allow_downloads=allow_downloads,
            include_sources=include_sources
        )

        gap_cfg = self.config.gap_targeting
        if not gap_pairs:
            print(f"\n  \033[92mNo gap pairs, using {len(general_pairs)} general pairs\033[0m")
            return general_pairs

        general_target = int(len(gap_pairs) * gap_cfg.general_ratio / max(1.0 - gap_cfg.general_ratio, 1e-6))
        if general_target > 0 and general_pairs:
            general_pairs = general_pairs[:general_target]

        combined = gap_pairs + general_pairs
        random.shuffle(combined)
        print(f"\n  \033[92mCombined: {len(gap_pairs)} gap + {len(general_pairs)} general = {len(combined)} pairs\033[0m")
        self._persist_run_event("targeted_pairs_built", {
            "gap_pairs": len(gap_pairs),
            "general_pairs": len(general_pairs),
            "total_pairs": len(combined)
        })
        return combined

    def export_targeted_corpus(
        self,
        output_file: Optional[str] = None,
        allow_downloads: Optional[bool] = None,
        include_sources: Optional[List[str]] = None
    ) -> str:
        """Export a corpus focused on GPT-5.1-codex-mini gap failures + general data."""
        pairs = self.build_targeted_training_pairs(
            allow_downloads=allow_downloads,
            include_sources=include_sources
        )
        corpus = self._format_training_pairs(pairs)
        if output_file:
            with open(output_file, "w") as f:
                f.write(corpus)
        self._persist_run_event("export_targeted_corpus", {
            "pairs": len(pairs),
            "output_file": output_file
        })
        return corpus

    def export_targeted_jsonl(
        self,
        output_file: str,
        allow_downloads: Optional[bool] = None,
        include_sources: Optional[List[str]] = None
    ) -> str:
        """Export gap-targeted pairs to JSONL (messages format)."""
        pairs = self.build_targeted_training_pairs(
            allow_downloads=allow_downloads,
            include_sources=include_sources
        )
        self._write_pairs_jsonl(pairs, output_file)
        self._persist_run_event("export_targeted_jsonl", {
            "pairs": len(pairs),
            "output_file": output_file
        })
        return output_file

    def export_repository_jsonl(
        self,
        output_file: str,
        **load_kwargs: Any
    ) -> str:
        """Export repository pairs to JSONL (messages format)."""
        pairs = self.load_repository_data(**load_kwargs)
        self._write_pairs_jsonl(pairs, output_file)
        self._persist_run_event("export_repository_jsonl", {
            "pairs": len(pairs),
            "output_file": output_file
        })
        return output_file

    def _get_cache_path(self, iteration: int) -> Path:
        """Get cache file path for an iteration."""
        return self.cache_dir / f"enhanced_data_iter_{iteration}.jsonl"

    def _load_cached(self, iteration: int) -> Optional[List[EnhancedExample]]:
        """Load cached enhanced data if available."""
        cache_path = self._get_cache_path(iteration)
        if not cache_path.exists():
            return None

        examples = []
        with open(cache_path) as f:
            for line in f:
                examples.append(EnhancedExample.from_dict(json.loads(line)))
        return examples

    def _save_cached(self, iteration: int, examples: List[EnhancedExample]):
        """Save enhanced data to cache."""
        cache_path = self._get_cache_path(iteration)
        with open(cache_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

    def _log_rejection(self, example: EnhancedExample, reason: str):
        """Log rejected examples for analysis."""
        if not self.config.log_rejections:
            return

        log_path = Path(self.config.rejection_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'a') as f:
            f.write(json.dumps({
                "example": example.to_dict(),
                "rejection_reason": reason,
                "timestamp": datetime.now().isoformat()
            }) + "\n")

    def generate_from_model(
        self,
        model,
        tokenizer,
        prompts: List[str],
        device,
        max_tokens: int = 256,
        temperature: float = 0.8
    ) -> List[Tuple[str, str]]:
        """
        Generate responses from the current model.

        Args:
            model: The trained model (MiniGPT or InfiniGPT)
            tokenizer: The tokenizer
            prompts: List of prompts to generate responses for
            device: torch device
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature

        Returns:
            List of (prompt, response) tuples
        """
        model.eval()
        results = []

        for prompt in prompts:
            formatted_prompt = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"
            try:
                output = model.generate(
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    device=device
                )

                # Extract just the assistant's response
                if "<|assistant|>" in output:
                    response = output.split("<|assistant|>")[-1].strip()
                else:
                    response = output[len(formatted_prompt):].strip()
                # Remove end_turn token if present
                if "<|end_turn|>" in response:
                    response = response.split("<|end_turn|>")[0].strip()

                results.append((prompt, response))
                self.stats["total_generated"] += 1

            except Exception as e:
                print(f"Generation failed for prompt: {prompt[:50]}... Error: {e}")

        if results:
            self._persist_stats()
            self._persist_run_event("generation_completed", {
                "generated": len(results)
            })

        return results

    def enhance_training_data(
        self,
        examples: List[Tuple[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        iteration: Optional[int] = None
    ) -> List[EnhancedExample]:
        """
        Enhance training data using GPT-5.1-codex-mini.

        Args:
            examples: List of (prompt, response) tuples
            progress_callback: Optional progress callback
            iteration: Optional iteration number for persistence metadata

        Returns:
            List of enhanced and verified examples
        """
        batch_id = datetime.now().isoformat()
        self._persist_run_event("enhance_batch_started", {
            "batch_id": batch_id,
            "total_examples": len(examples)
        })

        # Determine which examples to enhance
        if self.config.enhancement_ratio < 1.0:
            sample_size = int(len(examples) * self.config.enhancement_ratio)
            to_enhance = random.sample(examples, min(sample_size, len(examples)))
        else:
            to_enhance = examples

        # Enhance with GPT-5.1-codex-mini
        enhanced = self.gpt_client.batch_enhance(to_enhance, progress_callback)

        # Filter by quality threshold
        accepted = []
        for ex in enhanced:
            self.stats["total_enhanced"] += 1

            # Track enhancement types
            etype = ex.enhancement_type
            self.stats["enhancement_types"][etype] = \
                self.stats["enhancement_types"].get(etype, 0) + 1

            if ex.quality_score >= self.config.verification_threshold:
                accepted.append(ex)
                self.stats["total_accepted"] += 1
            else:
                self.stats["total_rejected"] += 1
                self._log_rejection(ex, f"Quality score {ex.quality_score} below threshold")

        # Update average quality
        if accepted:
            total_quality = sum(ex.quality_score for ex in accepted)
            self.stats["avg_quality_score"] = total_quality / len(accepted)

        self._persist_enhanced_batch(
            enhanced,
            batch_id,
            source="enhance_training_data",
            iteration=iteration
        )
        self._persist_stats()
        self._persist_run_event("enhance_batch_completed", {
            "batch_id": batch_id,
            "enhanced_total": len(enhanced),
            "accepted": len(accepted),
            "rejected": len(enhanced) - len(accepted)
        })

        return accepted

    def run_iteration(
        self,
        model,
        tokenizer,
        device,
        iteration: int,
        prompts: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[EnhancedExample]:
        """
        Run one iteration of the upgrade pipeline.

        Args:
            model: Current trained model
            tokenizer: Tokenizer for the model
            device: torch device
            iteration: Iteration number (for caching)
            prompts: Optional list of prompts (generates if not provided)
            use_cache: Whether to use cached results

        Returns:
            List of enhanced training examples
        """
        self._persist_run_event("iteration_started", {
            "iteration": iteration,
            "use_cache": use_cache
        })

        # Check cache
        if use_cache:
            cached = self._load_cached(iteration)
            if cached:
                print(f"Loaded {len(cached)} cached examples for iteration {iteration}")
                self._persist_run_event("iteration_loaded_cache", {
                    "iteration": iteration,
                    "cached_examples": len(cached)
                })
                return cached

        # Generate prompts if not provided
        if prompts is None:
            prompts = self._generate_diverse_prompts()

        print(f"Generating responses from model for {len(prompts)} prompts...")
        generated = self.generate_from_model(
            model, tokenizer, prompts, device,
            max_tokens=self.config.max_response_length
        )
        self._persist_generated_batch(generated, iteration=iteration, label="generated")

        print(f"Enhancing {len(generated)} examples with GPT-5.1-codex-mini...")
        enhanced = self.enhance_training_data(
            generated,
            progress_callback=lambda cur, tot: print(f"\rEnhancing: {cur}/{tot}", end=""),
            iteration=iteration
        )
        print()

        # Cache results
        if use_cache or self.config.persist_enhanced_data:
            self._save_cached(iteration, enhanced)

        self._persist_run_event("iteration_completed", {
            "iteration": iteration,
            "generated": len(generated),
            "enhanced": len(enhanced)
        })

        return enhanced

    def _generate_diverse_prompts(self, model_samples: List[Tuple[str, str]] = None) -> List[str]:
        """Generate OPTIMAL training data using mathematical optimization principles.

        Theory: Treat training data generation as information-theoretic optimization.
        Goal: Maximize I(Model; Data) - mutual information between model improvement and data.

        Approach:
        1. Analyze current model outputs to find weaknesses
        2. Generate prompts that target those specific weaknesses
        3. No fixed categories - pure optimization based on model behavior
        """
        import random

        prompts = []
        self.gpt_client._ensure_client()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        prompt_lock = threading.Lock()
        generated_count = [0]

        # Build context from model samples if available
        model_context = ""
        if model_samples and len(model_samples) > 0:
            sample_examples = random.sample(model_samples, min(5, len(model_samples)))
            model_context = "\n\n=== CURRENT MODEL OUTPUTS (analyze weaknesses) ===\n"
            for i, (prompt, response) in enumerate(sample_examples):
                model_context += f"\nExample {i+1}:\nPrompt: {prompt[:200]}...\nModel Response: {response[:300]}...\n"
            model_context += "\n=== END MODEL OUTPUTS ===\n"
            model_context += "\nBased on these outputs, identify weaknesses and generate prompts to FIX them.\n"

        # MASTER PROMPT: Generate optimal training data WITH model context
        master_instruction = f"""You are an AI training data optimization engine. Your goal is to generate
the SINGLE MOST VALUABLE training prompt that would maximally improve THIS SPECIFIC model.
{model_context}
Mathematical objective: Generate prompt P that maximizes:
- Information gain: How much new knowledge does answering this require?
- Coverage: Does this explore underrepresented regions of knowledge space?
- WEAKNESS TARGETING: Based on model outputs above, what does this model NEED to learn?
- Generalization: Will learning this transfer to many other tasks?

Rules:
1. NO CATEGORIES - generate from the infinite space of all possible questions
2. ANALYZE the model outputs and TARGET its weaknesses
3. Each prompt should fix something the model is DOING WRONG
4. Target edge cases where the model would fail
5. The prompt should require REASONING not just retrieval

Generate ONE prompt that would MOST improve this model. Just the question/task, no answer."""

        def generate_optimal_prompt(seed: int) -> str:
            """Generate a single optimal prompt using randomized seed for diversity."""
            try:
                # Add randomness to encourage diversity
                diversity_hint = f"[Diversity seed: {seed}] [Temperature: HIGH] [Explore: {'EDGE CASES' if seed % 3 == 0 else 'NOVEL COMBINATIONS' if seed % 3 == 1 else 'COUNTERINTUITIVE'}]"

                response = self.gpt_client._client.responses.create(
                    model="gpt-5.1-codex-mini",
                    input=f"{master_instruction}\n\n{diversity_hint}\n\nGenerate ONE optimal training prompt:",
                    temperature=0.95,  # High temperature for diversity
                    max_output_tokens=400
                )
                return response.output_text.strip()
            except Exception:
                try:
                    response = self.gpt_client._client.responses.create(
                        model="gpt-4o",
                        input=f"{master_instruction}\n\n{diversity_hint}\n\nGenerate ONE optimal training prompt:",
                        temperature=0.95,
                        max_output_tokens=400
                    )
                    return response.output_text.strip()
                except Exception:
                    return None

        # Generate prompts in parallel - pure optimization, no categories
        num_prompts = self.config.prompts_per_category
        max_workers = min(100, num_prompts)
        print(f"  [GPT-5.1-codex-mini OPTIMAL] Generating {num_prompts} optimal prompts with {max_workers} threads...")
        print(f"  [GPT-5.1-codex-mini OPTIMAL] Mode: PURE OPTIMIZATION (no categories)")

        tasks = list(range(num_prompts))  # Seeds for diversity

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_optimal_prompt, seed): seed for seed in tasks}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    with prompt_lock:
                        prompts.append(result)
                        generated_count[0] += 1
                # Real-time progress
                if completed % 5 == 0 or completed == len(tasks):
                    print(f"\r  [GPT-5.1-codex-mini OPTIMAL] ✓ {generated_count[0]}/{len(tasks)} optimal prompts", end="", flush=True)
            print()  # Newline at end

        print(f"  [GPT-5.1-codex-mini OPTIMAL] Generated {len(prompts)} unique optimal prompts")

        random.shuffle(prompts)
        return prompts

    def export_training_corpus(
        self,
        examples: List[EnhancedExample],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export enhanced examples as a training corpus.

        Args:
            examples: List of enhanced examples
            output_path: Optional output file path

        Returns:
            The corpus as a string
        """
        corpus_parts = []

        for ex in examples:
            pair = f"<|user|>\n{ex.original_prompt}\n<|end_turn|>\n<|assistant|>\n{ex.enhanced_response}\n<|end_turn|>"
            corpus_parts.append(pair)

        corpus = "\n\n".join(corpus_parts)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(corpus)

        self._persist_run_event("export_training_corpus", {
            "examples": len(examples),
            "output_path": output_path
        })

        return corpus

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.stats.copy()

    def print_statistics(self):
        """Print formatted statistics."""
        print("\n" + "=" * 50)
        print("Training Data Upgrade Pipeline Statistics")
        print("=" * 50)
        print(f"Total generated:     {self.stats['total_generated']}")
        print(f"Total enhanced:      {self.stats['total_enhanced']}")
        print(f"Total accepted:      {self.stats['total_accepted']}")
        print(f"Total rejected:      {self.stats['total_rejected']}")
        print(f"Avg quality score:   {self.stats['avg_quality_score']:.3f}")
        print("\nEnhancement types:")
        for etype, count in self.stats['enhancement_types'].items():
            print(f"  {etype}: {count}")
        print("=" * 50)


class IterativeTrainer:
    """
    Orchestrates iterative training with the upgrade pipeline.

    This class manages the full loop for FROM-SCRATCH proprietary model training:
    1. Train model generation N on current corpus
    2. Model N generates candidate responses
    3. GPT-5.1-codex-mini enhances/corrects the responses
    4. Enhanced data augments corpus for generation N+1
    5. Repeat until max_iterations

    Each iteration improves data quality through GPT-5.1-codex-mini verification,
    creating a self-improving training loop where each model generation
    is trained on progressively higher-quality data.

    Mathematical Foundation:
    - Let M_n be model at generation n with capability C(M_n)
    - Let G be GPT-5.1-codex-mini with capability C(G) >> C(M_0)
    - Training data D_n = Enhance(Generate(M_{n-1}), G)
    - Then: C(M_n) >= C(M_{n-1}) with high probability
    """

    def __init__(
        self,
        base_config,
        pipeline_config: Optional[PipelineConfig] = None,
        max_iterations: int = 3
    ):
        self.base_config = base_config
        self.pipeline = TrainingDataUpgradePipeline(pipeline_config)
        self.max_iterations = max_iterations
        self.iteration_history = []
        self.models = []  # Store each generation's model
        self.previous_samples = []  # Store samples from previous generation for context

    def run_generation(
        self,
        model,
        tokenizer,
        device,
        generation: int,
        num_prompts: int = 100
    ) -> List[Tuple[str, str]]:
        """
        Run one generation cycle: model generates, GPT-5.1-codex-mini enhances.

        Args:
            model: Current trained model (from-scratch proprietary)
            tokenizer: Tokenizer for the model
            device: torch device
            generation: Generation number (0-indexed)
            num_prompts: Number of prompts to generate responses for

        Returns:
            List of (prompt, enhanced_response) pairs
        """
        print(f"\n  [Generation {generation + 1}] Generating {num_prompts} responses...")
        if self.previous_samples:
            print(f"  [Generation {generation + 1}] Using {len(self.previous_samples)} samples from previous gen for context")

        # Generate diverse prompts WITH context from previous generation
        prompts = self.pipeline._generate_diverse_prompts(model_samples=self.previous_samples)[:num_prompts]

        # Model generates candidate responses
        generated = self.pipeline.generate_from_model(
            model, tokenizer, prompts, device,
            max_tokens=self.pipeline.config.max_response_length,
            temperature=0.8
        )
        print(f"  [Generation {generation + 1}] Generated {len(generated)} responses")

        # Store samples for NEXT generation's prompt generation
        self.previous_samples = generated[:50]  # Keep last 50 samples for context

        # GPT-5.1-codex-mini enhances and verifies responses
        print(f"  [Generation {generation + 1}] Enhancing with GPT-5.1-codex-mini...")
        enhanced = self.pipeline.enhance_training_data(
            generated,
            progress_callback=lambda cur, tot: print(f"\r    Enhancing: {cur}/{tot}", end=""),
            iteration=generation
        )
        print()
        print(f"  [Generation {generation + 1}] Accepted {len(enhanced)} enhanced examples")

        # Convert to training pairs
        pairs = [ex.to_training_pair() for ex in enhanced]
        return pairs

    def run_full_pipeline(
        self,
        initial_corpus: str,
        tokenizer_class,
        model_creator: Callable,
        trainer_func: Callable,
        device,
        prompts_per_generation: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Run the full iterative training pipeline for from-scratch model.

        This creates a virtuous cycle where each model generation is trained
        on data enhanced by GPT-5.1-codex-mini, progressively improving quality.

        Args:
            initial_corpus: Starting training corpus
            tokenizer_class: Class for creating tokenizer
            model_creator: Function(config, device) -> model
            trainer_func: Function(model, tokenizer, corpus, config, device) -> trained_model
            device: torch device
            prompts_per_generation: Prompts to generate per iteration

        Returns:
            History of generations with metrics
        """
        current_corpus = initial_corpus
        current_model = None

        self.pipeline._persist_run_event("iterative_training_started", {
            "max_iterations": self.max_iterations,
            "prompts_per_generation": prompts_per_generation,
            "initial_corpus_size": len(initial_corpus)
        })

        for generation in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"MODEL GENERATION {generation + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            print(f"  Corpus size: {len(current_corpus):,} chars")

            # Create tokenizer for this generation
            tokenizer = tokenizer_class()
            tokenizer.train(current_corpus, self.base_config.vocab_size)
            print(f"  Vocab size: {tokenizer.vocab_size}")

            # Create fresh model from scratch
            model = model_creator(self.base_config, device)
            print(f"  Model params: {model.get_num_params():,}")

            # Train this generation's model
            print(f"\n  Training generation {generation + 1}...")
            trained_model = trainer_func(model, tokenizer, current_corpus, self.base_config, device)
            current_model = trained_model
            self.models.append(trained_model)

            # Generate enhanced data for next generation (except last)
            enhanced_pairs = []
            if generation < self.max_iterations - 1:
                enhanced_pairs = self.run_generation(
                    trained_model, tokenizer, device,
                    generation, prompts_per_generation
                )

                # Augment corpus with GPT-5.1-codex-mini enhanced data
                if enhanced_pairs:
                    new_corpus = self.pipeline._format_training_pairs(enhanced_pairs)
                    current_corpus = current_corpus + "\n\n\n" + new_corpus
                    print(f"  Corpus augmented: +{len(new_corpus):,} chars")

            # Record generation metrics
            self.iteration_history.append({
                "generation": generation + 1,
                "corpus_size": len(current_corpus),
                "enhanced_examples": len(enhanced_pairs),
                "total_enhanced": self.pipeline.stats["total_enhanced"],
                "total_accepted": self.pipeline.stats["total_accepted"],
                "avg_quality_score": self.pipeline.stats["avg_quality_score"],
                "pipeline_stats": self.pipeline.get_statistics()
            })
            self.pipeline.persist_iteration_history(self.iteration_history)

        self.pipeline._persist_run_event("iterative_training_completed", {
            "generations": self.max_iterations,
            "final_corpus_size": len(current_corpus),
            "total_enhanced": self.pipeline.stats["total_enhanced"]
        })

        return self.iteration_history

    def get_latest_model(self):
        """Get the most recently trained model generation."""
        return self.models[-1] if self.models else None

    def get_model_generation(self, generation: int):
        """Get a specific model generation (1-indexed)."""
        if 1 <= generation <= len(self.models):
            return self.models[generation - 1]
        return None


def _register_repo_data_sources(pipeline: TrainingDataUpgradePipeline) -> None:
    """Register repository data sources using lazy loaders."""
    sources = [
        {
            "name": "core_training_data",
            "module": "data",
            "func": "get_all_training_data",
            "extra_kwargs": {"focused": False},
            "requires_network": False,
            "description": "Core hand-curated training data"
        },
        {
            "name": "general_training_data",
            "module": "general_training_data",
            "func": "get_general_training_data",
            "extra_kwargs": None,
            "requires_network": False,
            "description": "General-purpose training pairs"
        },
        {
            "name": "expanded_training_data",
            "module": "expanded_training_data",
            "func": "get_expanded_training_data",
            "extra_kwargs": None,
            "requires_network": False,
            "description": "Verified identity and coding pairs"
        },
        {
            "name": "external_synthetic_data",
            "module": "download_data",
            "func": "get_all_external_data",
            "extra_kwargs": None,
            "requires_network": False,
            "description": "Synthetic and cached external instruction data"
        },
        {
            "name": "instruction_data",
            "module": "download_instruction_data",
            "func": "get_all_instruction_data",
            "extra_kwargs": None,
            "requires_network": False,
            "description": "Instruction-following data with optional external downloads"
        },
        {
            "name": "security_data",
            "module": "download_security_data",
            "func": "get_all_security_data",
            "extra_kwargs": None,
            "requires_network": False,
            "description": "Security and sysadmin training data"
        },
        {
            "name": "knowledge_data",
            "module": "download_knowledge_data",
            "func": "get_all_knowledge_data",
            "extra_kwargs": None,
            "requires_network": False,
            "description": "Factual knowledge and balanced responses"
        },
        {
            "name": "instruction_datasets",
            "module": "download_instruction_datasets",
            "func": "get_all_instruction_datasets",
            "extra_kwargs": None,
            "requires_network": True,
            "description": "Large external instruction datasets (network required)"
        },
    ]

    for spec in sources:
        pipeline.register_repo_data_source(
            spec["name"],
            _lazy_loader(spec["module"], spec["func"], spec["extra_kwargs"]),
            description=spec["description"],
            requires_network=spec["requires_network"]
        )


# Global pipeline instance (initialized lazily)
_global_pipeline: Optional[TrainingDataUpgradePipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> TrainingDataUpgradePipeline:
    """Get or create the global pipeline instance."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = TrainingDataUpgradePipeline(config)
        _global_pipeline._ensure_repo_sources_registered()
    return _global_pipeline


def initialize_pipeline(config: Optional[PipelineConfig] = None) -> TrainingDataUpgradePipeline:
    """
    Initialize the training data upgrade pipeline.

    This MUST be called before training to ensure all data benefits from
    the upgrade infrastructure. Called automatically by __init__.py.

    Args:
        config: Optional pipeline configuration

    Returns:
        Initialized pipeline instance
    """
    global _global_pipeline
    _global_pipeline = TrainingDataUpgradePipeline(config)
    _global_pipeline._ensure_repo_sources_registered()
    print("Training Data Upgrade Pipeline initialized.")
    print(f"  Cache directory: {_global_pipeline.cache_dir}")
    print(f"  Persist directory: {_global_pipeline.persist_dir}")
    print(f"  Enhancement ratio: {_global_pipeline.config.enhancement_ratio}")
    print(f"  Quality threshold: {_global_pipeline.config.verification_threshold}")
    if _global_pipeline.config.require_full_source:
        print(f"  Full source required: yes ({_global_pipeline.config.full_source_env_var})")
    return _global_pipeline


# Export main components
__all__ = [
    "PipelineConfig",
    "GapTargetingConfig",
    "EnhancedExample",
    "RepoDataSource",
    "GapTask",
    "GapProbeResult",
    "GPT52Client",
    "TrainingDataUpgradePipeline",
    "IterativeTrainer",
    "get_pipeline",
    "initialize_pipeline"
]

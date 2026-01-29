"""
Huawei NPU (Ascend) Compatibility Layer.

Provides device abstraction and optimizations for Huawei Ascend NPUs
while maintaining full compatibility with CUDA, MPS, and CPU backends.

Huawei Ascend NPUs use the CANN (Compute Architecture for Neural Networks)
framework and the torch_npu extension for PyTorch integration.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class DeviceInfo:
    """Information about the compute device."""
    device: torch.device
    device_type: str  # 'npu', 'cuda', 'mps', 'cpu'
    device_name: str
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None
    supports_fp16: bool = True
    supports_bf16: bool = False
    recommended_dtype: torch.dtype = torch.float32


# Global flag to track NPU availability
_NPU_AVAILABLE: Optional[bool] = None
_torch_npu = None


def is_npu_available() -> bool:
    """Check if Huawei NPU (Ascend) is available."""
    global _NPU_AVAILABLE, _torch_npu

    if _NPU_AVAILABLE is not None:
        return _NPU_AVAILABLE

    try:
        import torch_npu
        _torch_npu = torch_npu
        _NPU_AVAILABLE = torch.npu.is_available()
    except ImportError:
        _NPU_AVAILABLE = False
    except Exception:
        _NPU_AVAILABLE = False

    return _NPU_AVAILABLE


def get_npu_module():
    """Get the torch_npu module if available."""
    global _torch_npu
    if _torch_npu is None and is_npu_available():
        import torch_npu
        _torch_npu = torch_npu
    return _torch_npu


def get_device_count() -> int:
    """Get the number of available compute devices."""
    if is_npu_available():
        return torch.npu.device_count()
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 1
    return 0


def get_device(device_id: int = 0) -> torch.device:
    """Get the best available compute device."""
    if is_npu_available():
        return torch.device(f"npu:{device_id}")
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(device: Optional[torch.device] = None) -> DeviceInfo:
    """Get detailed information about a device."""
    if device is None:
        device = get_device()

    device_type = device.type

    if device_type == "npu":
        idx = device.index if device.index is not None else 0
        name = f"Huawei Ascend NPU {idx}"
        try:
            props = torch.npu.get_device_properties(idx)
            memory_total = props.total_memory if hasattr(props, 'total_memory') else None
            name = props.name if hasattr(props, 'name') else name
        except Exception:
            memory_total = None

        return DeviceInfo(
            device=device,
            device_type="npu",
            device_name=name,
            memory_total=memory_total,
            supports_fp16=True,
            supports_bf16=True,  # Ascend 910 supports BF16
            recommended_dtype=torch.float16
        )

    elif device_type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            device=device,
            device_type="cuda",
            device_name=props.name,
            memory_total=props.total_memory,
            supports_fp16=True,
            supports_bf16=props.major >= 8,  # Ampere+
            recommended_dtype=torch.bfloat16 if props.major >= 8 else torch.float16
        )

    elif device_type == "mps":
        return DeviceInfo(
            device=device,
            device_type="mps",
            device_name="Apple Silicon MPS",
            supports_fp16=True,
            supports_bf16=False,
            recommended_dtype=torch.float16
        )

    else:
        return DeviceInfo(
            device=device,
            device_type="cpu",
            device_name="CPU",
            supports_fp16=False,
            supports_bf16=False,
            recommended_dtype=torch.float32
        )


def setup_device(seed: int = 42, device_id: int = 0, verbose: bool = True) -> torch.device:
    """
    Setup compute device with proper seeding and optimizations.

    Supports:
    - Huawei Ascend NPU (via torch_npu)
    - NVIDIA CUDA GPUs
    - Apple Silicon MPS
    - CPU fallback
    """
    torch.manual_seed(seed)

    device = get_device(device_id)
    info = get_device_info(device)

    if info.device_type == "npu":
        # Huawei NPU setup
        torch.npu.manual_seed_all(seed)
        torch.npu.set_device(device)

        # Enable optimizations
        try:
            # CANN specific optimizations
            torch.npu.set_compile_mode(jit_compile=True)
        except Exception:
            pass

        if verbose:
            print(f"\033[2mUsing Huawei NPU: {info.device_name}\033[0m")
            if info.memory_total:
                print(f"\033[2m  Memory: {info.memory_total / 1e9:.1f} GB\033[0m")

    elif info.device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device)

        # Enable TF32 for Ampere+ GPUs
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if verbose:
            print(f"\033[2mUsing CUDA: {info.device_name}\033[0m")
            print(f"\033[2m  Memory: {info.memory_total / 1e9:.1f} GB\033[0m")

    elif info.device_type == "mps":
        if verbose:
            print(f"\033[2mUsing Apple Silicon MPS\033[0m")

    else:
        if verbose:
            print(f"\033[2mUsing CPU\033[0m")

    return device


def empty_cache():
    """Clear GPU/NPU memory cache."""
    if is_npu_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """Synchronize device operations."""
    if is_npu_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_allocated(device: Optional[torch.device] = None) -> int:
    """Get currently allocated memory in bytes."""
    if device is None:
        device = get_device()

    if device.type == "npu":
        return torch.npu.memory_allocated(device)
    elif device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return 0


def memory_reserved(device: Optional[torch.device] = None) -> int:
    """Get currently reserved memory in bytes."""
    if device is None:
        device = get_device()

    if device.type == "npu":
        return torch.npu.memory_reserved(device)
    elif device.type == "cuda":
        return torch.cuda.memory_reserved(device)
    return 0


class DeviceAwareModule(nn.Module):
    """
    Base module with device-aware optimizations.

    Provides automatic dtype selection and device-specific optimizations
    for Huawei NPU, CUDA, and other backends.
    """

    def __init__(self):
        super().__init__()
        self._device_info: Optional[DeviceInfo] = None

    @property
    def device_info(self) -> DeviceInfo:
        """Get device info, caching the result."""
        if self._device_info is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            self._device_info = get_device_info(device)
        return self._device_info

    def _reset_device_info(self):
        """Reset cached device info (call after .to() operations)."""
        self._device_info = None

    def to(self, *args, **kwargs):
        """Override to reset device info cache."""
        result = super().to(*args, **kwargs)
        self._reset_device_info()
        return result


def get_optimal_attention_implementation(device: torch.device) -> str:
    """
    Get the optimal attention implementation for the given device.

    Returns:
        'sdpa': PyTorch scaled_dot_product_attention (default, works everywhere)
        'flash': Flash Attention v2 (CUDA Ampere+)
        'npu_flash': Huawei NPU optimized attention
    """
    if device.type == "npu":
        # Huawei NPU has its own optimized attention
        return "npu_flash"
    elif device.type == "cuda":
        # Check for Flash Attention support
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            return "sdpa"
        return "manual"
    else:
        return "sdpa"


def npu_compatible_einsum(equation: str, *operands) -> torch.Tensor:
    """
    Perform einsum with NPU compatibility.

    Some einsum operations may need special handling on Huawei NPUs.
    This wrapper provides fallbacks when needed.
    """
    try:
        return torch.einsum(equation, *operands)
    except RuntimeError as e:
        # Fallback for unsupported einsum patterns on NPU
        if "npu" in str(e).lower() or is_npu_available():
            # Try to decompose common patterns
            if equation == "bhld,bhmd->bhlm":
                # Batched matrix multiplication: Q @ K^T
                return torch.matmul(operands[0], operands[1].transpose(-2, -1))
            elif equation == "bhlm,bhmd->bhld":
                # Batched matrix multiplication: attn @ V
                return torch.matmul(operands[0], operands[1])
            elif equation == "ij,jk->ik":
                # Standard matrix multiplication
                return torch.matmul(operands[0], operands[1])
        raise


class NPUOptimizedLinear(nn.Linear):
    """
    Linear layer with Huawei NPU optimizations.

    Uses NPU-specific kernels when available for better performance.
    Falls back to standard PyTorch Linear on other devices.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type == "npu" and is_npu_available():
            # NPU-optimized path
            try:
                return torch.npu.npu_linear(input, self.weight, self.bias)
            except (AttributeError, RuntimeError):
                pass
        return super().forward(input)


def create_attention_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    causal: bool = True
) -> torch.Tensor:
    """
    Create attention mask optimized for the given device.

    Args:
        seq_len: Sequence length
        device: Target device
        dtype: Data type
        causal: If True, create causal (triangular) mask

    Returns:
        Attention mask tensor
    """
    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    else:
        mask = torch.ones(seq_len, seq_len, device=device, dtype=dtype)

    return mask


# Activation functions compatible with all backends
def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU + 1 activation, used in linear attention.

    This activation is used in Infini-attention for the σ function
    that maps keys and queries to non-negative values for linear attention.

    σ(x) = ELU(x) + 1

    This ensures:
    - Output is always positive (≥ ε for numerical stability)
    - Smooth gradients
    - Works well on all device types including Huawei NPU
    """
    return torch.nn.functional.elu(x) + 1.0


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Safe division with epsilon for numerical stability.

    Works correctly on all devices including Huawei NPU.
    """
    return numerator / (denominator + eps)

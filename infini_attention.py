"""
Infini-attention: Efficient Infinite Context Transformers

Implementation based on:
"Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
Tsendsuren Munkhdalai, Manaal Faruqui, Siddharth Gopal (Google Research)
arXiv: https://arxiv.org/abs/2404.07143

Key innovations:
1. Compressive memory that maintains fixed-size state across segments
2. Combined local attention + long-term linear attention in single block
3. Learned gating between local and memory-based attention
4. Delta rule for stable memory updates

Fully compatible with Huawei Ascend NPUs, NVIDIA CUDA, Apple MPS, and CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Optional Huawei NPU support - fallback to standard PyTorch if not available
try:
    from huawei_npu import (
        elu_plus_one,
        safe_divide,
        DeviceAwareModule,
        npu_compatible_einsum,
    )
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

    # Fallback implementations
    def elu_plus_one(x):
        """ELU + 1 activation for attention normalization."""
        return F.elu(x) + 1

    def safe_divide(a, b, eps=1e-6):
        """Safe division with epsilon to prevent NaN."""
        return a / (b + eps)

    class DeviceAwareModule(nn.Module):
        """Base module with device awareness."""
        pass

    def npu_compatible_einsum(equation, *operands):
        """Standard einsum fallback."""
        return torch.einsum(equation, *operands)


@dataclass
class InfiniAttentionConfig:
    """Configuration for Infini-attention module."""
    embed_dim: int = 192
    num_heads: int = 4
    dropout: float = 0.1
    use_rope: bool = False
    max_seq_len: int = 2048

    # Infini-attention specific
    segment_size: int = 128  # Local attention window size
    use_delta_rule: bool = True  # Use delta rule for memory updates
    init_gate_bias: float = 0.0  # Initial bias for gate (0 = equal weighting)
    memory_dim: Optional[int] = None  # Memory dimension (defaults to head_dim)


class CompressiveMemory(nn.Module):
    """
    Compressive memory for Infini-attention.

    Maintains a fixed-size memory matrix M and normalization vector z
    that accumulate information across segments via associative binding
    or delta rule updates.

    Memory shape: (batch, num_heads, d_key, d_value)
    Normalization shape: (batch, num_heads, d_key)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        use_delta_rule: bool = True,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_delta_rule = use_delta_rule
        self.eps = eps

    def init_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize empty memory state.

        Returns:
            M: Memory matrix of shape (batch, heads, d_key, d_value)
            z: Normalization vector of shape (batch, heads, d_key, 1)
        """
        M = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )
        z = torch.zeros(
            batch_size, self.num_heads, self.head_dim, 1,
            device=device, dtype=dtype
        )
        return M, z

    def retrieve(
        self,
        query: torch.Tensor,  # (batch, heads, seq, d_key)
        M: torch.Tensor,      # (batch, heads, d_key, d_value)
        z: torch.Tensor       # (batch, heads, d_key, 1)
    ) -> torch.Tensor:
        """
        Retrieve from compressive memory.

        A_mem = σ(Q) @ M / (σ(Q) @ z)

        Args:
            query: Query tensor after σ activation
            M: Memory matrix
            z: Normalization vector

        Returns:
            Retrieved memory content of shape (batch, heads, seq, d_value)
        """
        # Apply σ (ELU + 1) to query
        sigma_q = elu_plus_one(query)  # (B, H, T, D)

        # Memory retrieval: σ(Q) @ M
        # (B, H, T, D) @ (B, H, D, D) -> (B, H, T, D)
        mem_output = torch.matmul(sigma_q, M)

        # Normalization: σ(Q) @ z
        # (B, H, T, D) @ (B, H, D, 1) -> (B, H, T, 1)
        normalizer = torch.matmul(sigma_q, z)

        # Safe division
        return safe_divide(mem_output, normalizer, self.eps)

    def update(
        self,
        key: torch.Tensor,    # (batch, heads, seq, d_key)
        value: torch.Tensor,  # (batch, heads, seq, d_value)
        M: torch.Tensor,      # (batch, heads, d_key, d_value)
        z: torch.Tensor       # (batch, heads, d_key, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update compressive memory with new key-value pairs.

        Standard update:
            M_new = M + σ(K)^T @ V
            z_new = z + sum(σ(K), dim=seq)

        Delta rule update:
            M_new = M + σ(K)^T @ (V - σ(K) @ M / (σ(K) @ z))
            z_new = z + sum(σ(K), dim=seq)

        Args:
            key: Key tensor
            value: Value tensor
            M: Current memory matrix
            z: Current normalization vector

        Returns:
            Updated (M_new, z_new)
        """
        # Apply σ (ELU + 1) to key
        sigma_k = elu_plus_one(key)  # (B, H, T, D)

        if self.use_delta_rule:
            # Delta rule: prevents redundant storage
            # Retrieve what memory already knows about these keys
            # σ(K) @ M / (σ(K) @ z) -> what memory predicts for these keys
            mem_prediction = torch.matmul(sigma_k, M)  # (B, H, T, D)
            normalizer = torch.matmul(sigma_k, z)  # (B, H, T, 1)
            mem_prediction = safe_divide(mem_prediction, normalizer, self.eps)

            # Delta: what's new that memory doesn't know
            delta_v = value - mem_prediction  # (B, H, T, D)

            # Update memory with delta
            # σ(K)^T @ delta_v: (B, H, D, T) @ (B, H, T, D) -> (B, H, D, D)
            M_new = M + torch.matmul(sigma_k.transpose(-2, -1), delta_v)
        else:
            # Standard associative binding update
            # σ(K)^T @ V: (B, H, D, T) @ (B, H, T, D) -> (B, H, D, D)
            M_new = M + torch.matmul(sigma_k.transpose(-2, -1), value)

        # Update normalization: sum over sequence dimension
        # sum(σ(K), dim=seq): (B, H, T, D) -> (B, H, D) -> (B, H, D, 1)
        z_new = z + sigma_k.sum(dim=2, keepdim=True).transpose(-2, -1)

        return M_new, z_new


class InfiniAttention(DeviceAwareModule):
    """
    Infini-attention: Multi-head attention with compressive memory.

    Combines:
    1. Standard scaled dot-product attention (local, within segment)
    2. Linear attention over compressive memory (global, across segments)

    The outputs are combined via a learned gating mechanism:
        output = sigmoid(β) * A_mem + (1 - sigmoid(β)) * A_local

    This enables processing of infinitely long contexts with bounded
    memory and computation.
    """

    def __init__(self, config: InfiniAttentionConfig):
        super().__init__()

        assert config.embed_dim % config.num_heads == 0, \
            f"embed_dim ({config.embed_dim}) must be divisible by num_heads ({config.num_heads})"

        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.segment_size = config.segment_size
        self.use_rope = config.use_rope
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Compressive memory
        self.memory = CompressiveMemory(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_delta_rule=config.use_delta_rule
        )

        # Learned gating parameter β (one per head)
        # Initialize near 0 for equal weighting of local and memory attention
        self.gate_beta = nn.Parameter(
            torch.full((1, self.num_heads, 1, 1), config.init_gate_bias)
        )

        # Causal mask for local attention
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer('causal_mask', mask.view(1, 1, config.max_seq_len, config.max_seq_len))

        # Optional RoPE
        if config.use_rope:
            from model import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        else:
            self.rope = None

    def _compute_local_attention(
        self,
        q: torch.Tensor,  # (B, H, T, D)
        k: torch.Tensor,  # (B, H, T, D)
        v: torch.Tensor,  # (B, H, T, D)
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute standard scaled dot-product attention (local, within segment).

        A_local = softmax(Q @ K^T / sqrt(d)) @ V
        """
        B, H, T, D = q.shape

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply causal mask
        if T <= self.causal_mask.size(-1):
            attn_weights = attn_weights.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float('-inf')
            )

        # Apply external attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask[:, None, None, :] == 0, float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        return torch.matmul(attn_weights, v)  # (B, H, T, D)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_memory: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Infini-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
            memory_state: Optional (M, z) tuple from previous segment
            return_memory: Whether to return updated memory state

        Returns:
            output: Attention output of shape (batch, seq_len, embed_dim)
            memory_state: Updated (M, z) if return_memory=True, else None
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.rope is not None:
            from model import apply_rotary_emb
            cos, sin = self.rope(x, T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Initialize or use provided memory state
        if memory_state is None:
            M, z = self.memory.init_memory(B, x.device, x.dtype)
        else:
            M, z = memory_state

        # Retrieve from memory BEFORE updating (use previous state)
        A_mem = self.memory.retrieve(q, M, z)  # (B, H, T, D)

        # Compute local attention (standard dot-product)
        A_local = self._compute_local_attention(q, k, v, attention_mask)  # (B, H, T, D)

        # Update memory with current segment's K, V
        M_new, z_new = self.memory.update(k, v, M, z)

        # Combine local and memory attention via learned gate
        # gate = sigmoid(β)
        gate = torch.sigmoid(self.gate_beta)  # (1, H, 1, 1)

        # output = gate * A_mem + (1 - gate) * A_local
        combined = gate * A_mem + (1.0 - gate) * A_local  # (B, H, T, D)

        # Reshape and project
        combined = combined.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        output = self.resid_dropout(self.out_proj(combined))

        if return_memory:
            return output, (M_new, z_new)
        return output, None

    def forward_segments(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        initial_memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process input in segments, maintaining memory across segments.

        This enables processing of arbitrarily long sequences with
        bounded memory by breaking them into fixed-size segments.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
            initial_memory: Optional initial memory state

        Returns:
            output: Full attention output
            final_memory: Memory state after processing all segments
        """
        B, T, C = x.shape

        # Split into segments
        num_segments = (T + self.segment_size - 1) // self.segment_size
        outputs = []
        memory_state = initial_memory

        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_size
            end = min(start + self.segment_size, T)

            # Get segment
            x_seg = x[:, start:end, :]

            # Get segment mask if provided
            mask_seg = None
            if attention_mask is not None:
                mask_seg = attention_mask[:, start:end]

            # Process segment
            out_seg, memory_state = self.forward(
                x_seg,
                attention_mask=mask_seg,
                memory_state=memory_state,
                return_memory=True
            )
            outputs.append(out_seg)

        # Concatenate all segment outputs
        output = torch.cat(outputs, dim=1)

        return output, memory_state

    def get_memory_size(self) -> int:
        """Get the memory size in number of parameters per batch item."""
        # M: (heads, d_key, d_value) + z: (heads, d_key, 1)
        return self.num_heads * self.head_dim * (self.head_dim + 1)

    def get_compression_ratio(self, seq_len: int) -> float:
        """
        Calculate memory compression ratio vs storing full KV cache.

        For standard attention, we need to store:
            seq_len * 2 * heads * head_dim (K and V)

        For Infini-attention, we store:
            heads * head_dim * head_dim (M) + heads * head_dim (z)

        Returns:
            Compression ratio (higher is better)
        """
        standard_memory = seq_len * 2 * self.num_heads * self.head_dim
        infini_memory = self.get_memory_size()
        return standard_memory / infini_memory


class InfiniTransformerBlock(DeviceAwareModule):
    """
    Transformer block with Infini-attention.

    Pre-norm architecture:
        x = x + InfiniAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        segment_size: int = 128,
        use_delta_rule: bool = True,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        # Layer norms
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Infini-attention
        attn_config = InfiniAttentionConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            segment_size=segment_size,
            use_delta_rule=use_delta_rule
        )
        self.attn = InfiniAttention(attn_config)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_memory: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            memory_state: Optional memory state from previous segment
            return_memory: Whether to return updated memory state

        Returns:
            output: Block output
            memory_state: Updated memory if return_memory=True
        """
        # Pre-norm attention with residual
        attn_out, new_memory = self.attn(
            self.ln1(x),
            attention_mask=attention_mask,
            memory_state=memory_state,
            return_memory=return_memory
        )
        x = x + attn_out

        # Pre-norm FFN with residual
        x = x + self.ff(self.ln2(x))

        return x, new_memory

    def forward_segments(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        initial_memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process in segments, maintaining memory across segments."""
        B, T, C = x.shape
        segment_size = self.attn.segment_size

        num_segments = (T + segment_size - 1) // segment_size
        outputs = []
        memory_state = initial_memory

        for seg_idx in range(num_segments):
            start = seg_idx * segment_size
            end = min(start + segment_size, T)

            x_seg = x[:, start:end, :]
            mask_seg = attention_mask[:, start:end] if attention_mask is not None else None

            out_seg, memory_state = self.forward(
                x_seg,
                attention_mask=mask_seg,
                memory_state=memory_state,
                return_memory=True
            )
            outputs.append(out_seg)

        return torch.cat(outputs, dim=1), memory_state


class InfiniMemoryManager:
    """
    Manages memory states across layers for multi-layer Infini-attention models.

    This enables efficient streaming inference where memory is maintained
    across multiple forward passes.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.memory_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_memory(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get memory state for a layer."""
        return self.memory_states.get(layer_idx)

    def set_memory(self, layer_idx: int, memory: Tuple[torch.Tensor, torch.Tensor]):
        """Set memory state for a layer."""
        self.memory_states[layer_idx] = memory

    def clear(self):
        """Clear all memory states."""
        self.memory_states.clear()

    def get_all_memories(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Get all memory states."""
        return self.memory_states.copy()

    def set_all_memories(self, memories: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """Set all memory states."""
        self.memory_states = memories.copy()

    def detach(self):
        """Detach all memory states from computation graph."""
        for layer_idx in self.memory_states:
            M, z = self.memory_states[layer_idx]
            self.memory_states[layer_idx] = (M.detach(), z.detach())

    def to(self, device: torch.device):
        """Move all memory states to device."""
        for layer_idx in self.memory_states:
            M, z = self.memory_states[layer_idx]
            self.memory_states[layer_idx] = (M.to(device), z.to(device))

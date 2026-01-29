"""
Minimal GPT-style transformer model.
Supports MPS (Apple Silicon), CUDA, and CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 2048
    max_seq_len: int = 128
    embed_dim: int = 192
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 384
    dropout: float = 0.1
    use_rope: bool = False
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Infini-attention settings (for infinite context)
    use_infini_attention: bool = False
    segment_size: int = 128  # Local attention window for Infini-attention
    use_delta_rule: bool = True  # Use delta rule for memory updates


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._precompute(max_seq_len)

    def _precompute(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to x."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:, :d].unsqueeze(0).unsqueeze(0)
    sin = sin[:, :d].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 use_rope: bool = False, max_seq_len: int = 2048):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            cos, sin = self.rope(x, T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise FFN with GELU."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1,
                 use_rope: bool = False, max_seq_len: int = 2048, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout, use_rope, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """GPT-style decoder-only transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = None if config.use_rope else nn.Embedding(config.max_seq_len, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim, num_heads=config.num_heads, ff_dim=config.ff_dim,
                dropout=config.dropout, use_rope=config.use_rope, max_seq_len=config.max_seq_len,
                layer_norm_eps=config.layer_norm_eps
            ) for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.lm_head = None if config.tie_embeddings else nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_embed(input_ids)
        if self.pos_embed is not None:
            x = x + self.pos_embed(torch.arange(T, device=device))
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        self.last_hidden_states = x

        if self.lm_head is not None:
            return self.lm_head(x)
        return F.linear(x, self.token_embed.weight)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, tokenizer, prompt: str = "", max_tokens: int = 100, temperature: float = 0.7,
                 top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.05,
                 stop_token_id: Optional[int] = None, device: Optional[torch.device] = None) -> str:
        """Generate text from prompt."""
        if device is None:
            device = next(self.parameters()).device
        if stop_token_id is None:
            stop_token_id = tokenizer.eos_token_id

        self.eval()

        input_ids = [tokenizer.bos_token_id]
        if prompt:
            input_ids.extend(tokenizer.encode(prompt, add_special=False))
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            idx = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx)[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                sorted_mask = cumsum > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == stop_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return tokenizer.decode(input_ids[0].tolist())


class InfiniGPT(nn.Module):
    """
    GPT-style transformer with Infini-attention for infinite context.

    This model uses compressive memory to handle arbitrarily long sequences
    with bounded memory and computation. It's fully compatible with
    Huawei Ascend NPUs, NVIDIA CUDA, Apple MPS, and CPU.

    Key features:
    - Segment-level streaming computation
    - Compressive memory for long-term context
    - Learned gating between local and memory attention
    - Delta rule for stable memory updates
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Import here to avoid circular imports
        from infini_attention import InfiniTransformerBlock, InfiniMemoryManager

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = None if config.use_rope else nn.Embedding(config.max_seq_len, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Infini-attention transformer blocks
        self.blocks = nn.ModuleList([
            InfiniTransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                use_rope=config.use_rope,
                max_seq_len=config.max_seq_len,
                segment_size=config.segment_size,
                use_delta_rule=config.use_delta_rule,
                layer_norm_eps=config.layer_norm_eps
            ) for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        # Language model head
        self.lm_head = None if config.tie_embeddings else nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Memory manager for streaming inference
        self.memory_manager = InfiniMemoryManager(config.num_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional memory state management.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            use_memory: Whether to use stored memory from previous segments
            update_memory: Whether to update memory with current segment

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embed(input_ids)

        # Add positional embeddings if not using RoPE
        if self.pos_embed is not None:
            positions = torch.arange(T, device=device)
            x = x + self.pos_embed(positions)

        x = self.embed_dropout(x)

        # Process through Infini-attention blocks
        for layer_idx, block in enumerate(self.blocks):
            # Get memory state for this layer
            memory_state = None
            if use_memory:
                memory_state = self.memory_manager.get_memory(layer_idx)

            # Forward through block
            x, new_memory = block(
                x,
                attention_mask=attention_mask,
                memory_state=memory_state,
                return_memory=update_memory
            )

            # Update memory if requested
            if update_memory and new_memory is not None:
                self.memory_manager.set_memory(layer_idx, new_memory)

        # Final layer norm
        x = self.ln_f(x)
        self.last_hidden_states = x

        # Language model head
        if self.lm_head is not None:
            return self.lm_head(x)
        return F.linear(x, self.token_embed.weight)

    def forward_segments(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reset_memory: bool = True
    ) -> torch.Tensor:
        """
        Process long sequence by splitting into segments.

        This method explicitly handles segment-level streaming for
        very long sequences.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            reset_memory: Whether to clear memory before processing

        Returns:
            logits: Output logits
        """
        if reset_memory:
            self.memory_manager.clear()

        B, T = input_ids.shape
        segment_size = self.config.segment_size

        # Process in segments
        all_logits = []
        for start in range(0, T, segment_size):
            end = min(start + segment_size, T)
            segment_ids = input_ids[:, start:end]
            segment_mask = attention_mask[:, start:end] if attention_mask is not None else None

            logits = self.forward(
                segment_ids,
                attention_mask=segment_mask,
                use_memory=True,
                update_memory=True
            )
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)

    def reset_memory(self):
        """Clear all memory states for fresh inference."""
        self.memory_manager.clear()

    def detach_memory(self):
        """Detach memory from computation graph (for training)."""
        self.memory_manager.detach()

    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        if not self.blocks:
            return {}

        # Get stats from first block
        first_block = self.blocks[0]
        return {
            "memory_size_per_layer": first_block.attn.get_memory_size(),
            "num_layers": len(self.blocks),
            "total_memory_params": first_block.attn.get_memory_size() * len(self.blocks),
            "segment_size": self.config.segment_size,
        }

    def get_compression_ratio(self, seq_len: int) -> float:
        """Get memory compression ratio for given sequence length."""
        if not self.blocks:
            return 1.0
        return self.blocks[0].attn.get_compression_ratio(seq_len)

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt: str = "",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        stop_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        reset_memory: bool = True
    ) -> str:
        """
        Generate text with infinite context capability.

        Unlike standard transformers, this can maintain context from
        arbitrarily long prefixes via compressive memory.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            prompt: Initial prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            stop_token_id: Token ID to stop generation
            device: Target device
            reset_memory: Whether to clear memory before generation

        Returns:
            Generated text including prompt
        """
        if device is None:
            device = next(self.parameters()).device
        if stop_token_id is None:
            stop_token_id = tokenizer.eos_token_id

        self.eval()

        if reset_memory:
            self.reset_memory()

        # Encode prompt
        input_ids = [tokenizer.bos_token_id]
        if prompt:
            input_ids.extend(tokenizer.encode(prompt, add_special=False))
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        # If prompt is long, process in segments to build up memory
        if input_ids.size(1) > self.config.segment_size:
            # Process all but last segment to build memory
            num_full_segments = (input_ids.size(1) - 1) // self.config.segment_size
            for seg_idx in range(num_full_segments):
                start = seg_idx * self.config.segment_size
                end = start + self.config.segment_size
                _ = self.forward(
                    input_ids[:, start:end],
                    use_memory=True,
                    update_memory=True
                )

        # Generate tokens
        for _ in range(max_tokens):
            # Use last segment_size tokens for local attention
            # but memory contains info from all previous tokens
            idx = input_ids[:, -self.config.segment_size:]

            logits = self.forward(idx, use_memory=True, update_memory=True)
            logits = logits[:, -1, :]  # Last token prediction

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()[-100:]):  # Last 100 tokens
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                sorted_mask = cumsum > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == stop_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return tokenizer.decode(input_ids[0].tolist())


def create_model(config: ModelConfig, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create model from config.

    Returns InfiniGPT if use_infini_attention is True, otherwise MiniGPT.
    """
    if config.use_infini_attention:
        model = InfiniGPT(config)
    else:
        model = MiniGPT(config)

    if device is not None:
        model = model.to(device)
    return model

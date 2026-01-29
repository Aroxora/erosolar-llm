#!/usr/bin/env python3
"""
Tests for Infini-attention implementation.

Verifies:
1. Compressive memory operations
2. InfiniAttention module
3. InfiniGPT model
4. Huawei NPU compatibility layer
5. Memory compression ratios
"""

import torch
import torch.nn as nn
import sys


def test_huawei_npu_compatibility():
    """Test Huawei NPU compatibility layer."""
    print("Testing Huawei NPU compatibility layer...")

    from huawei_npu import (
        is_npu_available,
        get_device,
        get_device_info,
        elu_plus_one,
        safe_divide,
        setup_device
    )

    # Test device detection
    device = get_device()
    info = get_device_info(device)
    print(f"  Device: {info.device_type} ({info.device_name})")

    # Test ELU + 1 activation
    x = torch.randn(2, 4, 8)
    result = elu_plus_one(x)
    assert result.shape == x.shape, "ELU+1 shape mismatch"
    assert (result >= 0).all(), "ELU+1 should be non-negative"

    # Test safe division
    num = torch.ones(4, 4)
    denom = torch.zeros(4, 4)
    result = safe_divide(num, denom, eps=1e-6)
    assert not torch.isnan(result).any(), "Safe divide should not produce NaN"
    assert not torch.isinf(result).any(), "Safe divide should not produce Inf"

    print("  Huawei NPU compatibility layer: PASSED\n")


def test_compressive_memory():
    """Test CompressiveMemory class."""
    print("Testing CompressiveMemory...")

    from infini_attention import CompressiveMemory

    batch_size = 2
    num_heads = 4
    head_dim = 16
    seq_len = 32

    memory = CompressiveMemory(
        num_heads=num_heads,
        head_dim=head_dim,
        use_delta_rule=True
    )

    # Initialize memory
    M, z = memory.init_memory(batch_size, torch.device('cpu'))
    assert M.shape == (batch_size, num_heads, head_dim, head_dim), f"M shape: {M.shape}"
    assert z.shape == (batch_size, num_heads, head_dim, 1), f"z shape: {z.shape}"
    assert (M == 0).all(), "Initial M should be zeros"
    assert (z == 0).all(), "Initial z should be zeros"

    # Test retrieval from empty memory
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    retrieved = memory.retrieve(query, M, z)
    assert retrieved.shape == (batch_size, num_heads, seq_len, head_dim)

    # Test memory update
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    M_new, z_new = memory.update(key, value, M, z)
    assert M_new.shape == M.shape
    assert z_new.shape == z.shape
    assert not (M_new == 0).all(), "Updated M should not be all zeros"
    assert not (z_new == 0).all(), "Updated z should not be all zeros"

    # Test retrieval from updated memory
    retrieved_new = memory.retrieve(query, M_new, z_new)
    assert retrieved_new.shape == (batch_size, num_heads, seq_len, head_dim)

    print("  CompressiveMemory: PASSED\n")


def test_infini_attention():
    """Test InfiniAttention module."""
    print("Testing InfiniAttention...")

    from infini_attention import InfiniAttention, InfiniAttentionConfig

    config = InfiniAttentionConfig(
        embed_dim=64,
        num_heads=4,
        dropout=0.0,
        segment_size=16,
        use_delta_rule=True,
        max_seq_len=64
    )

    attn = InfiniAttention(config)
    attn.eval()

    batch_size = 2
    seq_len = 32

    x = torch.randn(batch_size, seq_len, config.embed_dim)

    # Test forward without memory
    with torch.no_grad():
        output, memory = attn(x, return_memory=True)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert memory is not None, "Memory should be returned"
    M, z = memory
    assert M.shape == (batch_size, config.num_heads, 16, 16)
    assert z.shape == (batch_size, config.num_heads, 16, 1)

    # Test forward with memory
    with torch.no_grad():
        output2, memory2 = attn(x, memory_state=memory, return_memory=True)

    assert output2.shape == x.shape

    # Test segment processing
    with torch.no_grad():
        full_output, final_memory = attn.forward_segments(x)

    assert full_output.shape == x.shape

    # Test compression ratio
    compression = attn.get_compression_ratio(1000)
    assert compression > 1, f"Compression ratio should be > 1, got {compression}"
    print(f"  Compression ratio at 1K tokens: {compression:.1f}x")

    compression_10k = attn.get_compression_ratio(10000)
    print(f"  Compression ratio at 10K tokens: {compression_10k:.1f}x")

    print("  InfiniAttention: PASSED\n")


def test_infini_transformer_block():
    """Test InfiniTransformerBlock."""
    print("Testing InfiniTransformerBlock...")

    from infini_attention import InfiniTransformerBlock

    block = InfiniTransformerBlock(
        embed_dim=64,
        num_heads=4,
        ff_dim=256,
        dropout=0.0,
        segment_size=16,
        use_delta_rule=True,
        max_seq_len=64
    )
    block.eval()

    batch_size = 2
    seq_len = 32

    x = torch.randn(batch_size, seq_len, 64)

    with torch.no_grad():
        output, memory = block(x, return_memory=True)

    assert output.shape == x.shape
    assert memory is not None

    # Test segment processing
    with torch.no_grad():
        full_output, final_memory = block.forward_segments(x)

    assert full_output.shape == x.shape

    print("  InfiniTransformerBlock: PASSED\n")


def test_infini_gpt():
    """Test InfiniGPT model."""
    print("Testing InfiniGPT...")

    from model import InfiniGPT, ModelConfig

    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.0,
        use_infini_attention=True,
        segment_size=16,
        use_delta_rule=True
    )

    model = InfiniGPT(config)
    model.eval()

    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Test forward
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Logits shape: {logits.shape}"

    # Test forward with memory management
    model.reset_memory()
    with torch.no_grad():
        logits1 = model(input_ids[:, :16], use_memory=True, update_memory=True)
        logits2 = model(input_ids[:, 16:], use_memory=True, update_memory=True)

    assert logits1.shape == (batch_size, 16, config.vocab_size)
    assert logits2.shape == (batch_size, 16, config.vocab_size)

    # Test forward_segments
    model.reset_memory()
    with torch.no_grad():
        full_logits = model.forward_segments(input_ids)

    assert full_logits.shape == (batch_size, seq_len, config.vocab_size)

    # Test memory stats
    stats = model.get_memory_stats()
    assert "memory_size_per_layer" in stats
    assert "total_memory_params" in stats

    # Test compression ratio
    compression = model.get_compression_ratio(10000)
    print(f"  Compression ratio at 10K tokens: {compression:.1f}x")
    assert compression > 10, f"Expected high compression, got {compression:.1f}x"

    print(f"  Model parameters: {model.get_num_params():,}")
    print("  InfiniGPT: PASSED\n")


def test_create_model():
    """Test create_model function with Infini-attention."""
    print("Testing create_model with Infini-attention...")

    from model import create_model, ModelConfig, InfiniGPT, MiniGPT

    # Test standard model
    config_standard = ModelConfig(
        vocab_size=1000,
        max_seq_len=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        use_infini_attention=False
    )
    model_standard = create_model(config_standard)
    assert isinstance(model_standard, MiniGPT), "Should create MiniGPT"

    # Test Infini-attention model
    config_infini = ModelConfig(
        vocab_size=1000,
        max_seq_len=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        use_infini_attention=True,
        segment_size=16
    )
    model_infini = create_model(config_infini)
    assert isinstance(model_infini, InfiniGPT), "Should create InfiniGPT"

    print("  create_model: PASSED\n")


def test_config_presets():
    """Test Infini-attention presets in config."""
    print("Testing Infini-attention presets...")

    from config import get_preset, PRESETS

    # Check Infini presets exist
    infini_presets = [k for k in PRESETS.keys() if k.startswith("infini")]
    assert len(infini_presets) >= 4, f"Expected at least 4 infini presets, found {len(infini_presets)}"
    print(f"  Found {len(infini_presets)} Infini-attention presets: {infini_presets}")

    # Test each preset
    for preset_name in infini_presets:
        config = get_preset(preset_name)
        assert config.use_infini_attention, f"{preset_name} should have use_infini_attention=True"
        assert config.segment_size > 0, f"{preset_name} should have segment_size > 0"
        print(f"  {preset_name}: segment_size={config.segment_size}, embed_dim={config.embed_dim}")

    print("  Config presets: PASSED\n")


def test_memory_manager():
    """Test InfiniMemoryManager."""
    print("Testing InfiniMemoryManager...")

    from infini_attention import InfiniMemoryManager

    num_layers = 4
    manager = InfiniMemoryManager(num_layers)

    # Test initial state
    for i in range(num_layers):
        assert manager.get_memory(i) is None

    # Test setting memory
    batch_size = 2
    num_heads = 4
    head_dim = 16

    for i in range(num_layers):
        M = torch.randn(batch_size, num_heads, head_dim, head_dim)
        z = torch.randn(batch_size, num_heads, head_dim, 1)
        manager.set_memory(i, (M, z))

    # Test getting memory
    for i in range(num_layers):
        memory = manager.get_memory(i)
        assert memory is not None
        M, z = memory
        assert M.shape == (batch_size, num_heads, head_dim, head_dim)

    # Test clear
    manager.clear()
    for i in range(num_layers):
        assert manager.get_memory(i) is None

    print("  InfiniMemoryManager: PASSED\n")


def test_long_sequence_handling():
    """Test handling of long sequences with segment processing."""
    print("Testing long sequence handling...")

    from model import InfiniGPT, ModelConfig

    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=256,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.0,
        use_infini_attention=True,
        segment_size=32,
        use_delta_rule=True
    )

    model = InfiniGPT(config)
    model.eval()

    # Test with sequence longer than segment_size
    batch_size = 1
    seq_len = 128  # 4 segments

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Process with segment streaming
    model.reset_memory()
    with torch.no_grad():
        logits = model.forward_segments(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # Verify memory was updated across segments
    memories = model.memory_manager.get_all_memories()
    assert len(memories) == config.num_layers, "Memory should be set for all layers"

    print(f"  Processed {seq_len} tokens in {seq_len // config.segment_size} segments")
    print("  Long sequence handling: PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Infini-attention Implementation Tests")
    print("=" * 60 + "\n")

    tests = [
        test_huawei_npu_compatibility,
        test_compressive_memory,
        test_infini_attention,
        test_infini_transformer_block,
        test_infini_gpt,
        test_create_model,
        test_config_presets,
        test_memory_manager,
        test_long_sequence_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

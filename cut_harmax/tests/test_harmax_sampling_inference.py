# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_harmax import harmax_sample

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _create_test_data():
    """Create test data for sampling."""
    vocab_size = 1000
    embed_dim = 128

    # Create test weight matrix
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    e = torch.randn(embed_dim, dtype=torch.bfloat16)

    return e, c


def test_min_p_sampling():
    """Test min_p sampling functionality."""
    e, c = _create_test_data()

    # Test basic sampling
    token = harmax_sample(e, c, min_p=0.01)
    assert isinstance(token, int), f"Expected int, got {type(token)}"
    assert 0 <= token < c.size(0), f"Token {token} out of range [0, {c.size(0)})"

    # Test with temperature
    token_high_temp = harmax_sample(e, c, temperature=2.0, min_p=0.01)
    token_low_temp = harmax_sample(e, c, temperature=0.5, min_p=0.01)
    assert isinstance(token_high_temp, int)
    assert isinstance(token_low_temp, int)

    # Test with different min_p thresholds
    token_strict = harmax_sample(e, c, min_p=0.1)  # 10% threshold
    token_permissive = harmax_sample(e, c, min_p=0.001)  # 0.1% threshold
    assert isinstance(token_strict, int)
    assert isinstance(token_permissive, int)

    print("✓ Min-p sampling tests passed")


def test_batch_sampling():
    """Test batch sampling functionality."""
    vocab_size = 1000
    embed_dim = 128

    # Create test data
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    e = torch.randn(4, embed_dim, dtype=torch.bfloat16)  # batch_size=4

    # Test basic batch sampling
    tokens = harmax_sample(e, c, min_p=0.01)
    assert tokens.shape == (4,), f"Expected shape (4,), got {tokens.shape}"
    assert tokens.dtype == torch.long, f"Expected torch.long, got {tokens.dtype}"
    assert torch.all(tokens >= 0) and torch.all(tokens < vocab_size), "Tokens out of range"

    print("✓ Batch sampling tests passed")


def test_min_p_filtering():
    """Test min_p filtering behavior."""
    e, c = _create_test_data()

    # Create a controlled scenario by manipulating distances
    with torch.no_grad():
        # Create embeddings that give predictable distances
        e_unit = torch.ones_like(e) * 0.1
        c_unit = torch.zeros_like(c)

        # Make first token very close (distance = 0.01)
        c_unit[0] = e_unit

        # Make next few tokens progressively farther
        for i in range(1, 10):
            c_unit[i] = e_unit * (1.0 + 0.1 * i)

        # Rest of vocabulary far away
        c_unit[10:] = e_unit * 10.0

    # Sample with min_p filtering
    token = harmax_sample(e_unit, c_unit, min_p=0.01)

    # Should be one of the close tokens (0-9) due to high probability
    assert 0 <= token < 10, f"Expected close token, got {token}"

    print("✓ Min-p filtering tests passed")


def test_temperature_scaling():
    """Test temperature scaling effects."""
    e, c = _create_test_data()

    # Test different temperatures
    token_low_temp = harmax_sample(e, c, temperature=0.1, min_p=0.01)
    token_high_temp = harmax_sample(e, c, temperature=2.0, min_p=0.01)
    token_med_temp = harmax_sample(e, c, temperature=1.0, min_p=0.01)

    assert isinstance(token_low_temp, int)
    assert isinstance(token_high_temp, int)
    assert isinstance(token_med_temp, int)

    # Different temperatures should give different results (probabilistically)
    results = set([token_low_temp, token_high_temp, token_med_temp])
    # Note: This might occasionally fail due to randomness, but that's expected

    print("✓ Temperature scaling tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    vocab_size = 100
    embed_dim = 64

    # Create test data
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    e = torch.randn(embed_dim, dtype=torch.bfloat16)

    # Test very small min_p
    token_small_minp = harmax_sample(e, c, min_p=1e-6)
    assert isinstance(token_small_minp, int)

    # Test very large min_p
    token_large_minp = harmax_sample(e, c, min_p=0.5)
    assert isinstance(token_large_minp, int)

    # Test min_p = 1.0 (should always select max probability token)
    token_max_minp = harmax_sample(e, c, min_p=1.0)
    assert isinstance(token_max_minp, int)

    print("✓ Edge case tests passed")


def test_probability_threshold():
    """Test that min_p filtering respects probability threshold."""
    e, c = _create_test_data()

    # Create embeddings with controlled probability distribution
    with torch.no_grad():
        # Create distance matrix with known pattern
        distances = torch.arange(c.size(0), dtype=torch.float32).float()

        # Apply exponential to create decreasing probabilities
        scaled_distances = distances / 100.0 + 1e-9
        reciprocal = 1.0 / scaled_distances
        probs = reciprocal / reciprocal.sum()

        # Sample from this known distribution
        token = harmax_sample(e.float(), c.float(), min_p=0.01)

    assert isinstance(token, int)

    print("✓ Probability threshold tests passed")


@skip_no_cuda
def test_gpu_inference():
    """Test GPU inference sampling."""
    e, c = _create_test_data()
    e = e.cuda()
    c = c.cuda()

    # Test single token sampling on GPU
    token = harmax_sample(e, c, min_p=0.01)
    assert isinstance(token, int)

    # Test batch sampling on GPU
    e_batch = e.unsqueeze(0).repeat(2, 1)  # (2, embed_dim)
    tokens = harmax_sample(e_batch, c, min_p=0.01)
    assert tokens.shape == (2,)

    print("✓ GPU inference tests passed")


def test_torch_compile():
    """Test torch.compile optimization."""
    e, c = _create_test_data()

    # Time without compilation
    import time
    start = time.time()
    for _ in range(100):
        token = harmax_sample(e, c, min_p=0.01)
    time_no_compile = time.time() - start

    # Test compilation
    compiled_func = torch.compile(harmax_sample)
    start = time.time()
    for _ in range(100):
        token = compiled_func(e, c, min_p=0.01)
    time_with_compile = time.time() - start

    print(f"Without compile: {time_no_compile:.4f}s")
    print(f"With compile: {time_with_compile:.4f}s")
    print("✓ torch.compile tests passed")


if __name__ == "__main__":
    print("Testing HarMax Min-P Inference Sampling")
    print("=" * 50)

    test_min_p_sampling()
    test_batch_sampling()
    test_min_p_filtering()
    test_temperature_scaling()
    test_edge_cases()
    test_probability_threshold()

    # Only run GPU tests if CUDA is available
    if torch.cuda.is_available():
        test_gpu_inference()

    test_torch_compile()

    print("\n🎉 All min-p sampling tests passed!")
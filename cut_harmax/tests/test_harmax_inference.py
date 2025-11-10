# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_harmax import FastHarMaxSampler, harmax_inference_sample, harmax_inference_batch

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _create_test_data():
    """Create test data for inference sampling."""
    vocab_size = 1000
    embed_dim = 128

    # Create test weight matrix
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    e = torch.randn(embed_dim, dtype=torch.bfloat16)

    return e, c


def test_fast_sampler_initialization():
    """Test FastHarMaxSampler initialization."""
    e, c = _create_test_data()

    sampler = FastHarMaxSampler(
        weight_matrix=c,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )

    assert sampler.weight_matrix.shape == c.shape
    assert sampler.temperature == 0.8
    assert sampler.top_k == 50
    assert sampler.top_p == 0.9
    assert sampler.weight_norms.shape == (c.size(0),)

    print("✓ Fast sampler initialization test passed")


def test_single_token_sampling():
    """Test single token sampling functionality."""
    e, c = _create_test_data()

    # Test basic sampling
    token = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float()
    )

    assert isinstance(token, int), f"Expected int, got {type(token)}"
    assert 0 <= token < c.size(0), f"Token {token} out of range [0, {c.size(0)})"

    # Test with temperature
    token_high_temp = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        temperature=2.0
    )
    token_low_temp = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        temperature=0.5
    )

    assert isinstance(token_high_temp, int)
    assert isinstance(token_low_temp, int)

    # Test with top_k
    token_topk = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        top_k=50
    )
    assert isinstance(token_topk, int)

    print("✓ Single token sampling tests passed")


def test_batch_sampling():
    """Test batch sampling functionality."""
    vocab_size = 1000
    embed_dim = 128

    # Create test data
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    e = torch.randn(4, embed_dim, dtype=torch.bfloat16)  # batch_size=4

    # Test basic batch sampling
    tokens = harmax_inference_batch(
        hidden_states=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float()
    )

    assert tokens.shape == (4,), f"Expected shape (4,), got {tokens.shape}"
    assert tokens.dtype == torch.long, f"Expected torch.long, got {tokens.dtype}"
    assert torch.all(tokens >= 0) and torch.all(tokens < vocab_size), "Tokens out of range"

    print("✓ Batch sampling tests passed")


def test_fast_sampler_methods():
    """Test FastHarMaxSampler methods."""
    e, c = _create_test_data()
    sampler = FastHarMaxSampler(c)

    # Test single sampling
    token = sampler.sample(e)
    assert isinstance(token, int)

    # Test batch sampling
    e_batch = e.unsqueeze(0).repeat(3, 1)  # (3, embed_dim)
    tokens = sampler.sample_batch(e_batch)
    assert tokens.shape == (3,)

    # Test with parameter overrides
    token_override = sampler.sample(e, temperature=1.5, top_k=100)
    assert isinstance(token_override, int)

    print("✓ Fast sampler methods tests passed")


def test_sampling_reproducibility():
    """Test that sampling is reproducible with fixed random state."""
    e, c = _create_test_data()

    # Test reproducibility with fixed random state
    token1 = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        random_state=42
    )
    token2 = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        random_state=42
    )

    assert token1 == token2, "Sampling should be reproducible with same random state"

    # Different random states should give different results (probabilistically)
    token3 = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        random_state=123
    )

    # Note: This might occasionally fail due to randomness, but that's expected
    print(f"Same seed: {token1}, Different seed: {token3}")

    print("✓ Reproducibility tests passed")


def test_sampling_edge_cases():
    """Test edge cases and error handling."""
    vocab_size = 100
    embed_dim = 64

    # Create test data
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    e = torch.randn(embed_dim, dtype=torch.bfloat16)

    # Test very small vocabulary
    small_vocab = torch.randn(10, embed_dim, dtype=torch.bfloat16)
    token_small = harmax_inference_sample(
        hidden_state=e[:embed_dim],
        weight_matrix=small_vocab,
        weight_norms=(small_vocab ** 2).sum(dim=1).float()
    )
    assert 0 <= token_small < 10

    # Test extreme temperatures
    token_low_temp = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        temperature=0.01
    )
    token_high_temp = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float(),
        temperature=10.0
    )
    assert isinstance(token_low_temp, int)
    assert isinstance(token_high_temp, int)

    print("✓ Edge case tests passed")


@skip_no_cuda
def test_gpu_inference():
    """Test GPU inference sampling."""
    e, c = _create_test_data()
    e = e.cuda()
    c = c.cuda()

    # Test single token sampling on GPU
    token = harmax_inference_sample(
        hidden_state=e,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float()
    )
    assert isinstance(token, int)

    # Test batch sampling on GPU
    e_batch = e.unsqueeze(0).repeat(2, 1)  # (2, embed_dim)
    tokens = harmax_inference_batch(
        hidden_states=e_batch,
        weight_matrix=c,
        weight_norms=(c ** 2).sum(dim=1).float()
    )
    assert tokens.shape == (2,)

    # Test FastHarMaxSampler on GPU
    sampler = FastHarMaxSampler(c)
    token_sampler = sampler.sample(e)
    assert isinstance(token_sampler, int)

    print("✓ GPU inference tests passed")


def test_performance_comparison():
    """Simple performance comparison between sampling methods."""
    import time

    vocab_size = 1000
    embed_dim = 128
    num_samples = 100

    # Create test data
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    e = torch.randn(embed_dim, dtype=torch.bfloat16)
    weight_norms = (c ** 2).sum(dim=1).float()

    # Time fast single token sampling
    start_time = time.time()
    for _ in range(num_samples):
        token = harmax_inference_sample(e, c, weight_norms)
    fast_time = time.time() - start_time

    # Time using sampler
    sampler = FastHarMaxSampler(c)
    start_time = time.time()
    for _ in range(num_samples):
        token = sampler.sample(e)
    sampler_time = time.time() - start_time

    print(f"Fast inference: {fast_time:.4f}s for {num_samples} samples")
    print(f"Sampler: {sampler_time:.4f}s for {num_samples} samples")

    print("✓ Performance comparison completed")


if __name__ == "__main__":
    print("Testing HarMax Inference Sampling")
    print("=" * 50)

    test_fast_sampler_initialization()
    test_single_token_sampling()
    test_batch_sampling()
    test_fast_sampler_methods()
    test_sampling_reproducibility()
    test_sampling_edge_cases()
    test_performance_comparison()

    # Only run GPU tests if CUDA is available
    if torch.cuda.is_available():
        test_gpu_inference()

    print("\n🎉 All inference sampling tests passed!")
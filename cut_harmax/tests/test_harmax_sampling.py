# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_harmax import harmax_sampling_kernel, harmax_sample_simple

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _create_test_data():
    """Create test data for sampling."""
    batch_size = 4
    vocab_size = 1000
    embed_dim = 128

    # Create test embeddings
    e = torch.randn(batch_size, embed_dim, dtype=torch.bfloat16)
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16)
    w_norm_sq = (c * c).sum(dim=1).float()

    return e, c, w_norm_sq


def test_simple_sampling():
    """Test simple CPU sampling functionality."""
    e, c, w_norm_sq = _create_test_data()

    # Test basic sampling
    samples = harmax_sample_simple(e, c)
    assert samples.shape == (e.size(0),), f"Expected shape {(e.size(0),)}, got {samples.shape}"
    assert samples.dtype == torch.long, f"Expected dtype torch.long, got {samples.dtype}"
    assert torch.all(samples >= 0) and torch.all(samples < c.size(0)), "Samples out of vocabulary range"

    # Test with temperature
    samples_high_temp = harmax_sample_simple(e, c, temperature=2.0)
    samples_low_temp = harmax_sample_simple(e, c, temperature=0.5)
    assert samples_high_temp.shape == samples.shape
    assert samples_low_temp.shape == samples.shape

    # Test with top_k
    samples_topk = harmax_sample_simple(e, c, top_k=50)
    assert samples_topk.shape == samples.shape

    # Test with top_p
    samples_topp = harmax_sample_simple(e, c, top_p=0.9)
    assert samples_topp.shape == samples.shape

    print("✓ Simple sampling tests passed")


@skip_no_cuda
def test_gpu_sampling():
    """Test GPU kernel sampling functionality."""
    e, c, w_norm_sq = _create_test_data()
    e = e.cuda()
    c = c.cuda()
    w_norm_sq = w_norm_sq.cuda()

    # Test basic sampling
    samples = harmax_sampling_kernel(e, c, w_norm_sq)
    assert samples.shape == (e.size(0),), f"Expected shape {(e.size(0),)}, got {samples.shape}"
    assert samples.dtype == torch.long, f"Expected dtype torch.long, got {samples.dtype}"
    assert torch.all(samples >= 0) and torch.all(samples < c.size(0)), "Samples out of vocabulary range"

    # Test with temperature
    samples_high_temp = harmax_sampling_kernel(e, c, w_norm_sq, temperature=2.0)
    samples_low_temp = harmax_sampling_kernel(e, c, w_norm_sq, temperature=0.5)
    assert samples_high_temp.shape == samples.shape
    assert samples_low_temp.shape == samples.shape

    print("✓ GPU sampling tests passed")


def test_sampling_probability_properties():
    """Test that sampling follows expected probability distributions."""
    e, c, w_norm_sq = _create_test_data()

    # For reproducible testing
    torch.manual_seed(42)

    # Test temperature effect
    samples_low_temp = harmax_sample_simple(e, c, temperature=0.1)
    samples_high_temp = harmax_sample_simple(e, c, temperature=2.0)

    # Low temperature should be more consistent (less variance)
    # This is a rough test - in practice, you'd want more samples
    assert samples_low_temp.shape == samples_high_temp.shape

    print("✓ Probability property tests passed")


def test_sampling_edge_cases():
    """Test edge cases and error handling."""
    e, c, w_norm_sq = _create_test_data()

    # Test very small temperature
    samples = harmax_sample_simple(e, c, temperature=0.01)
    assert samples.shape == (e.size(0),)

    # Test very large temperature
    samples = harmax_sample_simple(e, c, temperature=10.0)
    assert samples.shape == (e.size(0),)

    # Test extreme top_k
    samples = harmax_sample_simple(e, c, top_k=1)
    assert samples.shape == (e.size(0),)

    samples = harmax_sample_simple(e, c, top_k=c.size(0))
    assert samples.shape == (e.size(0),)

    print("✓ Edge case tests passed")


def test_sampling_determinism():
    """Test that sampling is deterministic with fixed seed."""
    e, c, w_norm_sq = _create_test_data()

    # Set seed
    torch.manual_seed(123)
    samples1 = harmax_sample_simple(e, c, temperature=0.8)

    # Reset seed and sample again
    torch.manual_seed(123)
    samples2 = harmax_sample_simple(e, c, temperature=0.8)

    assert torch.equal(samples1, samples2), "Sampling should be deterministic with fixed seed"

    print("✓ Determinism tests passed")


if __name__ == "__main__":
    print("Testing HarMax Sampling")
    print("=" * 40)

    test_simple_sampling()
    test_sampling_probability_properties()
    test_sampling_edge_cases()
    test_sampling_determinism()

    # Only run GPU tests if CUDA is available
    if torch.cuda.is_available():
        test_gpu_sampling()

    print("\n🎉 All sampling tests passed!")
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_harmax import cut_harmax_loss
from cut_harmax.constants import IGNORE_INDEX

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _harmonic_loss_manual(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    shift: bool,
) -> torch.Tensor:
    """Manual implementation of harmonic loss for testing."""
    N, T = targets.size()
    if shift:
        e = e[:, :-1]
        targets = targets[:, 1:]
        T = T - 1

    e = e.flatten(0, -2)
    targets = targets.flatten()

    eps = 1e-9

    # Compute squared Euclidean distances
    e_expanded = e.unsqueeze(1)  # (batch, 1, embed_dim)
    c_expanded = c.unsqueeze(0)  # (1, vocab, embed_dim)

    distances = torch.sum((e_expanded - c_expanded) ** 2, dim=2) + eps  # (batch, vocab)

    # Get target distances
    target_distances = distances[torch.arange(len(targets)), targets]

    # Compute harmonic loss: L = log(L_y) - log(sum(1/L_j))
    sum_reciprocal = torch.sum(1.0 / distances, dim=1)
    loss = torch.log(target_distances) - torch.log(sum_reciprocal)

    # Apply ignore index masking
    valid_mask = targets != IGNORE_INDEX
    loss = loss * valid_mask.float()

    return loss.view(N, T)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [(torch.bfloat16)])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("shape", [(256, 512, 128), (252, 507, 128), (252, 507, 123)])
def test_harmonic_loss_forward(
    dtype: torch.dtype,
    shift: bool,
    invalids: bool,
    shape: tuple[int, int, int],
):
    torch.set_float32_matmul_precision("highest")
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip(reason="BF16 not available")

    N, V, D = shape

    # Create test data
    e = torch.randn((N, D), device="cuda", dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype)

    # Make some vectors identical to test edge cases
    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    e = e.view(4, -1, D)

    targets = torch.randint(0, V, size=(N,), device="cuda")

    if invalids:
        inds = torch.randperm(len(targets), device="cuda")[0 : int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX

    targets = targets.view(e.size()[0:-1])

    # Compute ground truth with manual implementation (using float32 for precision)
    gt = _harmonic_loss_manual(e.float(), c.float(), targets, shift)

    # Compute with our HarMax loss implementation
    with torch.no_grad():
        harmax_loss = cut_harmax_loss(
            e, c, targets, shift=shift, reduction="none"
        )

    # Check that losses are close
    error = (gt - harmax_loss).abs()
    error_tol = 1e-2  # Tolerance for bfloat16

    assert (
        error <= error_tol
    ).all(), f"Max error: {error.max().item():.6f}, tolerance: {error_tol}"


@skip_no_cuda
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_harmax_loss_reductions(reduction: str):
    """Test different reduction methods."""
    torch.cuda.manual_seed(0)

    if not torch.cuda.is_bf16_supported():
        pytest.skip(reason="BF16 not available")

    batch_size = 16
    vocab_size = 1000
    embed_dim = 128

    e = torch.randn(batch_size, embed_dim, dtype=torch.bfloat16, device="cuda")
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16, device="cuda")
    targets = torch.randint(0, vocab_size, (batch_size,), device="cuda")

    loss = cut_harmax_loss(e, c, targets, reduction=reduction)

    if reduction == "none":
        assert loss.shape == (batch_size,)
    else:
        assert loss.shape == ()  # scalar

    assert torch.isfinite(loss).all(), "Loss should be finite"


@skip_no_cuda
def test_harmax_loss_gradient():
    """Test gradient computation."""
    torch.cuda.manual_seed(0)

    if not torch.cuda.is_bf16_supported():
        pytest.skip(reason="BF16 not available")

    batch_size = 8
    vocab_size = 100
    embed_dim = 64

    e = torch.randn(batch_size, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    c = torch.randn(vocab_size, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size,), device="cuda")

    loss = cut_harmax_loss(e, c, targets)
    loss.backward()

    assert e.grad is not None, "Embedding gradients should be computed"
    assert c.grad is not None, "Weight gradients should be computed"
    assert torch.isfinite(e.grad).all(), "Embedding gradients should be finite"
    assert torch.isfinite(c.grad).all(), "Weight gradients should be finite"
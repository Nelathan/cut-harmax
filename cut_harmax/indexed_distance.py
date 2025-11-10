# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import triton
import triton.language as tl

from cut_harmax.tl_autotune import indexed_dot_autotune
from cut_harmax.tl_utils import b_bin_fn


def _indexed_distance_forward_kernel(
    E,
    C,
    W_NORM_SQ,
    Inds,
    Valids,
    Out,
    B,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_ib,
    stride_vb,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    EVEN_D: tl.constexpr,
    SHIFT: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_d_chunks = tl.cdiv(D, BLOCK_D)
    num_d_in_group = GROUP_B * num_d_chunks
    group_id = pid // num_d_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_d_in_group) % group_size_b)
    pid_d = (pid % num_d_in_group) // group_size_b

    offs_b = (tl.arange(0, BLOCK_B) + pid_b * BLOCK_B) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_d = tl.arange(0, BLOCK_D) + pid_d * BLOCK_D
    e_ptrs = E + (stride_eb * offs_b[:, None] + stride_ed * offs_d[None, :])
    if EVEN_D:
        e = tl.load(e_ptrs)
    else:
        e = tl.load(e_ptrs, mask=offs_d[None, :] < D, other=0.0)

    inds = tl.load(Inds + stride_ib * ((offs_b + 1) if SHIFT else offs_b))

    c_ptrs = C + (inds[:, None] * stride_cv + offs_d[None, :] * stride_cd)
    if EVEN_D:
        c = tl.load(c_ptrs)
    else:
        c = tl.load(c_ptrs, mask=offs_d[None, :] < D, other=0.0)

    # Compute ||v||^2 for this block
    v_norm_sq = tl.sum(e * e, axis=1)

    # Get ||w_j||^2 for the target tokens
    w_norm_sq = tl.load(W_NORM_SQ + inds)

    # Compute dot product v · w_j
    dot_product = tl.sum(e * c, axis=1)

    # Compute squared Euclidean distance: L_j = ||v||^2 + ||w_j||^2 - 2(v·w_j)
    distance = v_norm_sq + w_norm_sq - 2.0 * dot_product
    distance = distance + EPS  # Add epsilon to prevent log(0)

    offs_b = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    out_ptrs = Out + offs_b
    tl.atomic_add(out_ptrs, distance.to(out_ptrs.dtype.element_ty), mask=offs_b < B)


_indexed_distance_forward_kernel = triton.jit(_indexed_distance_forward_kernel)
_indexed_distance_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "GROUP_B": lambda args: 8,
        "EPS": lambda args: 1e-9,
    }
)(_indexed_distance_forward_kernel)
_indexed_distance_forward_kernel = indexed_dot_autotune()(_indexed_distance_forward_kernel)  # type: ignore


def indexed_distance_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    w_norm_sq: torch.Tensor,
    inds: torch.Tensor,
    shift: bool = False,
    valids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute squared Euclidean distance between each embedding and its target token weight.

    Args:
        e: Input embeddings (batch_size, embedding_dim)
        c: Classifier/weight matrix (vocab_size, embedding_dim)
        w_norm_sq: Pre-computed ||w||^2 for all vocabulary items (vocab_size,)
        inds: Target indices (batch_size,)
        shift: Whether to shift targets (for causal language modeling)
        valids: Optional validity mask for batch elements

    Returns:
        Distance values for target tokens (batch_size,)
    """
    assert inds.ndim == 1
    assert e.ndim == 2
    assert c.ndim == 2
    assert inds.size(0) == e.size(0)
    assert c.size(1) == e.size(1)
    assert w_norm_sq.size(0) == c.size(0)
    assert e.dtype == torch.bfloat16, "HarMax loss requires bfloat16 inputs"
    assert c.dtype == torch.bfloat16, "HarMax loss requires bfloat16 weights"

    if valids is not None:
        assert valids.ndim == 1
        B = valids.size(0)
    else:
        B = e.size(0)

    out = e.new_zeros((B,), dtype=torch.float32)

    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(e.size(1), META["BLOCK_D"]),)

    _indexed_distance_forward_kernel[grid](
        e,
        c,
        w_norm_sq,
        inds,
        valids,
        out,
        B,
        e.size(1),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        inds.stride(0),
        1 if valids is None else valids.stride(0),
        B_BIN=b_bin_fn(B),
        SHIFT=shift,
    )

    return out
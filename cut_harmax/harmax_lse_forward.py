# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Literal, overload

import torch
import triton
import triton.language as tl

from cut_harmax.tl_autotune import cce_forward_autotune
from cut_harmax.tl_utils import b_bin_fn


def _harmax_lse_forward_kernel(
    E,
    C,
    V_NORM_SQ,
    LSE,
    Locks,
    Valids,
    B,
    V,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_lse_b,
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    GROUP_B: tl.constexpr,  #
    EVEN_D: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
    pid_v = (pid % num_pid_in_group) // group_size_b

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) % V
    offs_d = tl.arange(0, BLOCK_D)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    # Pre-compute ||v||^2 for this batch
    v_norm_sq = tl.zeros((BLOCK_B,), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            e_block = tl.load(e_ptrs)
        else:
            e_block = tl.load(e_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
        v_norm_sq += tl.sum(e_block * e_block, axis=1)
        e_ptrs += BLOCK_D * stride_ed

    # Load ||w_j||^2 for this vocabulary block
    w_norm_sq = tl.load(V_NORM_SQ + offs_v, mask=offs_v < V, other=0.0)

    # Reset pointers for dot product computation
    e_ptrs = E - (tl.cdiv(D, BLOCK_D) * BLOCK_D * stride_ed) + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    # Compute dot products v · w_j
    dot_products = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            e = tl.load(e_ptrs)
            c = tl.load(c_ptrs).to(e.dtype)
        else:
            e = tl.load(e_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
            c = tl.load(c_ptrs, mask=offs_d[:, None] < D - d * BLOCK_D, other=0.0).to(e.dtype)
        dot_products = tl.dot(e, c, dot_products, input_precision="ieee")
        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    # Compute squared Euclidean distances: L_j = ||v||^2 + ||w_j||^2 - 2(v·w_j)
    distances = v_norm_sq[:, None] + w_norm_sq[None, :] - 2.0 * dot_products
    distances = distances + EPS  # Add epsilon to prevent division by zero

    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) < V
    distances = tl.where(v_mask[None, :], distances, float("inf"))

    off_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    o_mask = off_b < B

    # Find minimum distance for stability (analogous to max in log-sum-exp)
    this_min = tl.min(distances, axis=1)

    # Compute stable sum of reciprocals: sum(L_min / L_j)
    stable_distances = distances / this_min[:, None]
    reciprocals = 1.0 / stable_distances
    sum_reciprocals = tl.sum(reciprocals, axis=1)

    # Compute harmonic loss: log(L_min) + log(sum(1/L_j)) - log(sum_reciprocals)
    # L = log(L_y) - log(sum(1/L_j))
    # But we use the stable formulation: L = log(L_min) + log(sum_reciprocals)
    harmonic_loss = tl.log(this_min) + tl.log(sum_reciprocals)

    lse_ptrs = LSE + (stride_lse_b * off_b)

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    current_loss = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    # Since we're doing reduction, we need to combine losses from different blocks
    # We use log-sum-exp again for the reduction across vocabulary blocks
    combined_loss = tl_logaddexp(current_loss, harmonic_loss)
    tl.store(lse_ptrs, combined_loss, mask=o_mask, eviction_policy="evict_last")

    tl.atomic_xchg(this_locks, 0)


def tl_logaddexp(a: tl.tensor, b: tl.tensor) -> tl.tensor:
    """Stable log-sum-exp operation for two tensors"""
    mx = tl.maximum(a, b)
    return mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))


_harmax_lse_forward_kernel = triton.jit(_harmax_lse_forward_kernel)
_harmax_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "GROUP_B": lambda args: 8,
        "EPS": lambda args: 1e-9,
    }
)(_harmax_lse_forward_kernel)
_harmax_lse_forward_kernel = cce_forward_autotune()(_harmax_lse_forward_kernel)  # type: ignore


def harmax_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    v_norm_sq: torch.Tensor,
    valids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Forward kernel for HarMax loss that computes log-sum-reciprocal normalization.

    Args:
        e: Input embeddings (batch_size, embedding_dim)
        c: Classifier/weight matrix (vocab_size, embedding_dim)
        v_norm_sq: Pre-computed ||v||^2 for all embeddings (batch_size,)
        valids: Optional validity mask for batch elements

    Returns:
        HarMax loss values per batch element
    """
    # Check constraints.
    assert e.shape[1] == c.shape[1], "Incompatible dimensions"
    assert e.is_contiguous(), "Matrix A must be contiguous"
    assert v_norm_sq.shape[0] == e.shape[0], "v_norm_sq batch size mismatch"
    assert e.dtype == torch.bfloat16, "HarMax loss requires bfloat16 inputs"
    assert c.dtype == torch.bfloat16, "HarMax loss requires bfloat16 weights"

    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B, _ = e.shape

    V, D = c.shape

    # Pre-compute ||w_j||^2 for all vocabulary items
    w_norm_sq = (c * c).sum(dim=1).float()  # (vocab_size,)

    # Allocates output.
    lse = e.new_full((B,), float("inf"), dtype=torch.float32)
    locks = e.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.uint32,
    )

    # 1D launch kernel where each block gets its own program.
    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _harmax_lse_forward_kernel[grid](
        e,
        c,
        w_norm_sq,
        lse,  #
        locks,
        valids,
        B,
        V,
        D,  #
        e.stride(0),
        e.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        lse.stride(0),
        1 if valids is None else valids.stride(0),
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
    )

    return lse
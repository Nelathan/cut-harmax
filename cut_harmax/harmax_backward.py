# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import triton
import triton.language as tl

from cut_harmax.tl_autotune import cce_backward_autotune
from cut_harmax.tl_utils import b_bin_fn, tl_lock_add


@triton.jit
def _mm_backward_harmonic(
    do,
    da_ptrs,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)[None, :]

    da_ptrs = da_ptrs + d_inds * stride_ad
    b_ptrs = b_ptrs + d_inds * stride_bd

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < (D - d * BLOCK_D))

        b = tl.load(b_ptrs, mask=mask, other=0.0).to(do.dtype)

        da_i = tl.dot(do, b).to(da_ptrs.dtype.element_ty)

        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < (D - d * BLOCK_D))

        lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        this_da_lock_ptr = da_lock_ptr + lock_offset

        tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)

        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad


@triton.jit
def _block_is_filtered(check_val: tl.tensor, filter_eps: tl.tensor) -> tl.tensor:
    return tl.reduce(check_val < filter_eps, None, tl.all)


def _harmax_backward_kernel(
    E,
    C,
    LSE,
    dOut,
    grad_scale,
    Valids,
    Targets,
    dE,
    dELocks,
    dC,
    dCLocks,
    B,
    D,
    V,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    filter_eps,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    FILTER_GRAD: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    SHIFT: tl.constexpr,
    REQUIRES_GRAD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_v_in_group) % group_size_b)
    pid_v = (pid % num_v_in_group) // group_size_b

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) % V
    offs_d = tl.arange(0, BLOCK_D)

    # Load embeddings and weights
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    # Load embeddings and weights for distance computation
    e_block = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    c_block = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            e_block_d = tl.load(e_ptrs)
            c_block_d = tl.load(c_ptrs).to(e_block_d.dtype)
        else:
            e_block_d = tl.load(e_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
            c_block_d = tl.load(c_ptrs, mask=offs_d[:, None] < D - d * BLOCK_D, other=0.0).to(e_block_d.dtype)

        e_block = tl.where(offs_d[None, :] < D - d * BLOCK_D, e_block_d, e_block)
        c_block = tl.where(offs_d[:, None] < D - d * BLOCK_D, c_block_d, c_block)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    # Compute dot products for distance calculation
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

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

    # Pre-compute ||v||^2
    v_norm_sq = tl.sum(e_block * e_block, axis=1)
    # Pre-compute ||w_j||^2
    w_norm_sq = tl.sum(c_block * c_block, axis=0)

    # Compute squared Euclidean distances: L_j = ||v||^2 + ||w_j||^2 - 2(v·w_j)
    distances = v_norm_sq[:, None] + w_norm_sq[None, :] - 2.0 * dot_products
    distances = distances + EPS  # Add epsilon to prevent division by zero

    if HAS_VALIDS:
        lse = tl.load(LSE + (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B)
    else:
        lse = tl.load(LSE + offs_b)

    # Find minimum distance for stable computation
    min_distances = tl.min(distances, axis=1)
    min_distances_stable = min_distances + EPS

    # Compute sum of reciprocals: S = sum(1/L_j)
    # Using stable computation: S = (1/L_min) * sum(L_min / L_j)
    stable_distances = distances / min_distances_stable[:, None]
    sum_reciprocals = tl.sum(1.0 / stable_distances, axis=1)
    S = (1.0 / min_distances_stable) * sum_reciprocals

    # Compute harmonic probabilities: p_j = (1/L_j) / S
    p_j = (1.0 / distances) / S[:, None]

    # Compute gradients: dL/dL_j
    if HAS_TARGETS:
        targets = tl.load(Targets + ((offs_b + 1) if SHIFT else offs_b))
        is_target = targets[:, None] == offs_v[None, :]

        # For correct token: dL/dL_j = (1 + p_j) / L_j
        # For incorrect token: dL/dL_j = p_j / L_j
        dL_dLj = tl.where(is_target, (1.0 + p_j) / distances, p_j / distances)
    else:
        dL_dLj = p_j / distances

    accum_valid_mask = ((pid_b * BLOCK_B + tl.arange(0, BLOCK_B))[:, None] < B) & (
        (pid_v * BLOCK_V + tl.arange(0, BLOCK_V))[None, :] < V
    )
    dL_dLj = tl.where(accum_valid_mask, dL_dLj, 0.0)

    if FILTER_GRAD:
        if _block_is_filtered(tl.abs(dL_dLj), filter_eps):
            return

    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        d_out = tl.load(dOut + ((offs_b + 1) if SHIFT else offs_b))[:, None]

    d_out = grad_scale * d_out

    dL_dLj = (dL_dLj * d_out).to(e_ptrs.dtype.element_ty)

    # Compute (v - w_j) for each pair
    # We need to reshape for broadcasting
    v_expanded = e_block[:, :, None]  # (BLOCK_B, BLOCK_D, 1)
    w_expanded = c_block[None, :, :]  # (1, BLOCK_V, BLOCK_D)
    v_minus_w = v_expanded - w_expanded  # (BLOCK_B, BLOCK_V, BLOCK_D)

    # For gradient wrt v: grad_v += dL/dL_j * 2 * (v - w_j)
    # We need to sum over vocabulary dimension
    grad_v = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    for j in range(BLOCK_V):
        if j < V - (pid_v * BLOCK_V):
            grad_v += dL_dLj[:, j:j+1] * 2.0 * v_minus_w[:, j, :]

    # For gradient wrt w_j: grad_w_j += dL/dL_j * (-2) * (v - w_j)
    # We need to sum over batch dimension
    grad_w = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)
    for i in range(BLOCK_B):
        if i < B - (pid_b * BLOCK_B):
            grad_w += dL_dLj[i:i+1, :].T * (-2.0) * v_minus_w[i, :, :]

    b_mask = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)[:, None]) < B
    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)[:, None]) < V

    lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1
    dELocks += lock_offset

    # Accumulate gradients for embeddings
    dE_block = tl.where(b_mask, grad_v, 0.0)
    dE_ptrs = dE + (offs_b[:, None] * stride_eb + tl.arange(0, BLOCK_D)[None, :] * stride_ed)

    for d_start in range(0, D, MM_BACK_BLOCK_D):
        d_end = min(d_start + MM_BACK_BLOCK_D, D)
        if MM_BACK_EVEN_D:
            mask = b_mask
        else:
            mask = b_mask & (tl.arange(0, MM_BACK_BLOCK_D)[None, :] < (d_end - d_start))

        this_lock = dELocks + (d_start // (MM_BACK_BLOCK_D * n_de_locks_1))
        tl_lock_add(dE_ptrs + d_start * stride_ed, dE_block[:, d_start:d_end], mask, this_lock)

    lock_offset = (pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1
    dCLocks += lock_offset

    if REQUIRES_GRAD:
        # Accumulate gradients for classifier weights
        dC_block = tl.where(v_mask, grad_w, 0.0)
        dC_ptrs = dC + (offs_v[:, None] * stride_cv + tl.arange(0, BLOCK_D)[None, :] * stride_cd)

        for d_start in range(0, D, MM_BACK_BLOCK_D):
            d_end = min(d_start + MM_BACK_BLOCK_D, D)
            if MM_BACK_EVEN_D:
                mask = v_mask
            else:
                mask = v_mask & (tl.arange(0, MM_BACK_BLOCK_D)[None, :] < (d_end - d_start))

            this_lock = dCLocks + (d_start // (MM_BACK_BLOCK_D * n_dc_locks_1))
            tl_lock_add(dC_ptrs + d_start * stride_cd, dC_block[:, d_start:d_end], mask, this_lock)


_harmax_backward_kernel = triton.jit(_harmax_backward_kernel)
_harmax_backward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: args["BLOCK_D"] * 2,
        "MM_BACK_EVEN_D": lambda args: (args["D"] % (args["BLOCK_D"] * 2)) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "FILTER_GRAD": lambda args: args["filter_eps"] is not None,
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "ITEM_DO": lambda args: args["dOut"].numel() == 1,
        "GROUP_B": lambda args: 8,
        "REQUIRES_GRAD" : lambda args: args["REQUIRES_GRAD"],
        "EPS": lambda args: 1e-9,
    }
)(_harmax_backward_kernel)
_harmax_backward_kernel = cce_backward_autotune()(_harmax_backward_kernel)  # type: ignore


def harmax_backward_kernel(
    do: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    lse: torch.Tensor,
    valids: torch.Tensor | None,
    targets: torch.Tensor | None = None,
    shift: bool = False,
    grad_scale: float = 1.0,
    filter_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert do.numel() in (e.size(0), 1)
    assert c.size(1) == e.size(1)
    assert lse.size(0) == e.size(0) or (valids is not None and lse.size(0) == valids.size(0))
    assert e.dtype == torch.bfloat16, "HarMax backward requires embeddings to be bfloat16"
    assert c.dtype == torch.bfloat16, "HarMax backward requires classifier to be bfloat16"

    do = do.contiguous()
    lse = lse.contiguous()

    de = torch.zeros_like(e)
    assert de.stride() == e.stride()

    if c.requires_grad:
        dc = torch.zeros_like(c)
        assert dc.stride() == c.stride()
        REQUIRES_GRAD = True
    else:
        dc = c
        REQUIRES_GRAD = False

    if valids is not None:
        assert valids.ndim == 1
        B = valids.size(0)
    else:
        B = e.size(0)

    if do.numel() > 1:
        do = do.contiguous()
        lse = lse.contiguous()
        assert do.stride(0) == lse.stride(0), f"{do.stride()=}, {lse.stride()=}"

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(c.size(0), META["BLOCK_V"]),)

    nd_locks = triton.cdiv(c.size(1), 64)
    de_locks = e.new_zeros((triton.cdiv(B, nd_locks), nd_locks), dtype=torch.int32)
    dc_locks = c.new_zeros((triton.cdiv(c.size(0), nd_locks), nd_locks), dtype=torch.int32)

    _harmonic_backward_kernel[grid](
        e,
        c,
        lse,
        do,
        grad_scale,
        valids,
        targets,
        de,
        de_locks,
        dc,
        dc_locks,
        B,
        e.size(1),
        c.size(0),
        de_locks.size(0),
        de_locks.size(1),
        dc_locks.size(0),
        dc_locks.size(1),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if valids is None else valids.stride(0),
        filter_eps,
        B_BIN=b_bin_fn(B),
        SHIFT=shift,
        REQUIRES_GRAD=REQUIRES_GRAD,
    )

    return de, dc.to(c.dtype) if REQUIRES_GRAD else None
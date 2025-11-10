# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import cast

import torch

from cut_harmax.harmax_backward import harmax_backward_kernel
from cut_harmax.harmax_lse_forward import harmax_lse_forward_kernel
from cut_harmax.indexed_distance import indexed_distance_forward_kernel
from cut_harmax.constants import IGNORE_INDEX
from cut_harmax.doc import LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_harmax.utils import (
    _build_flat_valids,
    handle_reduction_none,
)


@dataclass
class HarMaxParams:
    targets: torch.Tensor
    valids: torch.Tensor | None
    reduction: str
    shift: bool
    batch_shape: torch.Size


class HarMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        params: HarMaxParams,
    ) -> torch.Tensor:
        needs_grad = e.requires_grad or c.requires_grad

        # Pre-compute ||v||^2 for all embeddings
        v_norm_sq = (e * e).sum(dim=1).float()  # (batch_size,)

        # Compute HarMax LSE (log-sum-reciprocal normalization)
        harmax_lse = harmax_lse_forward_kernel(
            e,
            c,
            v_norm_sq,
            valids=params.valids,
        )

        # Compute distance for the correct token
        # Pre-compute ||w||^2 for all vocabulary items
        w_norm_sq = (c * c).sum(dim=1).float()  # (vocab_size,)
        target_distance = indexed_distance_forward_kernel(
            e, c, w_norm_sq, params.targets, params.shift, params.valids
        )

        # Final loss: L = log(L_y) - log(sum(1/L_j))
        # Note: harmax_lse already contains log(sum(1/L_j)) term
        nll = torch.log(target_distance) - harmax_lse

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        ctx.save_for_backward(e, c, harmax_lse, params.targets, params.valids, v_norm_sq, w_norm_sq)
        ctx.params = params

        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        h, w, harmax_lse, targets, valids, v_norm_sq, w_norm_sq = ctx.saved_tensors

        params = cast(HarMaxParams, ctx.params)
        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / harmax_lse.numel()
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
            grad_out = grad_out.view(-1)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        de, dc = harmax_backward_kernel(
            grad_out,
            h,
            w,
            harmax_lse,
            valids,
            targets=targets,
            shift=params.shift,
            grad_scale=grad_scale,
        )

        return de, dc, None


def harmax_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    params: HarMaxParams,
) -> torch.Tensor:
    loss = HarMaxFunction.apply(e, c, params)
    assert isinstance(loss, torch.Tensor)

    if params.shift and params.reduction == "none":
        loss = loss[..., 1:]

    return loss


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def cut_harmax_loss(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    shift: bool = False,
) -> torch.Tensor:
    """
    HarMax Loss using fused Triton kernels.

    This function computes the harmonic loss: L = -log((1/L_y) / sum_j(1/L_j))
    where L_j = ||v - w_j||^2 is the squared Euclidean distance between
    embedding v and weight vector w_j.

    This implementation uses the "Cut" technique to avoid materializing the
    full (batch_size, vocab_size) distance matrix in VRAM.

    Args:
        e: Input embeddings (..., embedding_dim) - must be bfloat16
        c: Classifier/weight matrix (vocab_size, embedding_dim) - must be bfloat16
        targets: Target indices (...)
        ignore_index: Index to ignore in loss computation
        reduction: Reduction method ('mean', 'sum', or 'none')
        shift: Whether to shift targets (for causal language modeling)

    Returns:
        HarMax loss tensor
    """
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)

    # Enforce bfloat16 for HarMax loss stability
    if e.dtype != torch.bfloat16:
        raise ValueError(f"HarMax loss requires bfloat16 inputs, got {e.dtype}")
    if c.dtype != torch.bfloat16:
        raise ValueError(f"HarMax loss requires bfloat16 weights, got {c.dtype}")

    batch_shape = targets.size()

    e = e.contiguous()
    targets = targets.contiguous()

    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()

    return harmax_apply(
        e,
        c,
        HarMaxParams(
            targets,
            valids,
            reduction,
            shift,
            batch_shape,
        ),
    )
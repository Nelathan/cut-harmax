# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
from torch.nn import functional as F


@torch.compile
def harmax_sample(
    hidden_states: torch.Tensor,
    weight_matrix: torch.Tensor,
    temperature: float = 0.8,
    min_p: float = 0.01,
) -> torch.Tensor:
    """
    Fast HarMax sampling for inference.

    Uses min_p filtering for creative writing - allows dominant tokens
    while maintaining entropy for flat distributions.

    Args:
        hidden_states: Input embeddings (batch_size, embed_dim)
        weight_matrix: Language model head (vocab_size, embed_dim)
        temperature: Sampling temperature
        min_p: Minimum probability threshold (default 0.01 = 1%)

    Returns:
        Sampled token indices (batch_size,)
    """
    eps = 1e-9

    # Compute squared Euclidean distances
    distances = torch.sum((hidden_states[:, None, :] - weight_matrix[None, :, :]) ** 2, dim=2)

    # Apply temperature scaling
    scaled_distances = distances / temperature + eps

      # HarMax normalization: p_j = (1/L_j) / sum(1/L_k)
    reciprocal_distances = 1.0 / scaled_distances
    probs = reciprocal_distances / reciprocal_distances.sum(dim=-1, keepdim=True)

    # Min-P filtering: eliminate tokens below probability threshold
    max_prob, _ = torch.max(probs, dim=-1, keepdim=True)
    min_threshold = max_prob * min_p
    mask = probs >= min_threshold
    probs = probs.masked_fill(~mask, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Sample from filtered distribution
    return torch.multinomial(probs, 1).squeeze(-1)
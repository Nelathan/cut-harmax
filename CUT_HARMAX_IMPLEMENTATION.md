# HarMax Loss Implementation - Cut HarMax

## Overview

This implementation adapts the "Cut" technique (originally developed for Cross-Entropy loss) to compute the **Harmonic Loss** efficiently using **HarMax** normalization. The new **"Cut HarMax"** kernels compute the exact, full-vocabulary harmonic loss without materializing the (batch_size, vocab_size) distance matrix.

## Project Structure

The implementation follows the same naming principle as the original CCE (Cut Cross-Entropy):

- **CCE** = Cut Cross-Entropy
- **HarMax** = Cut HarMax (Harmonic Loss with HarMax normalization)

```
cut_cross_entropy/     # Original Cross-Entropy implementation
cut_harmax/           # New HarMax implementation
├── harmax.py                    # Main API and autograd function
├── harmax_lse_forward.py        # Forward kernel (log-sum-reciprocal)
├── harmax_backward.py           # Backward kernel (harmonic gradients)
├── indexed_distance.py          # Target token distance extraction
├── constants.py                 # Constants (IGNORE_INDEX)
├── utils.py                     # Utility functions
├── tl_autotune.py               # Triton autotuning
├── tl_utils.py                  # Triton utilities
├── doc.py                       # Documentation utilities
└── tests/                       # Test suite
    └── test_harmonic_loss_forward.py
```

## Mathematical Foundation

### Harmonic Loss vs Cross-Entropy Loss

**Original CCE Loss:**
- **Operation**: Dot-product logits, $z_j = v \cdot w_j$
- **Loss**: $L_{\text{CCE}} = -z_y + \log\left(\sum_{j=1}^{N} e^{z_j}\right)$
- **Normalization**: SoftMax (log-sum-exp)

**HarMax Loss (New Implementation):**
- **Operation**: Squared Euclidean distance, $L_j = || v - w_j ||^2$
- **Loss**: $L_{\text{Harm}} = -\log\left(\frac{1/L_y}{\sum_{j=1}^{N} (1/L_j)}\right)$
- **Simplified**: $L_{\text{Harm}} = \log(L_y) - \log\left(\sum_{j=1}^{N} \frac{1}{L_j}\right)$
- **Normalization**: HarMax (log-sum-reciprocal)

### Key Differences

| Aspect | CCE | HarMax |
|--------|-----|--------|
| **Operation** | Dot product | Squared Euclidean distance |
| **Loss formula** | $-z_y + \log(\sum e^{z_j})$ | $\log(L_y) - \log(\sum 1/L_j)$ |
| **Normalization** | SoftMax | HarMax |
| **Stability trick** | Maximum logit | Minimum distance |
| **Gradient wrt logits** | $p_j - y_j$ | $p_j/L_j$ or $(1+p_j)/L_j$ |
| **Backprop pattern** | $(p_j - y_j) \cdot w_j$ | $dL/dL_j \cdot 2(v - w_j)$ |

## Implementation Files

### 1. `harmax.py`
**Purpose**: Main API and autograd function

**Key Features:**
- `HarMaxFunction`: Custom autograd Function with forward/backward passes
- `cut_harmax_loss`: Public API function
- `HarMaxParams`: Parameter dataclass for configuration
- Enforces bfloat16 input requirements
- Integrates with existing "Cut" infrastructure

### 2. `harmax_lse_forward.py`
**Purpose**: Forward kernel for HarMax loss computation

**Key Features:**
- Fused distance calculation using identity: $L_j = ||v||^2 + ||w_j||^2 - 2(v \cdot w_j)$
- Pre-computes `||v||^2` per batch and `||w_j||^2` per vocabulary
- Stable computation using minimum distance (instead of maximum logits)
- Epsilon (1e-9) addition to prevent division by zero
- Float32 accumulators for precision stability

**Stability Trick:**
```
S = sum(1/L_j) = (1/L_min) * sum(L_min / L_j)
log(S) = -log(L_min) + log(sum(L_min / L_j))
```

### 3. `harmax_backward.py`
**Purpose**: Backward kernel for HarMax gradient computation

**Key Features:**
- Computes HarMax probabilities: $p_j = \frac{1/L_j}{\sum(1/L_k)}$
- Gradient calculation:
  - For incorrect tokens: $\frac{\partial L}{\partial L_j} = \frac{p_j}{L_j}$
  - For correct token: $\frac{\partial L}{\partial L_y} = \frac{1 + p_y}{L_y}$
- Gradient backpropagation: $\nabla_v L_j = 2 \cdot (v - w_j)$, $\nabla_{w_j} L_j = -2 \cdot (v - w_j)$
- Major change from CCE: uses `(v - w_j)` instead of `w_j` and `v`

### 4. `indexed_distance.py`
**Purpose**: Utility to extract correct token distances

**Key Features:**
- Computes squared Euclidean distance for target tokens only
- Analogous to `indexed_neg_dot_forward_kernel` in original CCE
- Pre-computes norm terms for efficiency

## API Usage

```python
from cut_harmax import cut_harmax_loss

# Basic usage
loss = cut_harmax_loss(
    e=embeddings,           # (batch_size, embedding_dim) bfloat16
    c=weight_matrix,        # (vocab_size, embedding_dim) bfloat16
    targets=targets,        # (batch_size,) long
    reduction="mean",       # "mean", "sum", or "none"
    shift=False,            # for causal language modeling
)

# Gradient computation
embeddings.requires_grad_(True)
weight_matrix.requires_grad_(True)

loss = cut_harmax_loss(embeddings, weight_matrix, targets)
loss.backward()

# Gradients are available in embeddings.grad and weight_matrix.grad
```

## Precision and Stability

### Input Requirements
- **Mandatory**: bfloat16 inputs (prevents overflow in reciprocal operations)
- **Embeddings**: bfloat16
- **Classifier weights**: bfloat16

### Accumulator Precision
- **All reductions**: float32 (prevents catastrophic precision loss)
- **Dot products**: float32
- **Norm calculations**: float32
- **Final loss computation**: float32

### Stability Features
- **Epsilon**: 1e-9 added to all distance terms before log/reciprocal
- **Min-based stability**: Uses minimum distance for stable reciprocal computation
- **Proper masking**: Handles ignore_index and shift parameters correctly

## Memory Efficiency

The implementation maintains the memory efficiency of the original CCE:
- No materialization of full (batch, vocab) distance matrix
- Block-wise computation with efficient memory access patterns
- Pre-computed norm terms to avoid redundant calculations

## Performance Considerations

- Similar block structure to original CCE for optimal GPU utilization
- Additional norm computations add minimal overhead
- Float32 accumulators ensure precision without significant performance impact
- Autotuning through existing Triton autotune infrastructure

## Installation and Usage

```bash
# Install dependencies (requires CUDA and Triton)
pip install torch triton

# Import and use
from cut_harmax import cut_harmax_loss

# Works exactly like standard PyTorch loss functions
loss = cut_harmax_loss(embeddings, weights, targets)
loss.backward()
```

## Testing

### Functional Tests
- **Location**: `cut_harmax/tests/test_harmonic_loss_forward.py`
- **Features**: Comparison with manual implementation, gradient verification
- **Requirements**: CUDA GPU with Triton support

### API Tests
- **Location**: `test_harmax_api.py` (root directory)
- **Features**: API validation, module structure verification
- **Requirements**: Works without CUDA

### Running Tests

```bash
# API tests (always work)
python test_harmax_api.py

# Kernel tests (require CUDA)
pytest cut_harmax/tests/
```

## Comparison with Manual Implementation

```python
# Manual HarMax computation (for reference)
def manual_harmax_loss(e, c, targets):
    eps = 1e-9

    # Compute all distances
    e_expanded = e.unsqueeze(1)  # (batch, 1, embed_dim)
    c_expanded = c.unsqueeze(0)  # (1, vocab, embed_dim)
    distances = torch.sum((e_expanded - c_expanded) ** 2, dim=2) + eps

    # Get target distances
    target_distances = distances[torch.arange(len(targets)), targets]

    # Compute HarMax loss
    sum_reciprocal = torch.sum(1.0 / distances, dim=1)
    loss = torch.log(target_distances) - torch.log(sum_reciprocal)

    return loss
```

## Future Considerations

1. **Performance Optimization**: Further tuning of block sizes for HarMax-specific patterns
2. **Mixed Precision**: Investigation of other precision schemes if needed
3. **Adaptive Epsilon**: Dynamic epsilon selection based on input statistics
4. **Additional Variants**: Support for other distance-based loss functions
5. **Integration**: Integration with popular ML frameworks

## Relationship to Original CCE

This implementation demonstrates the power of the "Cut" technique:
- **Original**: Applied to Cross-Entropy loss → **Cut Cross-Entropy**
- **New**: Applied to Harmonic loss → **Cut HarMax**

Both implementations share the same core innovation: avoiding materialization of the full (batch_size, vocab_size) matrix through fused kernel computation and block-wise processing.

## Acknowledgments

This implementation builds upon the excellent work of the original CCE (Cut Cross-Entropy) authors, adapting their innovative "Cut" technique to the harmonic loss domain.
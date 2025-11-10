# Cut HarMax: Memory-efficient Harmonic Loss Implementation

## Overview

This implementation adapts the "Cut" technique from CCE to compute the Harmonic Loss efficiently. The new "Cut HarMax" kernels compute the exact, full-vocabulary harmonic loss without materializing the (batch_size, vocab_size) distance matrix, just as CCE does for the logit matrix.

## Project Structure

The implementation follows the same naming principle as the original CCE:

- **CCE** = Cut Cross-Entropy
- **HarMax** = Cut HarMax (Harmonic Loss with HarMax normalization)

```
cut_cross_entropy/     # Original Cross-Entropy implementation
cut_harmax/           # New HarMax implementation
├── harmax.py                 # Main API and autograd function
├── harmax_lse_forward.py    # Forward kernel (log-sum-reciprocal)
├── harmax_backward.py       # Backward kernel (harmonic gradients)
├── indexed_distance.py       # Target token distance extraction
├── constants.py              # Constants (IGNORE_INDEX)
├── utils.py                  # Utility functions
├── tl_autotune.py            # Triton autotuning
├── tl_utils.py               # Triton utilities
├── doc.py                    # Documentation utilities
└── tests/                   # Test suite
    └── test_harmax_sampling_inference.py
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

### 2. `harmax_lse_forward.py`
**Purpose**: Forward kernel for HarMax loss computation

**Key Features:**
- Fused distance calculation using identity: $L_j = ||v||^2 + ||w_j||^2 - 2(v·w_j)$
- Pre-computes `||v||^2` per batch and `||w_j||^2` per vocabulary
- Stable computation using minimum distance (instead of maximum logits)
- Epsilon (1e-9) addition to prevent division by zero

### 3. `harmax_backward.py`
**Purpose**: Backward kernel for HarMax gradient computation

**Key Features:**
- Computes HarMax probabilities: $p_j = \frac{1/L_j}{\sum(1/L_k)}$
- Gradient calculation:
  - For incorrect tokens: $\frac{\partial L}{\partial L_j} = \frac{p_j}{L_j}$
  - For correct token: $\frac{\partial L}{\partial L_y} = \frac{1 + p_y}{L_y}$
- Gradient backpropagation: $\nabla_v L_j = 2 \cdot (v - w_j)$, $\nabla_{w_j} L_j = -2 \cdot (v - w_j)$

### 4. `indexed_distance.py`
**Purpose**: Utility to extract correct token distances

**Key Features:**
- Computes squared Euclidean distance for target tokens only
- Pre-computes norm terms for efficiency

### 5. `harmax_sampling_inference.py`
**Purpose**: Fast inference sampling with min_p filtering

**Key Features:**
- `@torch.compile` optimized pure PyTorch implementation
- Min-p filtering for creative writing quality control
- Adaptive probability thresholding
- Full HarMax normalization with quality filtering

## API Usage

### Training

```python
from cut_harmax import cut_harmax_loss

loss = cut_harmax_loss(
    e=embeddings,           # (batch_size, embedding_dim) bfloat16
    c=weight_matrix,        # (vocab_size, embedding_dim) bfloat16
    targets=targets,        # (batch_size,) long
    reduction="mean",       # "mean", "sum", or "none"
    shift=False,            # for causal language modeling
)
```

### Inference

```python
from cut_harmax import harmax_sample

tokens = harmax_sample(
    hidden_states=hidden,     # (batch_size, embed_dim)
    weight_matrix=weights,      # (vocab_size, embed_dim)
    temperature=0.8,           # Sampling temperature
    min_p=0.01                # Minimum probability threshold
)
```

## Complete Example

```python
from cut_harmax import cut_harmax_loss, harmax_sample

class HarMaxModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(...)

    def forward(self, input_ids, targets=None):
        hidden = self.embedding(input_ids)
        hidden = self.transformer(hidden)

        if targets is not None:
            loss = cut_harmax_loss(
                e=hidden.view(-1, hidden.size(-1)),
                c=self.embedding.weight,
                targets=targets.view(-1)
            )
            return hidden, loss
        return hidden

    def generate(self, prompt, max_tokens=100):
        self.eval()
        generated = prompt.clone()

        for _ in range(max_tokens):
            hidden = self.forward(generated)
            last_hidden = hidden[:, -1:, :]

            next_token = harmax_sample(
                hidden_state=last_hidden.squeeze(0),
                weight_matrix=self.embedding.weight,
                temperature=0.8,
                min_p=0.01
            )

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        return generated
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
- **Min-p filtering**: Eliminates tokens below probability threshold

## Memory Efficiency

The implementation maintains the memory efficiency of the original CCE:
- No materialization of full (batch, vocab) distance matrix
- Block-wise computation with efficient memory access patterns
- Pre-computed norm terms to avoid redundant calculations

## Performance Considerations

- Similar block structure to original CCE for optimal GPU utilization
- torch.compile optimization for inference
- Min-p filtering for quality control with minimal overhead
- Float32 accumulators ensure precision without significant performance impact

## Installation and Usage

```bash
pip install torch triton

from cut_harmax import cut_harmax_loss, harmax_sample

loss = cut_harmax_loss(embeddings, weights, targets)
token = harmax_sample(hidden_state, weights)
```

## Testing

### Functional Tests
- **Location**: `cut_harmax/tests/test_harmax_sampling_inference.py`
- **Features**: Min-p filtering, temperature scaling, batch processing
- **Requirements**: CUDA GPU with Triton support for training tests

### API Tests
- **Location**: `test_harmax_api.py` (root directory)
- **Features**: API validation, module structure verification
- **Requirements**: Works without CUDA

## Relationship to Original CCE

This implementation demonstrates the power of the "Cut" technique:
- **Original**: Applied to Cross-Entropy loss → **Cut Cross-Entropy**
- **New**: Applied to Harmonic loss → **Cut HarMax**

Both implementations share the same core innovation: avoiding materialization of the full (batch_size, vocab_size) matrix through fused kernel computation and block-wise processing.

## Creative Writing Applications

The HarMax loss is particularly suitable for creative writing applications:

- **Semantic Coherence**: Distance-based relationships focus on meaning
- **Min-P Filtering**: Eliminates improbable tokens that break immersion
- **Temperature Control**: Balances dominance vs. diversity
- **No Averaging**: Avoids hedging toward "safe" predictions

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
    sum_reciprocal = torch.sum(1.0 / distances, dim=-1)
    loss = torch.log(target_distances) - torch.log(sum_reciprocal)

    return loss
```

## Future Considerations

1. **Performance Optimization**: Further tuning for specific hardware
2. **Adaptive Min-P**: Dynamic threshold selection based on text quality
3. **Hybrid Approaches**: Combining HarMax with other distance metrics
4. **Integration**: Compatibility with popular ML frameworks
5. **Adaptive Temperature**: Dynamic temperature adjustment during generation
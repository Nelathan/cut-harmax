#!/usr/bin/env python3
"""
Complete example of training and sampling with Cut HarMax.

This example shows how to integrate HarMax into a training loop and
use memory-efficient sampling for inference.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import HarMax components
from cut_harmax import cut_harmax_loss
from cut_harmax.harmax_sampling_inference import FastHarMaxSampler, harmax_inference_sample


class HarMaxLanguageModel(nn.Module):
    """
    Simple language model using HarMax instead of cross-entropy.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(1000, embed_dim)  # max_seq_len = 1000

        # Simple transformer stack (using nn.TransformerEncoder for simplicity)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Note: No final linear layer! We use embeddings directly for HarMax

        # Pre-compute weight norms for efficiency
        self.register_buffer('weight_norms', torch.zeros(vocab_size))

        # Initialize fast sampler for inference
        self._sampler = None

    @property
    def sampler(self) -> FastHarMaxSampler:
        """Lazy initialization of fast sampler."""
        if self._sampler is None:
            self._sampler = FastHarMaxSampler(
                weight_matrix=self.token_embedding.weight
            )
        return self._sampler

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # (B, T, D)
        pos_embeds = self.position_embedding(torch.arange(seq_len, device=device))
        hidden_states = token_embeds + pos_embeds

        # Pass through transformer
        hidden_states = self.transformer(hidden_states)  # (B, T, D)

        # Pre-compute weight norms if not already done
        if self.weight_norms.sum() == 0:
            self.weight_norms = (self.token_embedding.weight ** 2).sum(dim=1).float()

        if targets is not None:
            # Training: compute HarMax loss
            # Reshape for loss computation
            hidden_flat = hidden_states.view(-1, self.embed_dim)  # (B*T, D)
            targets_flat = targets.view(-1)  # (B*T,)

            loss = cut_harmax_loss(
                e=hidden_flat,
                c=self.token_embedding.weight,
                targets=targets_flat
            )

            return hidden_states, loss
        else:
            # Inference: return hidden states for sampling
            return hidden_states

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 64,
        use_fast_sampler: bool = True
    ) -> torch.Tensor:
        """
        Generate text using optimized HarMax sampling.

        Args:
            prompt: Starting token sequence (1, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Number of top candidates to consider
            use_fast_sampler: Use optimized single-token sampler

        Returns:
            Generated token sequence (1, seq_len + max_new_tokens)
        """
        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get hidden states for last position
                hidden_states = self.forward(generated)  # (1, T, D)
                last_hidden = hidden_states[:, -1, :]  # (1, D)

                # Sample next token using fast single-token sampling
                if use_fast_sampler and last_hidden.is_cuda:
                    next_token = harmax_inference_sample(
                        hidden_state=last_hidden.squeeze(0),  # (D,)
                        weight_matrix=self.token_embedding.weight,
                        temperature=temperature,
                        top_k=top_k
                    )
                else:
                    # Fallback to batch sampling
                    next_token = self.sampler.sample_batch(
                        hidden_states=last_hidden.unsqueeze(0),  # (1, D)
                        temperature=temperature,
                        top_k=top_k
                    ).squeeze(0)  # Get the single token

                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        return generated


def train_harmax_model():
    """
    Example training loop with gradient monitoring.
    """
    # Model hyperparameters
    vocab_size = 10000
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    batch_size = 32
    seq_len = 128
    learning_rate = 1e-4  # Start conservative

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HarMaxLanguageModel(vocab_size, embed_dim, num_heads, num_layers).to(device)

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Training data (dummy for example)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print("Starting HarMax training with gradient monitoring...")
    print("=" * 50)

    model.train()
    for step in range(100):  # Short example run
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler:
            with torch.cuda.amp.autocast():
                _, loss = model(dummy_input, dummy_targets)
        else:
            _, loss = model(dummy_input, dummy_targets)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Monitor gradients - crucial for HarMax hyperparameter tuning
        if step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():8.4f} | Grad Norm: {grad_norm:8.4f}")

    print("\nTraining complete! Testing generation...")
    print("=" * 50)

    # Test generation
    model.eval()
    prompt = torch.tensor([[1, 2, 3, 4]], device=device)  # Dummy prompt

    # Test different sampling parameters
    test_configs = [
        {"temperature": 0.5, "top_k": 32, "name": "Focused"},
        {"temperature": 0.8, "top_k": 64, "name": "Balanced"},
        {"temperature": 1.2, "top_k": 128, "name": "Creative"},
    ]

    for config in test_configs:
        print(f"\nGenerating with {config['name']} settings:")
        print(f"  Temperature: {config['temperature']}, Top-K: {config['top_k']}")

        generated = model.generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=config['temperature'],
            top_k=config['top_k'],
            use_fast_sampler=device.type == "cuda"
        )

        print(f"  Generated: {generated[0].tolist()}")


def hyperparameter_guidance():
    """
    Guidance for tuning HarMax hyperparameters.
    """
    print("\n" + "=" * 60)
    print("HARMAX HYPERPARAMETER TUNING GUIDE")
    print("=" * 60)

    print("""
1. LEARNING RATE:
   - Start with 1e-4 (smaller than typical CE lr of 1e-3)
   - Monitor gradient norms in early training
   - If grad_norm > 10: reduce learning rate
   - If grad_norm < 0.1: increase learning rate

2. GRADIENT CLIPPING:
   - Use max_norm=1.0 as starting point
   - HarMax gradients can be explosive for close embeddings
   - Adjust based on observed gradient norms

3. TEMPERATURE SAMPLING:
   - Default: 0.8 (lower than CE's 1.0)
   - HarMax creates sharper distributions naturally
   - Lower = more coherent, Higher = more creative

4. TOP-K/P FILTERING:
   - Start with standard CE defaults (top_k=50, top_p=0.9)
   - HarMax probabilities are more meaningful than CE logits
   - Can be more selective while maintaining quality

5. MONITORING:
   - Track loss scale (should be ~2-5 for typical embeddings)
   - Watch gradient norms (should be stable around 1.0)
   - Compare with CE baseline if available

6. EXPECTED DIFFERENCES:
   - Loss scale different from CE (not comparable)
   - Convergence speed may differ
   - Focus on qualitative output, not just loss numbers
""")


if __name__ == "__main__":
    # Run the example
    hyperparameter_guidance()
    train_harmax_model()
"""
Layer Normalization (LayerNorm)

Before feeding embeddings into the Transformer Block,
layer normalization stabilizes training by normalizing
each token's feature vector to have mean=0 and variance=1.

Why Layer Normalization?
  - Deep networks suffer from internal covariate shift
    → activations grow or shrink unpredictably across layers
  - LayerNorm re-centers and rescales each token independently
    → keeps activations in a stable range throughout training
  - Unlike BatchNorm (normalizes across batch),
    LayerNorm normalizes across the feature dimension of each token

Steps:
  1. Manually compute normalization to understand the mechanics
  2. Wrap into a LayerNorm class with learnable scale (gamma) and shift (beta)
  3. Verify against PyTorch's built-in nn.LayerNorm
"""

import torch
import torch.nn as nn

# --- Step 1: Manually compute layer normalization ---
# Use a small example: batch of 2 tokens, each with 5 features
torch.manual_seed(123)
batch_example = torch.randn(2, 5)  # [batch_size, num_features]
print("Input:")
print(batch_example)

# Compute mean and variance for each token independently (dim=-1 = feature dim)
mean = batch_example.mean(dim=-1, keepdim=True)  # [2, 1]
var  = batch_example.var(dim=-1, keepdim=True, unbiased=False)  # [2, 1]
print(f"\nMean: {mean}")
print(f"Variance: {var}")

# Normalize: subtract mean, divide by std
# eps=1e-5: small constant to prevent division by zero
eps = 1e-5
norm = (batch_example - mean) / torch.sqrt(var + eps)
print("\nNormalized:")
print(norm)

# Verify: mean ≈ 0, variance ≈ 1 for each token
print(f"\nAfter norm - Mean : {norm.mean(dim=-1)}")   # → [~0.0, ~0.0]
print(f"After norm - Var  : {norm.var(dim=-1)}")    # → [~1.0, ~1.0]

# --- Step 2: LayerNorm class with learnable scale and shift ---
class LayerNorm(nn.Module):
    """Layer normalization with learnable scale (gamma) and shift (beta).

    After normalizing to mean=0 and variance=1,
    gamma and beta allow the model to rescale and shift the output.
    These are learned during training so the model can undo normalization
    if needed for a particular layer.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        # gamma: learnable scale, initialized to 1 (no scaling at start)
        # beta:  learnable shift, initialized to 0 (no shift at start)
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # beta

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        # Apply learnable scale and shift
        return self.scale * norm + self.shift

# --- Step 3: Verify against PyTorch's built-in nn.LayerNorm ---
torch.manual_seed(123)
batch_example = torch.randn(2, 5)

emb_dim = batch_example.shape[-1]  # 5

# Our implementation
layer_norm = LayerNorm(emb_dim)
out_custom = layer_norm(batch_example)
print("Custom LayerNorm output:")
print(out_custom)
print(f"Mean : {out_custom.mean(dim=-1)}")  # → ~0.0
print(f"Var  : {out_custom.var(dim=-1)}")   # → ~1.0

# PyTorch built-in
layer_norm_pytorch = nn.LayerNorm(emb_dim)
out_pytorch = layer_norm_pytorch(batch_example)
print("\nPyTorch nn.LayerNorm output:")
print(out_pytorch)

# Both should produce identical results
print(f"\nMax difference: {(out_custom - out_pytorch).abs().max():.6f}")
# → 0.000000 (or very close)
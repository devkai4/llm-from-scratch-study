"""
Feed Forward Network (FFN) with GELU Activation

Each Transformer Block contains a Feed Forward Network applied
to each token independently after the attention layer.

Structure:
  input → Linear(emb_dim, 4*emb_dim) → GELU → Linear(4*emb_dim, emb_dim) → output

Why 4x expansion?
  - The hidden layer is 4x wider than the input/output
  - This gives the network more capacity to learn complex transformations
  - The second linear layer projects back to the original dimension

Why GELU instead of ReLU?
  - ReLU: hard zero cutoff at x=0 → dead neurons problem
  - GELU: smooth curve, allows small negative values to pass
  - Works better in practice for Transformer-based models

Steps:
  1. Implement GELU activation manually
  2. Verify GELU against PyTorch's built-in nn.GELU
  3. Implement FeedForward class
  4. Demo
"""

import torch
import torch.nn as nn

# --- Step 1: GELU activation (manual implementation) ---
class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function.

    GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of the standard normal.

    In practice, a fast approximation is used (same as GPT-2):
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# --- Step 2: Verify GELU against PyTorch's built-in ---
gelu_custom = GELU()
gelu_pytorch = nn.GELU()
relu = nn.ReLU()

x = torch.linspace(-3, 3, 100)

print("Max difference (custom vs PyTorch GELU):")
print(f"{(gelu_custom(x) - gelu_pytorch(x)).abs().max():.6f}")  # → ~0.000000

# Compare GELU vs ReLU at a few key points
print("\nGELU vs ReLU comparison:")
print(f"{'x':>6} {'GELU':>10} {'ReLU':>10}")
for val in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    t = torch.tensor(val)
    print(f"{val:>6.1f} {gelu_custom(t).item():>10.4f} {relu(t).item():>10.4f}")

# --- Step 3: FeedForward class ---
class FeedForward(nn.Module):
    """Position-wise Feed Forward Network used inside each Transformer Block.

    Applied independently to each token after the attention layer.
    Expands the embedding dimension by 4x, applies GELU, then projects back.

    Architecture:
      Linear(emb_dim → 4*emb_dim) → GELU → Linear(4*emb_dim → emb_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # expand
            GELU(),                                           # activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # project back
        )

    def forward(self, x):
        return self.layers(x)


# --- Step 4: Demo ---
# GPT-2 small config (kept minimal for demonstration)
GPT_CONFIG = {
    "emb_dim": 768,  # embedding dimension (GPT-2 small)
}

torch.manual_seed(123)
ffn = FeedForward(GPT_CONFIG)

# Input: batch of 2 samples, 3 tokens each, 768-dim embeddings
x = torch.rand(2, 3, 768)  # [batch_size, num_tokens, emb_dim]
out = ffn(x)

print(f"Input shape : {x.shape}")    # → [2, 3, 768]
print(f"Output shape: {out.shape}")  # → [2, 3, 768]  shape preserved
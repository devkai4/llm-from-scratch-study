"""
Simple Self-Attention (no trainable weights)

Computes a context vector for each token by:
  1. Calculating attention scores via dot products between tokens
  2. Normalizing scores into weights using softmax
  3. Taking a weighted sum of all token embeddings

This is the simplest form of self-attention — no Q, K, V matrices yet.
"""

import torch

# --- Toy input embeddings ---
# Pretend these are already-embedded tokens (output of ch02 embedding layer).
# Each row = one token, each column = one embedding dimension.
# Shape: [seq_len=6, embed_dim=3]
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],   # "Your"     (x_1)
     [0.55, 0.87, 0.66],   # "journey"  (x_2)
     [0.57, 0.85, 0.64],   # "starts"   (x_3)
     [0.22, 0.58, 0.33],   # "with"     (x_4)
     [0.77, 0.25, 0.10],   # "one"      (x_5)
     [0.05, 0.80, 0.55]]   # "step"     (x_6)
)

print(f"Inputs shape: {inputs.shape}")  # → torch.Size([6, 3])
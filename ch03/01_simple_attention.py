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

# --- Step 1: Compute attention scores for one query ---
# Pick x_2 ("journey") as the query token.
# Goal: measure how relevant every other token is to "journey".
query = inputs[1]   # x_2 (journey), shape [3]

# Compute the dot product between the query and every token in inputs.
# Dot product is high when two vectors point in similar directions
# → high score means "this token is similar to the query".
attn_scores_2 = torch.empty(inputs.shape[0])  # 6 empty slots, one per token
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(f"Attention scores (query=x_2): {attn_scores_2}")
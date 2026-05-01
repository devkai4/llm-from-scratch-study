"""
Self-Attention with Trainable Weights (SelfAttentionV1 & V2)

Extends simple self-attention (01) by introducing three learnable weight matrices:
  - W_query : projects input into query space
  - W_key   : projects input into key space
  - W_value : projects input into value space

The attention score is scaled by sqrt(d_k) to prevent vanishing gradients
when the dot product grows large with high-dimensional vectors.

Steps:
  1. Define input embeddings (same 6-token example as the book)
  2. Implement SelfAttentionV1 using nn.Parameter (manual weights)
  3. Implement SelfAttentionV2 using nn.Linear  (cleaner, bias-free)
  4. Verify both produce equivalent results when weights are copied
"""

import torch
import torch.nn as nn

# --- Input embeddings ---
# 6 tokens, each represented as a 3-dimensional embedding vector
# (same example used throughout Chapter 3 in the book)
torch.manual_seed(123)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # token 0: "Your"
    [0.55, 0.87, 0.66],  # token 1: "journey"
    [0.57, 0.85, 0.64],  # token 2: "starts"
    [0.22, 0.58, 0.33],  # token 3: "with"
    [0.77, 0.25, 0.10],  # token 4: "one"
    [0.05, 0.80, 0.55],  # token 5: "step"
])
# shape: [6, 3]  →  [num_tokens, d_in]
# d_in = 3  : input embedding dimension
# d_out = 2 : output (query/key/value) dimension (kept small for readability)

d_in  = inputs.shape[1]  # 3
d_out = 2
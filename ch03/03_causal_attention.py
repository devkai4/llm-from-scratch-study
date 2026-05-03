"""
Causal Self-Attention (CausalAttention)

Extends SelfAttentionV2 by masking future tokens so each position
can only attend to itself and earlier positions.

Why causal masking?
  - GPT-style LLMs predict the next token autoregressively
  - Allowing token i to attend to token j > i would leak future information
  - Solution: set attention scores for j > i to -inf before softmax
              → softmax converts -inf to 0 → future tokens are invisible

Steps:
  1. Compute attention weights using SelfAttentionV2 as baseline
  2. Apply causal mask (upper triangle → -inf)
  3. Add dropout to attention weights for regularization
  4. Wrap into CausalAttention class
"""

import torch
import torch.nn as nn

# --- Input embeddings (reuse from ch03 book example) ---
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

d_in  = inputs.shape[1]  # 3
d_out = 2

# --- Step 1: Compute baseline attention weights (no masking yet) ---
# Reuse SelfAttentionV2 weights to get attention scores for all token pairs
torch.manual_seed(789)

# Temporary linear layers to compute Q, K, V
W_query = nn.Linear(d_in, d_out, bias=False)
W_key   = nn.Linear(d_in, d_out, bias=False)
W_value = nn.Linear(d_in, d_out, bias=False)

queries = W_query(inputs)  # [6, 2]
keys    = W_key(inputs)    # [6, 2]
values  = W_value(inputs)  # [6, 2]

# Raw attention scores: [6, 6]
# entry [i, j] = how much token i attends to token j
attn_scores = queries @ keys.T

# Scale and apply softmax → attention weights
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("Attention weights (no mask):")
print(attn_weights)
# Each row sums to 1, but token i can see ALL tokens including future ones

# --- Step 2: Apply causal mask (mask out future tokens) ---
# Create a mask matrix of shape [num_tokens, num_tokens]
# torch.triu(..., diagonal=1) returns the upper triangle excluding the diagonal
# entry [i, j] = True if j > i  → token i should NOT attend to token j
num_tokens = inputs.shape[0]  # 6
mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)  # [6, 6]
print("Causal mask (1 = position to block):")
print(mask)

# Replace masked positions with -inf before softmax
# -inf → softmax will convert to exactly 0 → future tokens become invisible
masked_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))
print("\nAttention scores after masking:")
print(masked_scores)

# Apply softmax → masked positions become 0
attn_weights_masked = torch.softmax(masked_scores / keys.shape[-1]**0.5, dim=-1)
print("\nAttention weights after masking:")
print(attn_weights_masked)
# Each row still sums to 1, but future tokens have weight = 0
# e.g. row 0: only token 0 itself is visible
# e.g. row 2: tokens 0, 1, 2 are visible; tokens 3, 4, 5 are masked

# --- Step 3: Add dropout to attention weights ---
# Dropout randomly zeros out some attention weights during training
# Purpose: prevents the model from over-relying on specific tokens → reduces overfitting
# dropout_rate=0.5 means 50% of weights are zeroed out (kept small here for demo)
torch.manual_seed(123)
dropout = nn.Dropout(p=0.5)

attn_weights_dropped = dropout(attn_weights_masked)
print("Attention weights after dropout:")
print(attn_weights_dropped)
# Note: remaining weights are scaled up by 1/(1-p) automatically
# e.g. p=0.5 → surviving weights are multiplied by 2
# this keeps the expected sum of each row equal to 1

# --- Step 4: CausalAttention class ---
class CausalAttention(nn.Module):
    """Self-attention with causal masking and dropout.

    Ensures each token can only attend to itself and earlier positions.
    Dropout is applied to attention weights for regularization.
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Register causal mask as a buffer (not a learnable parameter)
        # buffer: saved with the model but not updated by the optimizer
        # torch.triu: upper triangle of ones → positions to mask
        # shape: [context_length, context_length]
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # x shape: [batch_size, num_tokens, d_in]
        batch_size, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # [batch_size, num_tokens, d_out]
        keys    = self.W_key(x)    # [batch_size, num_tokens, d_out]
        values  = self.W_value(x)  # [batch_size, num_tokens, d_out]

        # Compute attention scores
        # transpose(-2, -1): swap last two dims [batch, d_out, num_tokens]
        # result: [batch_size, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(-2, -1)

        # Apply causal mask: slice to actual sequence length and mask future positions
        attn_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens].bool(), float('-inf')
        )

        # Scale + softmax + dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values → context vectors
        # [batch_size, num_tokens, num_tokens] @ [batch_size, num_tokens, d_out]
        # = [batch_size, num_tokens, d_out]
        context_vecs = attn_weights @ values
        return context_vecs

# --- Demo ---
torch.manual_seed(123)

# CausalAttention expects input shape: [batch_size, num_tokens, d_in]
# unsqueeze(0): add batch dimension → [1, 6, 3]
batch = inputs.unsqueeze(0)
print(f"Input shape: {batch.shape}")  # → torch.Size([1, 6, 3])

context_length = inputs.shape[0]  # 6
ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)

context_vecs = ca(batch)
print(f"Output shape: {context_vecs.shape}")  # → torch.Size([1, 6, 2])
print("Context vectors:")
print(context_vecs)
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
  2. Manually compute Q, K, V for a single query token (x_2)
  3. Expand to all tokens and compute attention weights
  4. Compute context vectors via weighted sum of values
  5. Wrap steps 2-4 into SelfAttentionV1 (nn.Parameter)
  6. Refactor into SelfAttentionV2 (nn.Linear, cleaner)
  7. Verify both produce equivalent results when weights are copied
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

# --- Step 2: Manually define W_q, W_k, W_v ---
# Each weight matrix projects input from d_in=3 dimensions to d_out=2 dimensions
# requires_grad=False: we are not training here, just demonstrating the mechanics
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # [3, 2]
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # [3, 2]
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # [3, 2]

# --- Step 3: Compute Q, K, V for a single query token (x_2 = "starts") ---
x_2 = inputs[2]                 # shape: [3]  → the query token
query_2 = x_2 @ W_query         # [3] @ [3, 2] = [2]
key_2   = x_2 @ W_key           # [2]
value_2 = x_2 @ W_value         # [2]
print(f"query_2 shape: {query_2.shape}")  # → torch.Size([2])

# --- Step 4: Expand keys and values to all tokens ---
# compute keys and values for every token in the sequence
keys   = inputs @ W_key    # [6, 3] @ [3, 2] = [6, 2]
values = inputs @ W_value  # [6, 3] @ [3, 2] = [6, 2]
print(f"keys shape  : {keys.shape}")    # → torch.Size([6, 2])
print(f"values shape: {values.shape}")  # → torch.Size([6, 2])

# --- Step 5: Compute attention scores for token 2 against all tokens ---
# query_2 @ keys.T: [2] @ [2, 6] = [6]
# each element = dot product of query_2 with each key → raw attention score
attn_scores_2 = query_2 @ keys.T  # [6]
print(f"attn_scores_2: {attn_scores_2}")

# Scale by sqrt(d_out) and apply softmax to get attention weights
# scaling prevents softmax from saturating when d_out is large
attn_weights_2 = torch.softmax(attn_scores_2 / d_out**0.5, dim=-1)  # [6]
print(f"attn_weights_2 (sum={attn_weights_2.sum():.1f}): {attn_weights_2}")

# --- Step 6: Compute context vector for token 2 ---
# weighted sum of all value vectors
# attn_weights_2: [6], values: [6, 2] → context_vec_2: [2]
context_vec_2 = attn_weights_2 @ values  # [2]
print(f"context_vec_2: {context_vec_2}")

# --- SelfAttentionV1 ---
class SelfAttentionV1(nn.Module):
    """Self-attention using nn.Parameter for explicit weight matrices."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys    = x @ self.W_key
        values  = x @ self.W_value
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / self.W_key.shape[-1]**0.5, dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs

# --- SelfAttentionV2 ---
class SelfAttentionV2(nn.Module):
    """Self-attention using nn.Linear instead of nn.Parameter.

    nn.Linear internally manages W and (optionally) bias as Parameters.
    bias=False makes it equivalent to SelfAttentionV1 (pure matrix multiplication).
    This is the cleaner, more conventional PyTorch style.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # nn.Linear(d_in, d_out, bias=False) is equivalent to x @ W.T
        # internally holds a weight matrix of shape [d_out, d_in] (transposed)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # nn.Linear handles the matrix multiplication internally
        # equivalent to x @ W.T in SelfAttentionV1
        queries = self.W_query(x)  # [6, 3] → [6, 2]
        keys    = self.W_key(x)    # [6, 3] → [6, 2]
        values  = self.W_value(x)  # [6, 3] → [6, 2]

        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs

# --- Demo ---
torch.manual_seed(789)
sa_v1 = SelfAttentionV1(d_in, d_out)
print("SelfAttentionV1 output:")
print(sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
print("\nSelfAttentionV2 output:")
print(sa_v2(inputs))

# --- Verify equivalence: copy V1 weights into V2 ---
# SelfAttentionV1 stores W as [d_in, d_out]
# SelfAttentionV2 (nn.Linear) stores W as [d_out, d_in] → need .T to copy
sa_v2.W_query.weight = nn.Parameter(sa_v1.W_query.T)
sa_v2.W_key.weight   = nn.Parameter(sa_v1.W_key.T)
sa_v2.W_value.weight = nn.Parameter(sa_v1.W_value.T)

print("\nAfter copying V1 weights into V2:")
print("SelfAttentionV1 output:")
print(sa_v1(inputs))
print("SelfAttentionV2 output (should match V1):")
print(sa_v2(inputs))
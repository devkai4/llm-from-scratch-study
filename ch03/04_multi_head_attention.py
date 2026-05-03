"""
Multi-Head Attention (MultiHeadAttentionWrapper & MultiHeadAttention)

Extends CausalAttention by running multiple attention heads in parallel.
Each head learns to attend to different aspects of the input sequence.

Why multiple heads?
  - A single attention head can only capture one type of relationship at a time
  - Multiple heads allow the model to jointly attend to information
    from different representation subspaces (syntax, semantics, position, etc.)
  - Outputs from all heads are concatenated and projected via a linear layer

Steps:
  1. Implement MultiHeadAttentionWrapper (stack multiple CausalAttention heads)
  2. Implement MultiHeadAttention (single efficient class with split heads)
  3. Verify both produce the same output shape
"""

import torch
import torch.nn as nn

# --- Input embeddings ---
torch.manual_seed(123)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # token 0: "Your"
    [0.55, 0.87, 0.66],  # token 1: "journey"
    [0.57, 0.85, 0.64],  # token 2: "starts"
    [0.22, 0.58, 0.33],  # token 3: "with"
    [0.77, 0.25, 0.10],  # token 4: "one"
    [0.05, 0.80, 0.55],  # token 5: "step"
])

d_in  = inputs.shape[1]  # 3
d_out = 2

# CausalAttention expects [batch_size, num_tokens, d_in]
batch = inputs.unsqueeze(0)  # [1, 6, 3]

# --- CausalAttention (reused from 03_causal_attention.py) ---
class CausalAttention(nn.Module):
    """Single-head causal self-attention (from Section 3.5)."""

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens].bool(), float('-inf')
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values

# --- Step 1: MultiHeadAttentionWrapper ---
class MultiHeadAttentionWrapper(nn.Module):
    """Multi-head attention implemented as a stack of CausalAttention heads.

    Each head independently computes attention with its own W_q, W_k, W_v.
    Outputs from all heads are concatenated along the last dimension.

    Simple but inefficient: each head runs sequentially in a Python loop.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Create num_heads independent CausalAttention heads
        # each head has its own W_query, W_key, W_value → learns different patterns
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # Run each head independently and concatenate outputs along last dim
        # each head output: [batch_size, num_tokens, d_out]
        # after concat:     [batch_size, num_tokens, d_out * num_heads]
        return torch.cat([head(x) for head in self.heads], dim=-1)

# --- Step 2: MultiHeadAttention ---
class MultiHeadAttention(nn.Module):
    """Multi-head attention implemented as a single efficient class.

    Instead of running num_heads separate CausalAttention instances,
    this class computes all heads in parallel using tensor reshaping.

    Key idea:
      - Project input into [batch, num_tokens, d_out] once
      - Reshape to [batch, num_tokens, num_heads, head_dim]
      - Transpose to [batch, num_heads, num_tokens, head_dim]
      - Compute attention for all heads simultaneously
      - Combine outputs via linear projection
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # dimension per head

        # Single linear layer for all heads combined
        # output size = d_out (will be split into num_heads later)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection: combine all head outputs into final representation
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        # Project input → [batch, num_tokens, d_out]
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # Split d_out into num_heads × head_dim
        # [batch, num_tokens, d_out] → [batch, num_tokens, num_heads, head_dim]
        # → transpose → [batch, num_heads, num_tokens, head_dim]
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores for all heads simultaneously
        # [batch, num_heads, num_tokens, head_dim] @ [batch, num_heads, head_dim, num_tokens]
        # = [batch, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(-2, -1)

        # Apply causal mask
        attn_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens].bool(), float('-inf')
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum → [batch, num_heads, num_tokens, head_dim]
        context_vecs = attn_weights @ values

        # Merge heads back: transpose → [batch, num_tokens, num_heads, head_dim]
        # contiguous + view → [batch, num_tokens, d_out]
        context_vecs = context_vecs.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # Final linear projection
        return self.out_proj(context_vecs)

# --- Demo ---
torch.manual_seed(123)

context_length = inputs.shape[0]  # 6
num_heads = 2

# --- Wrapper version ---
mha_wrapper = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, dropout=0.0, num_heads=num_heads
)
context_vecs_wrapper = mha_wrapper(batch)
print(f"MultiHeadAttentionWrapper output shape: {context_vecs_wrapper.shape}")
# → [1, 6, 4]  (d_out * num_heads = 2 * 2 = 4)
print(context_vecs_wrapper)

# --- Efficient version ---
# d_out=4 so each head gets head_dim = 4 // 2 = 2
mha = MultiHeadAttention(
    d_in, d_out=4, context_length=context_length, dropout=0.0, num_heads=num_heads
)
context_vecs_mha = mha(batch)
print(f"\nMultiHeadAttention output shape: {context_vecs_mha.shape}")
# → [1, 6, 4]
print(context_vecs_mha)
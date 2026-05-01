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

# --- Step 2: Normalize scores into weights using softmax ---
# Raw scores aren't usable as weights:
#   - they don't sum to 1
#   - they can be negative
# Softmax fixes both: outputs are positive AND sum to 1.
attn_weights_2 = torch.softmax(attn_scores_2, dim=0) # dim=0 means softmax over the first dimension

print(f"Attention weights (query=x_2): {attn_weights_2}")
print(f"Sum of weights: {attn_weights_2.sum()}")

# --- Step 3: Compute the context vector for x_2 ---
# The context vector is a weighted sum of all input tokens,
# where weights = attention weights computed above.
# This produces a new vector that "blends in" information from
# the most relevant tokens, weighted by relevance.
context_vec_2 = torch.zeros(query.shape)   # initialize as [0, 0, 0]
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(f"Context vector for x_2: {context_vec_2}")

# --- Step 4: Compute attention for ALL tokens at once (using matrix ops) ---
# Until now we only processed x_2.
# Now compute context vectors for every token in a single matrix multiplication.

# Step 4-1: All-pairs attention scores
# inputs @ inputs.T computes the dot product between every pair of tokens.
# Result shape: [6, 6]
#   attn_scores[i][j] = dot product of inputs[i] and inputs[j]
#   row i = "how relevant every token is when token i is the query"
attn_scores = inputs @ inputs.T
print(f"All attention scores shape: {attn_scores.shape}")
print(f"All attention scores:\n{attn_scores}")

# Step 4-2: Apply softmax along the last dimension (each row sums to 1)
# dim=-1 means: normalize along the rightmost dimension (each row)
# So every row now represents an attention weight distribution.
attn_weights = torch.softmax(attn_scores, dim=-1)
print(f"\nAll attention weights:\n{attn_weights}")
print(f"All rows sum to 1: {attn_weights.sum(dim=-1)}")

# Step 4-3: Compute all context vectors at once
# attn_weights @ inputs:
#   shape [6, 6] @ [6, 3] → [6, 3]
#   row i = weighted sum of all tokens, weighted by attn_weights[i]
all_context_vecs = attn_weights @ inputs
print(f"\nAll context vectors shape: {all_context_vecs.shape}")
print(f"All context vectors:\n{all_context_vecs}")

# Sanity check: row 1 (x_2's context vector) should match context_vec_2 from Step 3
print(f"\nContext vector for x_2 (from Step 3):  {context_vec_2}")
print(f"Context vector for x_2 (from Step 4):  {all_context_vecs[1]}")
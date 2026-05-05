"""
Transformer Block (Shortcut Connections + LayerNorm + Attention + FFN)

A Transformer Block combines all components from ch04 so far:
  - Multi-Head Causal Attention  (ch03)
  - Feed Forward Network         (ch04 4.3)
  - Layer Normalization          (ch04 4.2)
  - Shortcut (residual) connections

Why shortcut connections?
  - Deep networks suffer from vanishing gradients
    → gradients shrink to near-zero as they propagate back through many layers
  - Shortcut: add the input directly to the output of each sub-layer
    → creates a "highway" for gradients to flow back unchanged
  - Enables training of very deep networks (GPT-2: 12 layers, GPT-3: 96 layers)

Structure of one Transformer Block:
  x → LayerNorm → MultiHeadAttention → + x  (residual)
    → LayerNorm → FeedForward        → + x  (residual)

Steps:
  1. Demonstrate shortcut connections with a simple example
  2. Implement TransformerBlock class
  3. Demo
"""

import torch
import torch.nn as nn

# --- Reuse components from previous files ---
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out    = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens].bool(), float('-inf')
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = attn_weights @ values
        context_vecs = context_vecs.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.out_proj(context_vecs)


# --- Step 1: Shortcut connections demo ---
# Without shortcut: gradient shrinks through each layer
# With shortcut:    input is added directly → gradient flows unchanged

# Simple 5-layer network WITHOUT shortcut
class ExampleDeepNetPlain(nn.Module):
    def __init__(self, layer_sizes, use_shortcut=False):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]), GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            # shortcut: add input to output if dimensions match
            if self.use_shortcut and x.shape == layer_out.shape:
                x = x + layer_out  # residual connection
            else:
                x = layer_out
        return x

torch.manual_seed(123)
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])

# Without shortcut
model_plain = ExampleDeepNetPlain(layer_sizes, use_shortcut=False)
output = model_plain(sample_input)
output.backward()  # compute gradients

print("Gradients WITHOUT shortcut:")
for i, layer in enumerate(model_plain.layers):
    print(f"  layer {i}: {layer[0].weight.grad.abs().mean():.6f}")

# With shortcut
torch.manual_seed(123)
model_shortcut = ExampleDeepNetPlain(layer_sizes, use_shortcut=True)
output = model_shortcut(sample_input)
output.backward()

print("\nGradients WITH shortcut:")
for i, layer in enumerate(model_shortcut.layers):
    print(f"  layer {i}: {layer[0].weight.grad.abs().mean():.6f}")
# Gradients should be more stable (less shrinkage) with shortcut

# --- Step 2: TransformerBlock class ---
class TransformerBlock(nn.Module):
    """One Transformer Block: Attention + FFN with residual connections and LayerNorm.

    Pre-LayerNorm style (used in GPT-2):
      normalize BEFORE attention/FFN, not after.
      This is more stable during training than post-LayerNorm.

    Structure:
      x → LayerNorm → MultiHeadAttention → + x  (residual)
        → LayerNorm → FeedForward        → + x  (residual)
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff   = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # --- Attention sub-layer with residual ---
        shortcut = x                          # save input for residual
        x = self.norm1(x)                     # normalize first (pre-LN)
        x = self.att(x)                       # multi-head attention
        x = self.drop_shortcut(x)             # dropout
        x = x + shortcut                      # add residual

        # --- FeedForward sub-layer with residual ---
        shortcut = x                          # save input for residual
        x = self.norm2(x)                     # normalize first (pre-LN)
        x = self.ff(x)                        # feed forward
        x = self.drop_shortcut(x)             # dropout
        x = x + shortcut                      # add residual

        return x


# --- Step 3: Demo ---
GPT_CONFIG_124M = {
    "vocab_size"    : 50257,  # GPT-2 vocabulary size
    "context_length": 1024,   # maximum sequence length
    "emb_dim"       : 768,    # embedding dimension
    "n_heads"       : 12,     # number of attention heads
    "n_layers"      : 12,     # number of transformer blocks
    "drop_rate"     : 0.1,    # dropout rate
    "qkv_bias"      : False,  # no bias in Q, K, V projections
}

torch.manual_seed(123)
x = torch.rand(2, 4, 768)  # [batch_size, num_tokens, emb_dim]

block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print(f"Input shape : {x.shape}")       # → [2, 4, 768]
print(f"Output shape: {output.shape}")  # → [2, 4, 768]  shape preserved
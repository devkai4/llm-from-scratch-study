"""
Saving and Loading Model Weights (Section 5.4)

After training, we need to save the model so we can:
  - Resume training later without starting from scratch
  - Load the model for inference (text generation)
  - Share the model with others

Two approaches:
  1. Save weights only (state_dict) → recommended, more flexible
  2. Save entire model object       → convenient but fragile

Also demonstrates saving/loading optimizer state for resuming training.
"""

import torch
import torch.nn as nn
import os

# --- Reuse GPTModel (condensed) ---
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
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.W_query   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key     = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj  = nn.Linear(d_out, d_out)
        self.dropout   = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        queries = self.W_query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = self.W_key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), float('-inf'))
        attn_weights = self.dropout(torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1))
        out = (attn_weights @ values).transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff            = FeedForward(cfg)
        self.norm1         = LayerNorm(cfg["emb_dim"])
        self.norm2         = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb    = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        batch_size, num_tokens = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(num_tokens, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


GPT_CONFIG_124M = {
    "vocab_size"    : 50257,
    "context_length": 256,
    "emb_dim"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 12,
    "drop_rate"     : 0.1,
    "qkv_bias"      : False,
}


# =============================================================================
# Save and Load
# =============================================================================

torch.manual_seed(123)
model     = GPTModel(GPT_CONFIG_124M)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# --- Save: weights + optimizer state ---
# state_dict(): ordered dict of all learnable parameter tensors
# saving optimizer state allows resuming training from exact same point
save_path = os.path.join(os.path.dirname(__file__), "model_and_optimizer.pth")
torch.save({
    "model_state_dict"    : model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, save_path)
print(f"Saved model and optimizer to: {save_path}")

# --- Load: restore weights into a new model instance ---
# must create a new model with the same architecture first
model_new     = GPTModel(GPT_CONFIG_124M)
optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=0.0004, weight_decay=0.1)

checkpoint = torch.load(save_path, weights_only=True)
model_new.load_state_dict(checkpoint["model_state_dict"])
optimizer_new.load_state_dict(checkpoint["optimizer_state_dict"])
model_new.eval()
print("Loaded model and optimizer from checkpoint.")

# --- Verify: both models produce identical output ---
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model_new.to(device)
model.eval()
model_new.eval()

test_input = torch.tensor([[1, 2, 3, 4]]).to(device)
with torch.no_grad():
    out_original = model(test_input)
    out_loaded   = model_new(test_input)

max_diff = (out_original - out_loaded).abs().max().item()
print(f"\nMax output difference (original vs loaded): {max_diff:.6f}")
# → 0.000000
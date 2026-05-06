"""
Evaluating LLM Output Quality (Cross-Entropy Loss + Perplexity)

Before training, we need a way to measure how well the model predicts text.

Cross-Entropy Loss:
  - Measures how different the model's predicted probability distribution
    is from the true next token
  - Lower loss = model is more confident about the correct next token
  - Loss = 0 would mean perfect prediction (never happens in practice)

Perplexity:
  - perplexity = exp(loss)
  - More intuitive: "how many tokens is the model equally confused between?"
  - perplexity=1   → perfect prediction
  - perplexity=50  → as confused as guessing among 50 equally likely tokens

Steps:
  1. Run a forward pass and inspect logits
  2. Compute cross-entropy loss manually
  3. Compute perplexity
  4. Implement calc_loss_batch and calc_loss_loader utilities
"""

import torch
import torch.nn as nn
import tiktoken

# --- Reuse GPTModel from ch04 ---
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
        self.ff          = FeedForward(cfg)
        self.norm1       = LayerNorm(cfg["emb_dim"])
        self.norm2       = LayerNorm(cfg["emb_dim"])
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


# --- Config ---
GPT_CONFIG_124M = {
    "vocab_size"    : 50257,
    "context_length": 256,   # shorter for faster training in ch05
    "emb_dim"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 12,
    "drop_rate"     : 0.1,
    "qkv_bias"      : False,
}

# --- Step 1: Forward pass and inspect logits ---
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

# Two sample inputs (batch_size=2)
batch = torch.stack([
    torch.tensor(tokenizer.encode("Every effort moves you")),
    torch.tensor(tokenizer.encode("Every day holds a")),
])
print(f"Input shape : {batch.shape}")  # → [2, 4]

logits = model(batch)
print(f"Logits shape: {logits.shape}")  # → [2, 4, 50257]
# For each of the 4 input tokens, the model predicts a distribution over 50257 vocab tokens

# --- Step 2: Compute cross-entropy loss manually ---
# The model predicts the next token for each position.
# Target: the actual next token at each position (input shifted by 1)

# Example: input  = ["Every", "effort", "moves", "you"]
#          target = ["effort", "moves",  "you",  "forward"]

targets = torch.stack([
    torch.tensor(tokenizer.encode("effort moves you forward")[:4]),
    torch.tensor(tokenizer.encode("day holds a promise")[:4]),
])
print(f"Targets shape: {targets.shape}")  # → [2, 4]

# logits:  [batch_size, num_tokens, vocab_size] = [2, 4, 50257]
# targets: [batch_size, num_tokens]             = [2, 4]
# CrossEntropyLoss expects:
#   input:  [N, vocab_size]  → flatten batch and token dims
#   target: [N]

logits_flat  = logits.flatten(0, 1)   # [2*4, 50257] = [8, 50257]
targets_flat = targets.flatten()       # [2*4]        = [8]

loss = nn.functional.cross_entropy(logits_flat, targets_flat)
print(f"\nCross-entropy loss: {loss.item():.4f}")
# Untrained model → loss ≈ log(50257) ≈ 10.82 (random guessing baseline)

# --- Step 3: Perplexity ---
perplexity = torch.exp(loss)
print(f"Perplexity      : {perplexity.item():.2f}")
# → ~50257 for a random model (equally confused about all vocab tokens)

# --- Step 4: Utility functions for training ---
def calc_loss_batch(input_batch, target_batch, model, device):
    """Compute cross-entropy loss for a single batch."""
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)                        # [batch, num_tokens, vocab_size]
    loss   = nn.functional.cross_entropy(
        logits.flatten(0, 1),                          # [batch*num_tokens, vocab_size]
        target_batch.flatten()                         # [batch*num_tokens]
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute average cross-entropy loss over a DataLoader.

    Args:
        num_batches: if set, only evaluate on this many batches (faster)
    """
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
"""
Pretraining Loop (Data Preparation + Training Loop)

Implements the full pretraining pipeline:
  1. Load and split text into train/validation sets
  2. Create DataLoaders (reuse GPTDatasetV1 from ch02)
  3. Implement the training loop with loss tracking
  4. Generate sample text after each epoch to monitor progress

Training objective:
  - For each input sequence, predict the next token at every position
  - Minimize cross-entropy loss between predictions and true next tokens
  - Optimizer: AdamW (Adam with weight decay for better generalization)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import os
import requests


# =============================================================================
# 1. Model components (reused from ch04)
# =============================================================================

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


# =============================================================================
# 2. Utility functions
# =============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """Compute cross-entropy loss for a single batch."""
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute average cross-entropy loss over a DataLoader."""
    total_loss  = 0.0
    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy decoding: repeatedly predict and append the most likely next token."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """Simple pretraining loop with periodic loss evaluation and text generation.

    Args:
        eval_freq:     evaluate every N steps
        eval_iter:     number of batches to use for loss estimation
        start_context: prompt string for sample text generation
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                   # reset gradients from previous step
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                         # compute gradients
            optimizer.step()                        # update weights
            tokens_seen += input_batch.numel()
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                    val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} | Step {global_step:06d} | "
                      f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")
                model.train()

        # Generate sample text after each epoch to monitor quality
        model.eval()
        encoded   = tokenizer.encode(start_context)
        idx       = torch.tensor(encoded).unsqueeze(0).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model, idx,
                max_new_tokens=50,
                context_size=GPT_CONFIG_124M["context_length"]
            )
        print(f"  Sample: {tokenizer.decode(token_ids.squeeze(0).tolist())!r}\n")
        model.train()

    return train_losses, val_losses, track_tokens_seen


# =============================================================================
# 3. Dataset and DataLoader (reused from ch02)
# =============================================================================

class GPTDatasetV1(Dataset):
    """Sliding window dataset for next-token prediction."""
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        self.input_ids  = []
        self.target_ids = []
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, tokenizer, batch_size, max_length, stride,
                      shuffle=True, drop_last=True, num_workers=0):
    dataset    = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader


# =============================================================================
# 4. Config
# =============================================================================

GPT_CONFIG_124M = {
    "vocab_size"    : 50257,
    "context_length": 256,   # shorter than GPT-2's 1024 for faster training
    "emb_dim"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 12,
    "drop_rate"     : 0.1,
    "qkv_bias"      : False,
}


# =============================================================================
# 5. Data preparation
# =============================================================================

def download_sample_text(filepath):
    """Download the-verdict.txt if not already present."""
    if not os.path.exists(filepath):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
    return filepath

filepath = download_sample_text(
    os.path.join(os.path.dirname(__file__), "../ch02/the-verdict.txt")
)
with open(filepath, "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer  = tiktoken.get_encoding("gpt2")
split_idx  = int(len(raw_text) * 0.9)
train_text = raw_text[:split_idx]
val_text   = raw_text[split_idx:]

train_loader = create_dataloader(
    train_text, tokenizer,
    batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True, drop_last=True
)
val_loader = create_dataloader(
    val_text, tokenizer,
    batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False, drop_last=False
)

print(f"Train batches: {len(train_loader)}")
print(f"Val   batches: {len(val_loader)}")


# =============================================================================
# 6. Run training
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

# AdamW: Adam optimizer with weight decay
# weight decay penalizes large weights → reduces overfitting
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=10,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)

print("Training complete.")
print(f"Final train loss: {train_losses[-1]:.3f}")
print(f"Final val loss  : {val_losses[-1]:.3f}")
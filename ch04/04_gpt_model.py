"""
GPT Model (GPTModel + text generation)

Assembles all components into a complete GPT-2-like language model:
  - Token embedding    : token ID → dense vector
  - Positional embedding: position → dense vector
  - Transformer Blocks : n_layers stacked TransformerBlock
  - Final LayerNorm    : stabilize output before projection
  - Output head        : project emb_dim → vocab_size (logits)

Text generation:
  - Feed a token sequence into the model → get logits
  - Pick the next token (greedy: argmax)
  - Append to sequence and repeat

Steps:
  1. Assemble GPTModel class
  2. Verify parameter count matches GPT-2 small (124M)
  3. Generate text with a simple greedy decoding loop
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
        context_vecs = (attn_weights @ values).transpose(1, 2).contiguous()
        return self.out_proj(context_vecs.view(batch_size, num_tokens, self.d_out))


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


# --- GPTModel ---
class GPTModel(nn.Module):
    """Complete GPT-2-like language model.

    Embeds tokens and positions, passes through n_layers TransformerBlocks,
    applies final LayerNorm, then projects to vocabulary logits.
    """

    def __init__(self, cfg):
        super().__init__()
        # Token embedding: token ID → emb_dim vector
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Positional embedding: position index → emb_dim vector
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of Transformer Blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer norm before output projection
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output head: project emb_dim → vocab_size (logits for each token)
        # No softmax here: CrossEntropyLoss expects raw logits
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx: [batch_size, num_tokens]  integer token IDs
        batch_size, num_tokens = in_idx.shape

        # Token embeddings: [batch_size, num_tokens, emb_dim]
        tok_embeds = self.tok_emb(in_idx)

        # Positional embeddings: [num_tokens, emb_dim] → broadcast over batch
        pos_embeds = self.pos_emb(torch.arange(num_tokens, device=in_idx.device))

        # Combine and apply dropout
        x = self.drop_emb(tok_embeds + pos_embeds)

        # Pass through all Transformer Blocks
        x = self.trf_blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary logits: [batch_size, num_tokens, vocab_size]
        logits = self.out_head(x)
        return logits

# --- Demo ---
GPT_CONFIG_124M = {
    "vocab_size"    : 50257,
    "context_length": 1024,
    "emb_dim"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 12,
    "drop_rate"     : 0.1,
    "qkv_bias"      : False,
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # disable dropout for inference

# --- Step 2: Verify parameter count ---
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # → ~124M

# --- Step 3: Text generation with greedy decoding ---
import tiktoken

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generate text by repeatedly predicting the next token.

    Args:
        model:          GPTModel instance
        idx:            [batch_size, num_tokens] input token IDs
        max_new_tokens: number of new tokens to generate
        context_size:   maximum context length the model supports
    """
    for _ in range(max_new_tokens):
        # Crop input to the last context_size tokens if sequence is too long
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # [batch, num_tokens, vocab_size]

        # Focus on the last token's logits → next token prediction
        logits = logits[:, -1, :]  # [batch, vocab_size]

        # Greedy decoding: pick the token with highest probability
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]

        # Append predicted token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # [batch, num_tokens+1]

    return idx

# Tokenize prompt and generate
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "Hello, I am"
encoded = tokenizer.encode(prompt)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # [1, num_tokens]

print(f"\nPrompt: {prompt!r}")
print(f"Encoded: {encoded}")

token_ids = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
print(f"Generated: {decoded!r}")
# Note: model is randomly initialized → output is gibberish
# Meaningful output requires pretraining on large text corpus
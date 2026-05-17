"""
Loading Pretrained GPT-2 Weights from OpenAI (Section 5.5)

Instead of training from scratch, we can load OpenAI's pretrained GPT-2 weights
into our GPTModel architecture and use it for inference immediately.

Why load pretrained weights?
  - Training GPT-2 from scratch requires massive compute (weeks on many GPUs)
  - OpenAI has released the original GPT-2 weights publicly
  - We can reuse these weights in our own GPTModel implementation
  - This is the foundation of "fine-tuning": start from pretrained, adapt to task

GPT-2 model sizes:
  - small  : 124M parameters  (12 layers, 768 emb_dim,  12 heads)
  - medium : 355M parameters  (24 layers, 1024 emb_dim, 16 heads)
  - large  : 774M parameters  (36 layers, 1280 emb_dim, 20 heads)
  - xl     : 1558M parameters (48 layers, 1600 emb_dim, 25 heads)

Steps:
  1. Download GPT-2 weights using the `gpt_download` utility
  2. Load weights into our GPTModel
  3. Generate text using the pretrained model
"""

import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
import json
import tiktoken

# --- GPT-2 weight download utility ---
def download_and_load_gpt2(model_size, models_dir):
    """Download GPT-2 weights from OpenAI and load into numpy arrays.

    Args:
        model_size: "124M", "355M", "774M", or "1558M"
        models_dir: directory to save downloaded files
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    # Files to download from OpenAI's CDN
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download each file if not already present
    for filename in filenames:
        file_url  = f"{base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(file_url, file_path)

    # Load hyperparameters
    with open(os.path.join(model_dir, "hparams.json")) as f:
        hparams = json.load(f)
    print(f"GPT-2 hparams: {hparams}")

    # Load weights using TensorFlow checkpoint reader
    # (GPT-2 weights were released in TensorFlow format)
    try:
        import tensorflow as tf
        ckpt_path = os.path.join(model_dir, "model.ckpt")
        settings  = hparams

        # Read all variable names and values
        params = {}
        for name, _ in tf.train.list_variables(ckpt_path):
            array = tf.train.load_variable(ckpt_path, name)
            params[name] = array

    except ImportError:
        raise ImportError(
            "TensorFlow is required to load GPT-2 weights.\n"
            "Install with: pip install tensorflow --break-system-packages"
        )

    return settings, params

    """
Loading Pretrained GPT-2 Weights from OpenAI (Section 5.5)

Instead of training from scratch, we can load OpenAI's pretrained GPT-2 weights
into our GPTModel architecture and use it for inference immediately.

Why load pretrained weights?
  - Training GPT-2 from scratch requires massive compute (weeks on many GPUs)
  - OpenAI has released the original GPT-2 weights publicly
  - We can reuse these weights in our own GPTModel implementation
  - This is the foundation of "fine-tuning": start from pretrained, adapt to task

GPT-2 model sizes:
  - small  : 124M parameters  (12 layers, 768 emb_dim,  12 heads)
  - medium : 355M parameters  (24 layers, 1024 emb_dim, 16 heads)
  - large  : 774M parameters  (36 layers, 1280 emb_dim, 20 heads)
  - xl     : 1558M parameters (48 layers, 1600 emb_dim, 25 heads)

Steps:
  1. Download GPT-2 weights using the download_and_load_gpt2 utility
  2. Map OpenAI weight names to our GPTModel parameter names
  3. Load weights into our GPTModel
  4. Generate text using the pretrained model
"""

import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
import json
import tiktoken


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
# 2. Download GPT-2 weights from OpenAI
# =============================================================================

def download_and_load_gpt2(model_size, models_dir):
    """Download GPT-2 weights from OpenAI and load into a params dict.

    Args:
        model_size: "124M", "355M", "774M", or "1558M"
        models_dir: directory to save downloaded files
    Returns:
        settings (dict): model hyperparameters from hparams.json
        params   (dict): weight arrays keyed by TF variable name
    """
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    base_url  = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    for filename in filenames:
        file_url  = f"{base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(file_url, file_path)
        else:
            print(f"Already exists: {filename}")

    with open(os.path.join(model_dir, "hparams.json")) as f:
        settings = json.load(f)
    print(f"\nGPT-2 hparams: {settings}")

    # Load TensorFlow checkpoint
    try:
        import tensorflow as tf
        ckpt_path = os.path.join(model_dir, "model.ckpt")
        params = {}
        for name, _ in tf.train.list_variables(ckpt_path):
            array = tf.train.load_variable(ckpt_path, name)
            params[name] = array
        print(f"Loaded {len(params)} weight tensors from checkpoint.")
    except ImportError:
        raise ImportError(
            "TensorFlow is required to load GPT-2 weights.\n"
            "Install with: pip install tensorflow --break-system-packages"
        )

    return settings, params


# =============================================================================
# 3. Map OpenAI weight names → our GPTModel parameter names
# =============================================================================

def load_weights_into_gpt(gpt, params):
    """Copy weights from OpenAI's params dict into our GPTModel.

    OpenAI uses TensorFlow naming conventions; we map them to our PyTorch names.
    Key differences:
      - TF stores attention QKV as a single matrix; we split into W_query/W_key/W_value
      - TF uses row-major layout; some weights need transposing
    """

    def assign(left, right):
        """Copy numpy array into a PyTorch parameter (in-place)."""
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
        with torch.no_grad():
            left.copy_(torch.tensor(right))

    # Token and positional embeddings
    assign(gpt.tok_emb.weight, params["model/wte"])
    assign(gpt.pos_emb.weight, params["model/wpe"])

    # Transformer blocks
    for b in range(len(gpt.trf_blocks)):
        # Attention Q, K, V weights (OpenAI stores as single [emb_dim, 3*emb_dim])
        q_w, k_w, v_w = np.split(params[f"model/h{b}/attn/c_attn/w"][0], 3, axis=-1)
        assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        assign(gpt.trf_blocks[b].att.W_key.weight,   k_w.T)
        assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # Attention Q, K, V biases
        q_b, k_b, v_b = np.split(params[f"model/h{b}/attn/c_attn/b"], 3, axis=-1)
        assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        assign(gpt.trf_blocks[b].att.W_key.bias,   k_b)
        assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # Attention output projection
        assign(gpt.trf_blocks[b].att.out_proj.weight,
               params[f"model/h{b}/attn/c_proj/w"][0].T)
        assign(gpt.trf_blocks[b].att.out_proj.bias,
               params[f"model/h{b}/attn/c_proj/b"])

        # Feed forward
        assign(gpt.trf_blocks[b].ff.layers[0].weight,
               params[f"model/h{b}/mlp/c_fc/w"][0].T)
        assign(gpt.trf_blocks[b].ff.layers[0].bias,
               params[f"model/h{b}/mlp/c_fc/b"])
        assign(gpt.trf_blocks[b].ff.layers[2].weight,
               params[f"model/h{b}/mlp/c_proj/w"][0].T)
        assign(gpt.trf_blocks[b].ff.layers[2].bias,
               params[f"model/h{b}/mlp/c_proj/b"])

        # Layer norms
        assign(gpt.trf_blocks[b].norm1.scale, params[f"model/h{b}/ln_1/g"])
        assign(gpt.trf_blocks[b].norm1.shift, params[f"model/h{b}/ln_1/b"])
        assign(gpt.trf_blocks[b].norm2.scale, params[f"model/h{b}/ln_2/g"])
        assign(gpt.trf_blocks[b].norm2.shift, params[f"model/h{b}/ln_2/b"])

    # Final layer norm
    assign(gpt.final_norm.scale, params["model/ln_f/g"])
    assign(gpt.final_norm.shift, params["model/ln_f/b"])

    # Output head shares weights with token embedding (weight tying)
    assign(gpt.out_head.weight, params["model/wte"])

    print("Weights loaded successfully.")


# =============================================================================
# 4. Text generation utility
# =============================================================================

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy decoding: repeatedly predict and append the most likely next token."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# =============================================================================
# 5. Config + run
# =============================================================================

# GPT-2 small (124M) config
# Note: qkv_bias=True because OpenAI's GPT-2 uses bias in attention projections
GPT_CONFIG_124M = {
    "vocab_size"    : 50257,
    "context_length": 1024,
    "emb_dim"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 12,
    "drop_rate"     : 0.0,   # disable dropout for inference
    "qkv_bias"      : True,  # GPT-2 uses bias in Q, K, V projections
}

# Download and load weights
models_dir = os.path.join(os.path.dirname(__file__), "gpt2_weights")
settings, params = download_and_load_gpt2("124M", models_dir)

# Build model and load pretrained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GPTModel(GPT_CONFIG_124M)
load_weights_into_gpt(model, params)
model.to(device)
model.eval()

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
prompt    = "Every effort moves you"
encoded   = tokenizer.encode(prompt)
idx       = torch.tensor(encoded).unsqueeze(0).to(device)

token_ids = generate_text_simple(
    model, idx,
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
print(f"\nPrompt   : {prompt!r}")
print(f"Generated: {decoded!r}")
# Pretrained model should produce coherent, meaningful text
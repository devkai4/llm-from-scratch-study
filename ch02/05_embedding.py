"""
Token Embeddings and Positional Embeddings

Before feeding token IDs into an LLM, they must be converted into
continuous vector representations (embeddings).

Two types of embeddings are combined:
  1. Token embedding    : converts token ID → dense vector
  2. Positional embedding: encodes the position of each token in the sequence

Final input embedding = token embedding + positional embedding

Steps:
  1. Load tokenized data using the DataLoader from 04_dataloader.py
  2. Create a token embedding layer (vocab_size × output_dim)
  3. Create a positional embedding layer (context_length × output_dim)
  4. Add them together to produce the final input embeddings
"""

import torch                                      # PyTorch core
from torch.utils.data import Dataset, DataLoader  # for batching
import tiktoken                                   # BPE tokenizer
import os                                         # for file path operations
import requests                                   # for downloading sample text

# --- Load sample text ---
def download_sample_text(filepath="the-verdict.txt"):
    """Download sample text from GitHub if not already present."""
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
    os.path.join(os.path.dirname(__file__), "the-verdict.txt")
)

with open(filepath, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Total characters: {len(raw_text)}")

# --- Reuse GPTDatasetV1 and create_dataloader_v1 from 04_dataloader.py ---
# This is a copy of the dataset and dataloader from 04_dataloader.py
# In a real project, this would be imported as a module instead of duplicated
tokenizer = tiktoken.get_encoding("gpt2")

class GPTDatasetV1(Dataset):
    """PyTorch Dataset that generates input/target pairs using a sliding window."""

    def __init__(self, txt, tokenizer, max_length, stride):
        # Tokenize the entire text into a flat list of token IDs
        # allowed_special: treat <|endoftext|> as a single token instead of raising an error
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Safety check: the text must be longer than one context window
        assert len(token_ids) > max_length, \
            "Number of tokenized inputs must be greater than max_length"

        self.input_ids = []   # stores input tensors  e.g. [A, B, C, D]
        self.target_ids = []  # stores target tensors e.g. [B, C, D, E]

        # Slide a window of size max_length over the token sequence
        # range(start, stop, step):
        #   start = 0               → begin at the first token
        #   stop  = len - max_length → stop before the last incomplete window
        #   step  = stride          → how far to move the window each iteration
        for i in range(0, len(token_ids) - max_length, stride):
            # input window: tokens at positions [i, i+max_length)
            input_chunk  = token_ids[i : i + max_length]

            # target window: same window shifted 1 position to the right
            # this represents the "next token" for each position in input_chunk
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # Convert Python lists to PyTorch tensors so DataLoader can batch them
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Required by DataLoader: returns the total number of input/target pairs
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Required by DataLoader: returns the idx-th input/target pair
        # DataLoader calls this method internally when building each batch
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """Create a DataLoader that yields batched input/target pairs.

    Args:
        txt:         raw text string to tokenize and sample from
        batch_size:  number of input/target pairs per batch
        max_length:  context window size (number of tokens per sample)
        stride:      how far the sliding window moves each step
        shuffle:     if True, randomize sample order each epoch
        drop_last:   if True, discard the final batch if it is smaller than batch_size
        num_workers: number of subprocesses for data loading (0 = main process only)
    """
    # Instantiate the dataset: tokenizes txt and builds all input/target pairs
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Wrap dataset in DataLoader for automatic batching and shuffling
    # DataLoader internally calls dataset.__getitem__() to collect samples into a batch
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# Fetch one batch to use as input for the embedding layers below
# batch_size=8: fetch 8 samples at once
# max_length=4: each sample contains 4 tokens (kept small for readability)
# stride=max_length: no overlap between samples (stride = window size)
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

# iter(): converts DataLoader into a Python iterator
# next(): fetches the first batch
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(f"Token IDs shape: {inputs.shape}")   # → torch.Size([8, 4])  [batch_size, max_length]
print(f"Token IDs:\n{inputs}")

# --- 2.7 Token Embedding Layer ---
# An embedding layer is a lookup table: token ID → dense vector
# vocab_size:  number of unique tokens in the GPT-2 vocabulary
# output_dim:  the size of the vector each token is mapped to
vocab_size = 50257  # GPT-2 BPE vocabulary size
output_dim = 256    # embedding dimension (GPT-2 actual uses 768, kept small here)

# torch.nn.Embedding(vocab_size, output_dim) creates a weight matrix of shape:
# [vocab_size, output_dim] = [50257, 256]
# each row corresponds to one token ID → its vector representation
# these weights are randomly initialized and updated during training
torch.manual_seed(123)  # fix random seed for reproducibility
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(f"Embedding weight shape: {token_embedding_layer.weight.shape}")
# → torch.Size([50257, 256])

# Pass the input batch through the embedding layer
# inputs shape:           [8, 4]        (batch_size × max_length)
# token_embeddings shape: [8, 4, 256]   (batch_size × max_length × output_dim)
# Each token ID is replaced by its corresponding 256-dimensional vector
token_embeddings = token_embedding_layer(inputs)
print(f"Token embeddings shape: {token_embeddings.shape}")
# → torch.Size([8, 4, 256])

# --- 2.8 Positional Embedding Layer ---
# Problem: the token embedding layer maps the same token ID to the same vector
# regardless of its position in the sequence.
# e.g. "the" at position 0 and "the" at position 3 would get identical vectors.
# Solution: add a positional embedding that encodes each token's position.

# context_length: the maximum number of tokens in a sequence
# output_dim:     must match the token embedding dimension so they can be added
context_length = max_length  # 4 (same as the sliding window size)

# torch.nn.Embedding(context_length, output_dim) creates a weight matrix of shape:
# [context_length, output_dim] = [4, 256]
# each row corresponds to one position (0, 1, 2, 3) → its vector representation
# these weights are randomly initialized and updated during training
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# torch.arange(max_length) → tensor([0, 1, 2, 3])
# passes position indices [0, 1, 2, 3] through the embedding layer
# pos_embeddings shape: [4, 256]  (context_length × output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(f"Positional embeddings shape: {pos_embeddings.shape}")
# → torch.Size([4, 256])

# --- Combine token and positional embeddings ---
# token_embeddings shape: [8, 4, 256]
# pos_embeddings shape:   [4, 256]  ← broadcast across batch dimension
# input_embeddings shape: [8, 4, 256]
# PyTorch automatically broadcasts pos_embeddings to match the batch dimension
input_embeddings = token_embeddings + pos_embeddings
print(f"Input embeddings shape: {input_embeddings.shape}")
# → torch.Size([8, 4, 256])

print("\nDone. input_embeddings are ready to be fed into the LLM.")
"""
Data Sampling with a Sliding Window (GPTDatasetV1 + DataLoader)

LLMs are trained to predict the next token in a sequence.
This module prepares input/target pairs using a sliding window approach:
  - input:  [token1, token2, token3, token4]
  - target: [token2, token3, token4, token5]  ← shifted by 1

Steps:
  1. Tokenize raw text using BPE (tiktoken)
  2. Slide a window over the token sequence to create input/target pairs
  3. Wrap in a PyTorch Dataset and DataLoader for batch training
"""

import torch                                    # PyTorch core
from torch.utils.data import Dataset, DataLoader  # for batching and shuffling
import tiktoken                                 # BPE tokenizer
import os                                       # for file path operations
import requests                                 # for downloading sample text

# --- Load sample text ---
def download_sample_text(filepath="the-verdict.txt"):
    """Download sample text from GitHub if not already present."""
    if not os.path.exists(filepath):  # skip download if file already exists
        url = (  # implicit string concatenation
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        response = requests.get(url, timeout=30)  # Send GET request to GitHub
        response.raise_for_status()  # raise an error if the request failed
        with open(filepath, "wb") as f:  # write bytes to local file
            f.write(response.content)
    return filepath

# Build absolute path to the-verdict.txt in the same directory as this script
filepath = download_sample_text(
    os.path.join(os.path.dirname(__file__), "the-verdict.txt")
)

# Open the file in read mode and load all text into raw_text
with open(filepath, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Total characters: {len(raw_text)}")

# --- Tokenize the entire text using GPT-2 BPE tokenizer ---
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
print(f"Total tokens: {len(enc_text)}")  # → 5145

# --- GPTDatasetV1 ---
class GPTDatasetV1(Dataset):
    """PyTorch Dataset that generates input/target pairs using a sliding window.

    For each window position i:
      input:  token_ids[i : i + max_length]
      target: token_ids[i + 1 : i + max_length + 1]  ← shifted by 1
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        # Tokenize the entire text into a flat list of token IDs
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, \
            "Number of tokenized inputs must be greater than max_length"

        self.input_ids = []
        self.target_ids = []

        # Slide a window of size max_length over the token sequence
        # stride controls how far the window moves each step
        # stride = 1   → maximum overlap (more training pairs, more overfitting risk)
        # stride = max_length → no overlap (fewer pairs, less overfitting risk)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]      # input window
            target_chunk = token_ids[i + 1 : i + max_length + 1]  # shifted by 1
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Return the total number of input/target pairs."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Return a single input/target pair by index."""
        return self.input_ids[idx], self.target_ids[idx]


# --- create_dataloader_v1 ---
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """Create a DataLoader that yields batched input/target pairs.

    Args:
        txt:        raw text string
        batch_size: number of samples per batch
        max_length: context window size (number of tokens per sample)
        stride:     step size of the sliding window
        shuffle:    whether to shuffle the dataset each epoch
        drop_last:  whether to drop the last incomplete batch
        num_workers: number of worker processes for data loading
    """
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Wrap dataset in DataLoader for batching and shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,   # drop last batch if smaller than batch_size
        num_workers=num_workers
    )
    return dataloader

    # --- Demo ---
# Test with batch_size=1 and context_size=4 to see individual pairs
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)        # convert dataloader to iterator
first_batch = next(data_iter)       # fetch first batch
second_batch = next(data_iter)      # fetch second batch

print("--- batch_size=1, max_length=4, stride=1 ---")
print(f"First batch input : {first_batch[0]}")
print(f"First batch target: {first_batch[1]}")
print(f"Second batch input : {second_batch[0]}")
print(f"Second batch target: {second_batch[1]}")

# Test with batch_size=8 and stride=max_length (no overlap between batches)
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("\n--- batch_size=8, max_length=4, stride=4 ---")
print(f"Inputs:\n{inputs}")
print(f"Targets:\n{targets}")
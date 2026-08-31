"""
Ch06 - Fine-tuning for Classification: Dataset & DataLoader

Prepares the balanced SMS spam CSVs (from 01_data_preparation.py) for training
a *classification* model. Unlike the pretraining DataLoader in ch02/ch05
(which chops one long text stream into fixed-size windows), here each row is
one SMS message = one sample, and messages vary in length.

Key new concept: PADDING
  - PyTorch batches require every sample in a batch to have the same length
  - SMS messages have different token counts
  - Solution: pad all messages to a common length using the padding token

GPT-2 has no dedicated padding token, so we reuse <|endoftext|> (ID 50256),
following the book's approach.

Steps:
  1. Tokenize each message with tiktoken (GPT-2 BPE)
  2. Determine max_length from the training set
  3. Pad (or truncate) every message to max_length
  4. Wrap in a PyTorch Dataset returning (input_ids, label)
  5. Build DataLoaders for train / val / test
"""

import os                                          # for file path operations
import torch                                       # PyTorch core
from torch.utils.data import Dataset, DataLoader   # for batching
import tiktoken                                    # GPT-2 BPE tokenizer
import pandas as pd                                # for reading the CSV splits


# --- SpamDataset ---
class SpamDataset(Dataset):
    """PyTorch Dataset for SMS spam classification.

    Each item is a single SMS message (as padded token IDs) paired with its
    integer label (0 = ham, 1 = spam).

    All messages are padded/truncated to a common `max_length` so they can be
    stacked into batches. If max_length is None, it is inferred from the longest
    message in this dataset (used for the training set); val/test datasets are
    passed the training max_length so all splits share the same input width.
    """

    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        # Load the split (train.csv / validation.csv / test.csv)
        self.data = pd.read_csv(csv_file)

        # Tokenize every message up front (stored as list of ID lists)
        # This is done once at init so __getitem__ stays cheap.
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # Decide the common sequence length
        if max_length is None:
            # Training set: use the longest message in this dataset
            self.max_length = self._longest_encoded_length()
        else:
            # Val/test set: reuse the training max_length
            # Truncate any message longer than max_length so shapes stay valid
            self.max_length = max_length
            self.encoded_texts = [
                enc[:self.max_length] for enc in self.encoded_texts
            ]

        # Pad every message to max_length with the padding token
        # short message + [pad, pad, ...] → exactly max_length tokens
        self.encoded_texts = [
            enc + [pad_token_id] * (self.max_length - len(enc))
            for enc in self.encoded_texts
        ]

    def _longest_encoded_length(self):
        """Return the token count of the longest message in this dataset."""
        return max(len(enc) for enc in self.encoded_texts)

    def __getitem__(self, index):
        """Return one (input_ids, label) pair as tensors.

        DataLoader calls this internally to assemble each batch.
        """
        encoded = self.encoded_texts[index]         # padded list of token IDs
        label   = self.data.iloc[index]["Label"]    # 0 or 1
        return (
            torch.tensor(encoded, dtype=torch.long),  # input IDs
            torch.tensor(label,   dtype=torch.long),  # class label
        )

    def __len__(self):
        """Required by DataLoader: number of samples in this split."""
        return len(self.data)


# --- create_dataloaders ---
def create_dataloaders(train_csv, val_csv, test_csv, tokenizer,
                       batch_size=8, num_workers=0):
    """Build train / val / test DataLoaders sharing a common max_length.

    The training set defines max_length; val and test reuse it so every split
    produces batches of identical width.
    """
    # Training set: infer max_length from its own longest message
    train_dataset = SpamDataset(train_csv, tokenizer, max_length=None)

    # Val/test: reuse the training max_length for consistent input shapes
    val_dataset  = SpamDataset(val_csv,  tokenizer,
                               max_length=train_dataset.max_length)
    test_dataset = SpamDataset(test_csv, tokenizer,
                               max_length=train_dataset.max_length)

    # drop_last=True on train: avoid a smaller final batch during training
    #   (keeps batch shapes uniform, slightly stabilizes training)
    # drop_last=False on val/test: we want to evaluate on every sample
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, train_dataset.max_length


# --- Config ---
BASE_DIR = os.path.dirname(__file__)
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VAL_CSV   = os.path.join(BASE_DIR, "validation.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test.csv")


# --- Demo ---
tokenizer = tiktoken.get_encoding("gpt2")

train_loader, val_loader, test_loader, max_length = create_dataloaders(
    TRAIN_CSV, VAL_CSV, TEST_CSV, tokenizer, batch_size=8
)

# max_length is inferred from the longest training message
print(f"Detected max_length (from train): {max_length}")

# Inspect one training batch to verify shapes
# input_batch:  [batch_size, max_length]  padded token IDs
# target_batch: [batch_size]              class labels (0/1)
input_batch, target_batch = next(iter(train_loader))
print(f"Input batch shape : {input_batch.shape}")   # → [8, max_length]
print(f"Target batch shape: {target_batch.shape}")  # → [8]
print(f"Sample labels     : {target_batch.tolist()}")

# Report the number of batches per split
print(f"\nTrain batches: {len(train_loader)}")
print(f"Val   batches: {len(val_loader)}")
print(f"Test  batches: {len(test_loader)}")
"""
Simple Regex-based Tokenizer (SimpleTokenizerV2)

Extends SimpleTokenizerV1 by adding special tokens:
  - <|unk|>        : represents out-of-vocabulary words
  - <|endoftext|>  : marks the boundary between unrelated texts

Steps:
  1. Split raw text into tokens using regex
  2. Build a vocabulary with special tokens appended
  3. Encode (text -> IDs) replacing unknown tokens with <|unk|>
  4. Decode (IDs -> text)
"""

import os      # for file path operations
import re      # for regex-based text splitting
import requests  # for downloading sample text

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

# Print total character count to verify the file was loaded correctly
print(f"Total characters: {len(raw_text)}")

# --- Preprocessing: split text into tokens ---
def preprocess(text):
    """Split text into tokens by punctuation and whitespace."""
    # r'' = raw string: backslashes are passed as-is to the regex engine
    # ([,.:;?_!"()\'"] = capture group: keeps punctuation as separate tokens
    # |-- = match literal double dash
    # |\s) = match any whitespace character (space, newline, tab)
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    # t.strip() removes surrounding whitespace; empty/whitespace-only strings are falsy and filtered out
    return [t for t in tokens if t.strip()]

# --- Build vocabulary ---
# preprocess(raw_text): tokenize the entire text
# set(...): remove duplicate tokens
# sorted(...): sort alphabetically so token IDs are consistent across runs
all_tokens = sorted(set(preprocess(raw_text)))

# Append special tokens at the end of the vocabulary
# <|endoftext|>: marks the boundary between unrelated texts
# <|unk|>: represents any token not found in the vocabulary
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab_size = len(all_tokens)
print(f"Vocabulary size: {vocab_size}")  # 1130 + 2 = 1132

# enumerate(all_tokens): yields (idx, token) pairs → (0, "!"), (1, ","), ...
# token_to_id: maps each token string to its integer ID  e.g. {"!": 0, ",": 1, ...}
token_to_id = {token: idx for idx, token in enumerate(all_tokens)}

# invert token_to_id: swap key and value to enable ID -> token lookup
# .items(): returns (token, idx) pairs from token_to_id
# id_to_token: maps each integer ID back to its token string  e.g. {0: "!", 1: ",", ...}
id_to_token = {idx: token for token, idx in token_to_id.items()}

# --- SimpleTokenizerV2 ---
class SimpleTokenizerV2:
    """An improved tokenizer that handles out-of-vocabulary tokens.

    Unknown tokens are replaced with <|unk|> instead of raising KeyError.
    <|endoftext|> can be used to separate unrelated texts.
    """

    def __init__(self, vocab):
        # Store the original vocab dict as-is for token -> ID lookup
        # e.g. {"!": 0, ..., "<|endoftext|>": 1130, "<|unk|>": 1131}
        self.str_to_int = vocab

        # Invert vocab to enable ID -> token lookup
        # vocab.items() yields (tok, idx) pairs → swap to (idx, tok)
        self.int_to_str = {idx: tok for tok, idx in vocab.items()}

    def encode(self, text):
        """Convert text to a list of token IDs.

        Unknown tokens are replaced with <|unk|> instead of raising KeyError.
        """
        # Split text into tokens using the same regex as preprocessing
        # e.g. "Hello, world." → ["Hello", ",", "world", "."]
        tokens = preprocess(text)

        # Replace any token not in the vocabulary with <|unk|>
        # e.g. ["Hello", ",", "world"] → ["<|unk|>", ",", "world"]
        tokens = [
            t if t in self.str_to_int else "<|unk|>" for t in tokens
        ]

        # Map each token to its integer ID
        # e.g. ["<|unk|>", ",", "world"] → [1131, 1, 5]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids):
        """Convert a list of token IDs back to a string."""
        # Look up each ID in int_to_str and join with spaces
        # e.g. [1131, 1, 5] → "<|unk|> , world"
        text = " ".join(self.int_to_str[i] for i in ids)

        # Remove the extra space inserted before punctuation by the join above
        # e.g. "<|unk|> , world ." → "<|unk|>, world."
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text

# --- Demo ---
# Instantiate the tokenizer with the extended vocabulary
tokenizer = SimpleTokenizerV2(token_to_id)

# Join two unrelated texts with <|endoftext|> as separator
text1 = "Hello, do you like tea?"           # "Hello" is out-of-vocabulary → <|unk|>
text2 = "In the sunlit terraces of the palace."  # "palace" is out-of-vocabulary → <|unk|>
text = " <|endoftext|> ".join((text1, text2))
print(f"Original: {text}")

# Encode: text → list of integer IDs
encoded = tokenizer.encode(text)
print(f"Encoded : {encoded}")

# Decode: list of integer IDs → text
decoded = tokenizer.decode(encoded)
print(f"Decoded : {decoded}")
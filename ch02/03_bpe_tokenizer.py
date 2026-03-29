"""
BPE Tokenizer using tiktoken (GPT-2)

Unlike SimpleTokenizerV1/V2 which use a fixed vocabulary built from a single text file,
BPE (Byte Pair Encoding) tokenizer breaks unknown words into subword units,
so out-of-vocabulary words never occur.

Steps:
  1. Load the GPT-2 BPE tokenizer via tiktoken
  2. Encode (text -> IDs)
  3. Decode (IDs -> text)
"""

import tiktoken  # OpenAI's BPE tokenizer library

# --- Load GPT-2 BPE tokenizer ---
# tiktoken provides pretrained tokenizers; "gpt2" has a vocabulary of 50,257 tokens
tokenizer = tiktoken.get_encoding("gpt2")
print(f"Vocabulary size: {tokenizer.n_vocab}")  # → 50257

# --- Demo: basic encode and decode ---
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)
print(f"Original: {text}")

# Encode: text → list of integer IDs
# allowed_special: by default tiktoken raises an error for special tokens
# passing {"<|endoftext|>"} tells the tokenizer to encode it as a single token
encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(f"Encoded : {encoded}")

# Decode: list of integer IDs → text
decoded = tokenizer.decode(encoded)
print(f"Decoded : {decoded}")
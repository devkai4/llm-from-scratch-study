"""
Simple Regex-based Tokenizer (SimpleTokenizerV1)

Steps:
  1. Split raw text into tokens using regex
  2. Build a vocabulary (token <-> integer ID mapping)
  3. Encode (text -> IDs) and Decode (IDs -> text)
"""

import os # for file operations
import re # for regex operations
import requests # for downloading sample text

# --- Load sample text ---
def download_sample_text(filepath="the-verdict.txt"):
    """Download sample text from GitHub if not already present."""
    if not os.path.exists(filepath):  # skip download if file already exists
        url = (　# implicit string concatenation
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        response = requests.get(url, timeout=30)  # Send GET request to GitHub and store the HTTP response
        response.raise_for_status()  # raise an error if the request failed
        with open(filepath, "wb") as f:  # Save downloaded bytes to a local file in binary mode
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
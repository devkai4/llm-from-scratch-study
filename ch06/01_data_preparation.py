"""
Ch06 - Fine-tuning for Classification: Data Preparation

Chapter 6 fine-tunes the pretrained GPT model for a *classification* task:
deciding whether an SMS message is "spam" or "ham" (not spam).

This first step prepares the dataset:
  1. Download the SMS Spam Collection dataset (UCI ML Repository)
  2. Load it into a pandas DataFrame
  3. Create a *balanced* dataset (equal number of spam and ham messages)
  4. Split into train / validation / test sets
  5. Save each split as a CSV file for later use

Why balance the dataset?
  - The raw data is heavily skewed toward "ham" (~87% ham, ~13% spam)
  - A model trained on imbalanced data can reach high accuracy by always
    predicting the majority class -> useless as a classifier
  - Undersampling "ham" to match the "spam" count removes this shortcut
"""

import os            # for file path operations
import zipfile       # for extracting the downloaded .zip archive
import requests      # for downloading the dataset
import pandas as pd  # for loading and manipulating tabular data


# --- Download and extract the dataset ---
def _download_zip(url, zip_path, extract_dir, data_file_path):
    """Download the .zip from a single URL and extract the data file."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # The archive contains "SMSSpamCollection" (no extension) -> rename to .tsv
    original_file = os.path.join(extract_dir, "SMSSpamCollection")
    os.rename(original_file, data_file_path)


def download_and_extract(url, zip_path, extract_dir, data_file_path, backup_url=None):
    """Download the SMS spam .zip and extract the tab-separated data file.

    Falls back to backup_url if the primary URL fails, because the UCI
    repository is frequently unavailable (timeouts / 403 / cert errors).
    Skips download entirely if the data file already exists.
    """
    if os.path.exists(data_file_path):
        print(f"{data_file_path} already exists. Skipping download.")
        return

    try:
        _download_zip(url, zip_path, extract_dir, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        # Primary (UCI) URL failed -> try the backup mirror if provided
        if backup_url is None:
            raise
        print(f"Primary URL failed: {e}\nTrying backup URL...")
        _download_zip(backup_url, zip_path, extract_dir, data_file_path)

    print(f"Downloaded and extracted to {data_file_path}")


# --- Create a balanced dataset ---
def create_balanced_dataset(df):
    """Undersample the majority class ("ham") to match the "spam" count.

    Returns a new DataFrame with an equal number of spam and ham rows.
    """
    # Count how many spam messages exist
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample the same number of ham messages
    # random_state=123 -> reproducible sampling
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Concatenate the sampled ham rows with all spam rows
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


# --- Split into train / validation / test ---
def random_split(df, train_frac, val_frac):
    """Shuffle and split a DataFrame into train / val / test subsets.

    test_frac is implied: 1 - train_frac - val_frac
    """
    # Shuffle the entire DataFrame (frac=1 -> return all rows in random order)
    # reset_index(drop=True): renumber rows 0..N-1 after shuffling
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Compute integer split boundaries
    train_end = int(len(df) * train_frac)
    val_end   = train_end + int(len(df) * val_frac)

    # Slice into three subsets
    train_df = df[:train_end]
    val_df   = df[train_end:val_end]
    test_df  = df[val_end:]
    return train_df, val_df, test_df


# --- Config ---
BASE_DIR    = os.path.dirname(__file__)
URL         = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
BACKUP_URL  = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms+spam+collection.zip"
ZIP_PATH    = os.path.join(BASE_DIR, "sms_spam_collection.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "sms_spam_collection")
DATA_FILE   = os.path.join(EXTRACT_DIR, "SMSSpamCollection.tsv")


# --- Data preparation ---
# Step 1: download + extract (with backup URL fallback)
download_and_extract(URL, ZIP_PATH, EXTRACT_DIR, DATA_FILE, backup_url=BACKUP_URL)

# Step 2: load into a DataFrame
# sep="\t"    : the file is tab-separated
# header=None : the file has no header row
# names=[...] : assign column names manually
df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["Label", "Text"])
print(f"Total messages: {len(df)}")
print("Label distribution (raw):")
print(df["Label"].value_counts())

# Step 3: balance the dataset
balanced_df = create_balanced_dataset(df)
print("\nLabel distribution (balanced):")
print(balanced_df["Label"].value_counts())

# Step 4: map string labels to integers (ham -> 0, spam -> 1)
# The model outputs numeric class indices, so labels must be numeric
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# Step 5: split and save
train_df, val_df, test_df = random_split(balanced_df, train_frac=0.7, val_frac=0.1)
print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Save each split to CSV (index=False -> don't write the row index column)
train_df.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(BASE_DIR, "validation.csv"), index=False)
test_df.to_csv(os.path.join(BASE_DIR, "test.csv"), index=False)
print("Saved train.csv, validation.csv, test.csv")
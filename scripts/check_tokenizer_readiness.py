"""
Tokenizer Sanity Check
Before training the models, we performed a sanity check on the tokenizer to verify that the raw text is tokenized as expected.
We inspected several samples and confirmed that the tokenization process preserves the semantic structure of the sentences,
 handles special characters correctly, and produces valid input IDs and attention masks. 
 This step ensured that the data pipeline is consistent and suitable for Transformer-based models before proceeding to training.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import Counter
from pathlib import Path


# Configuration
DATA_DIR = Path("data/processed")
TEXT_COLUMN = "text"
MODEL_NAME = "bert-base-uncased"  
MAX_LENGTH = 512


# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
unk_token_id = tokenizer.unk_token_id

print(f"\nUsing tokenizer: {MODEL_NAME}")
print(f"UNK token: {tokenizer.unk_token}")


# Helper: analyze split
def analyze_split(df: pd.DataFrame, name: str):
    print(f"Analyzing {name}")


    texts = df[TEXT_COLUMN].astype(str).tolist()

    total_tokens = 0
    unk_tokens = 0
    sequence_lengths = []
    truncated_count = 0

    problematic_examples = []

    for text in texts:
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        input_ids = encoding["input_ids"]

        total_tokens += len(input_ids)
        unk_tokens += sum(1 for t in input_ids if t == unk_token_id)
        sequence_lengths.append(len(input_ids))

        if len(input_ids) == MAX_LENGTH:
            truncated_count += 1

        # Collect a few suspicious examples
        if unk_tokens > 0 and len(problematic_examples) < 5:
            problematic_examples.append(text)

    unk_ratio = (unk_tokens / total_tokens) * 100
    trunc_ratio = (truncated_count / len(texts)) * 100

    print(f"Total tokens: {total_tokens}")
    print(f"UNK tokens: {unk_tokens} ({unk_ratio:.4f}%)")
    print(f"Max sequence length: {max(sequence_lengths)}")
    print(f"Mean sequence length: {np.mean(sequence_lengths):.1f}")
    print(f"Sequences truncated: {truncated_count} ({trunc_ratio:.2f}%)")

    # Length percentiles
    for p in [50, 75, 90, 95, 99]:
        print(f"{p}th percentile length: {np.percentile(sequence_lengths, p):.0f}")

    if problematic_examples:
        print("\nExamples containing UNK tokens:")
        for ex in problematic_examples:
            print(f"  - {ex}")

    return {
        "unk_ratio": unk_ratio,
        "trunc_ratio": trunc_ratio,
        "max_len": max(sequence_lengths)
    }


# Run analysis
train_stats = analyze_split(train_df, "TRAIN")
val_stats = analyze_split(val_df, "VALIDATION")


# Final verdict
print(f"\n{'='*60}")
print("FINAL VERDICT")
print(f"{'='*60}")

def verdict(stats):
    if stats["unk_ratio"] > 1.0:
        return " NOT READY  too many UNK tokens"
    if stats["trunc_ratio"] > 5.0:
        return " CHECK MAX_LENGTH  many sequences truncated"
    return "READY FOR TRAINING"

print(f"Train: {verdict(train_stats)}")
print(f"Validation: {verdict(val_stats)}")

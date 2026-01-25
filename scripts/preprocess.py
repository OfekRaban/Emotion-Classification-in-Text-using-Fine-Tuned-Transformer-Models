"""
Preprocessing script for Emotion Detection using Transformer Models (BERT, ALBERT, DeBERTa).

This script applies minimal and non-aggressive preprocessing since transformer models
have their own tokenizers that are designed to handle raw text. Aggressive preprocessing
(like stemming, lemmatization, or heavy text normalization) can actually hurt performance
as it removes information the models were pre-trained to understand.

Preprocessing steps:
1. Remove duplicate samples
2. Normalize whitespace (collapse multiple spaces, strip leading/trailing)
3. Basic text validation (non-empty, reasonable length)

Note: No shuffling is done here - shuffling should be handled by the DataLoader during training.
The preprocessed data maintains the original order and structure suitable for
transformer tokenizers (BertTokenizer, AlbertTokenizer, DebertaTokenizer).
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Emotion label mapping for reference
EMOTION_LABELS = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}


def normalize_whitespace(text: str) -> str:

    text = re.sub(r'[\t\n\r]+', ' ', text)   # Replace (pattern(tabs/newlines),replacement (with space),text).r=regex,[] -one of those element., + -one or more
    text = re.sub(r' +', ' ', text) # Collapse multiple spaces into one
    text = text.strip() # move whitespace from beginning and end of text
    return text


def validate_text(text: str, min_length: int = 1, max_length: int = 512) -> bool:

    if not isinstance(text, str):  #check if text is string
        return False
    if len(text) < min_length: #check if length of text is less than min length
        return False
    if len(text) > max_length: #check if length of text is greater than max length
        return False
    return True


def preprocess_text(text: str) -> str:
    """
    Apply minimal preprocessing to text for transformer models.

    We intentionally keep preprocessing minimal because:
    1. Transformer tokenizers (WordPiece, SentencePiece) handle subword tokenization
    2. Pre-trained models learned from natural text with punctuation, casing, etc.
    3. Aggressive preprocessing can remove semantic information

    Args:
        text: Raw input text

    Returns:
        Minimally preprocessed text
    """
    text = normalize_whitespace(text)
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    remove_duplicates: bool = True,
    min_text_length: int = 1,
    max_text_length: int = 512
) -> pd.DataFrame:
    """
    Preprocess a dataframe containing text and labels.
    Maintains original order - no shuffling.

    Args:
        df: Input dataframe
        text_column: Name of the text column
        label_column: Name of the label column
        remove_duplicates: Whether to remove duplicate texts
        min_text_length: Minimum text length to keep
        max_text_length: Maximum text length to keep

    Returns:
        Preprocessed dataframe (same order as input)
    """
    logger.info(f"Starting preprocessing. Input shape: {df.shape}")

    # Create a copy to avoid modifying original
    df = df.copy()

    # Track original count
    original_count = len(df)

    # Remove rows with missing text or labels
    df = df.dropna(subset=[text_column, label_column])
    if len(df) < original_count:
        logger.info(f"Removed {original_count - len(df)} rows with missing values")

    # Apply text preprocessing
    df[text_column] = df[text_column].apply(preprocess_text)

    # Remove duplicates (keeping first occurrence)
    if remove_duplicates:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=[text_column], keep='first')
        if len(df) < before_dedup:
            logger.info(f"Removed {before_dedup - len(df)} duplicate texts")

    # Filter by text length
    before_filter = len(df)
    valid_length = df[text_column].apply(
        lambda x: validate_text(x, min_text_length, max_text_length)
    )
    df = df[valid_length]
    if len(df) < before_filter:
        logger.info(f"Removed {before_filter - len(df)} texts outside length bounds [{min_text_length}, {max_text_length}]")

    # Ensure labels are integers
    df[label_column] = df[label_column].astype(int)

    # Reset index (maintains original order)
    df = df.reset_index(drop=True)

    logger.info(f"Preprocessing complete. Output shape: {df.shape}")

    return df


def print_dataset_statistics(df: pd.DataFrame, name: str, label_column: str = 'label') -> None:
    """
    Print statistics about a dataset.

    Args:
        df: Dataframe to analyze
        name: Name of the dataset for display
        label_column: Name of the label column
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset: {name}")
    logger.info(f"{'='*50}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"\nLabel distribution:")

    label_counts = df[label_column].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        emotion = EMOTION_LABELS.get(label, 'unknown')
        logger.info(f"  {label} ({emotion}): {count} ({percentage:.1f}%)")

    logger.info(f"\nText length statistics:")
    text_lengths = df['text'].str.len()
    logger.info(f"  Min: {text_lengths.min()}")
    logger.info(f"  Max: {text_lengths.max()}")
    logger.info(f"  Mean: {text_lengths.mean():.1f}")
    logger.info(f"  Median: {text_lengths.median():.1f}")


def main(raw_data_dir: str, processed_data_dir: str) -> None:
    """
    Main preprocessing pipeline.
    Preprocesses train and validation data without any splitting or shuffling.

    Args:
        raw_data_dir: Directory containing raw CSV files
        processed_data_dir: Directory to save processed files
    """
    raw_path = Path(raw_data_dir)
    processed_path = Path(processed_data_dir)

    # Create output directory if it doesn't exist
    processed_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    logger.info("Loading raw data...")
    train_df = pd.read_csv(raw_path / 'train.csv')
    val_df = pd.read_csv(raw_path / 'validation.csv')

    logger.info(f"Loaded train: {len(train_df)} samples")
    logger.info(f"Loaded validation: {len(val_df)} samples")

    # Preprocess training data
    logger.info("\nPreprocessing training data...")
    train_df = preprocess_dataframe(train_df)

    # Preprocess validation data
    logger.info("\nPreprocessing validation data...")
    val_df = preprocess_dataframe(val_df)

    # Print statistics
    print_dataset_statistics(train_df, "Train")
    print_dataset_statistics(val_df, "Validation")

    # Save processed data
    logger.info("\nSaving processed data...")
    train_df.to_csv(processed_path / 'train.csv', index=False)
    val_df.to_csv(processed_path / 'validation.csv', index=False)

    logger.info(f"\nProcessed files saved to: {processed_path}")
    logger.info("Files created:")
    logger.info(f"  - train.csv ({len(train_df)} samples)")
    logger.info(f"  - validation.csv ({len(val_df)} samples)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess emotion detection data for transformer models'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='Directory containing raw data (default: data/raw)'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data (default: data/processed)'
    )

    args = parser.parse_args()

    main(
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir
    )

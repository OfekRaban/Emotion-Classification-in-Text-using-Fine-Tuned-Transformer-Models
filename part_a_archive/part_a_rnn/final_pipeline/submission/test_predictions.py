#!/usr/bin/env python3
"""
Emotion Detection - Test Set Prediction Script
===============================================

This script performs emotion classification on a test CSV file using both the best
GRU and LSTM models from our ablation study. The preprocessing pipeline is identical
to the training pipeline to ensure consistency.

Best Models:
    - GRU: Learning Rate 0.01, Validation Accuracy 92.04%, Macro F1 0.902
    - LSTM: RNN Units 128, Validation Accuracy 92.59%, Macro F1 0.906

Author: Ofek Raban & Ron Gabay
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import re
import json
from typing import Dict, List, Tuple
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, f1_score)


class TextPreprocessor:
    """
    Text preprocessing pipeline for emotion detection.

    Applies the same preprocessing steps used during model training:
        1. Contraction expansion
        2. Elongation normalization
        3. URL, email, mention removal
        4. Lowercasing and punctuation removal

    This ensures consistency between training and inference.
    """

    def __init__(self):
        """Initialize the preprocessor with contraction mappings."""
        # Standard English contractions mapping for expansion
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have",
            "couldn't": "could not", "didn't": "did not",
            "doesn't": "does not", "don't": "do not", "hadn't": "had not",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would",
            "he'll": "he will", "he's": "he is", "how'd": "how did",
            "how'll": "how will", "how's": "how is", "i'd": "i would",
            "i'll": "i will", "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would", "it'll": "it will",
            "it's": "it is", "let's": "let us", "mustn't": "must not",
            "shan't": "shall not", "she'd": "she would", "she'll": "she will",
            "she's": "she is", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what'll": "what will",
            "what're": "what are", "what's": "what is",
            "what've": "what have", "where's": "where is",
            "who'll": "who will", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }

    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.

        Args:
            text: Input text string

        Returns:
            Text with expanded contractions

        Example:
            "I can't believe it's happening" -> "I cannot believe it is happening"
        """
        for contraction, expansion in self.contractions.items():
            # Use word boundaries to avoid partial matches
            text = re.sub(r'\b' + contraction + r'\b', expansion, text,
                         flags=re.IGNORECASE)
        return text

    def normalize_elongations(self, text: str) -> str:
        """
        Normalize elongated words to reduce noise.

        Args:
            text: Input text string

        Returns:
            Text with normalized elongations

        Example:
            "sooooo happyyyy" -> "soo happyy"
        """
        # Replace 3+ consecutive identical characters with 2
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def clean_text(self, text: str) -> str:
        """
        Complete text cleaning pipeline.

        Applies all preprocessing steps in sequence:
            1. Handle missing values
            2. Convert to lowercase
            3. Expand contractions
            4. Normalize elongations
            5. Remove URLs, emails, mentions, hashtags
            6. Remove numbers
            7. Remove punctuation
            8. Remove extra whitespace

        Args:
            text: Raw input text

        Returns:
            Cleaned and preprocessed text
        """
        # Handle missing values
        if pd.isna(text):
            return ""

        # Convert to string and lowercase
        text = str(text).lower()

        # Expand contractions
        text = self.expand_contractions(text)

        # Normalize elongations
        text = self.normalize_elongations(text)

        # Remove URLs (http/https/www)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove Twitter-specific elements (mentions and hashtags)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation - keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Remove extra whitespace and strip
        text = re.sub(r'\s+', ' ', text).strip()

        return text


def load_model_and_tokenizer(model_path: str, tokenizer_path: str) -> Tuple:
    """
    Load a trained Keras model and its associated tokenizer.

    Args:
        model_path: Path to the .h5 model file
        tokenizer_path: Path to the .pkl tokenizer file

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        FileNotFoundError: If model or tokenizer files don't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")

    print(f"Loading tokenizer from: {tokenizer_path}")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✓ Tokenizer loaded successfully")
    print(f"  Vocabulary size: {len(tokenizer.word_index)}")

    return model, tokenizer


def predict_with_model(texts: List[str], model, tokenizer,
                       preprocessor: TextPreprocessor,
                       max_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions using a trained model.

    Args:
        texts: List of raw text strings
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        preprocessor: TextPreprocessor instance
        max_len: Maximum sequence length (default: 60)

    Returns:
        Tuple of (predicted_labels, prediction_probabilities)
    """
    # Preprocess all texts
    cleaned_texts = [preprocessor.clean_text(text) for text in texts]

    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(cleaned_texts)

    # Pad sequences to fixed length
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Generate predictions
    predictions = model.predict(X, batch_size=32, verbose=0)
    predicted_labels = predictions.argmax(axis=1)

    return predicted_labels, predictions


def evaluate_predictions(true_labels: np.ndarray, predicted_labels: np.ndarray,
                        emotion_map: Dict[int, str]) -> Dict:
    """
    Calculate evaluation metrics for predictions.

    Args:
        true_labels: Array of ground truth labels
        predicted_labels: Array of predicted labels
        emotion_map: Dictionary mapping label indices to emotion names

    Returns:
        Dictionary containing evaluation metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate macro F1 score
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Generate classification report
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=list(emotion_map.values()),
        digits=4,
        output_dict=True
    )

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'classification_report': report,
        'confusion_matrix': cm
    }


def process_test_csv(test_file: str, gru_model_path: str, lstm_model_path: str,
                     tokenizer_path: str, output_dir: str = '.',
                     max_len: int = 60):
    """
    Main function to process test CSV and generate predictions.

    Loads both GRU and LSTM models, processes the test CSV, generates predictions,
    and saves results to output files.

    Args:
        test_file: Path to test CSV file
        gru_model_path: Path to best GRU model
        lstm_model_path: Path to best LSTM model
        tokenizer_path: Path to tokenizer pickle file
        output_dir: Directory to save output files
        max_len: Maximum sequence length
    """
    # Emotion label mapping (consistent with training)
    emotion_map = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }

    print("="*80)
    print("EMOTION DETECTION - TEST SET PREDICTION")
    print("="*80)
    print("\nThis script uses the best models from our ablation study:")
    print("  • GRU (LR=0.01): 92.04% validation accuracy, 0.902 macro F1")
    print("  • LSTM (Units=128): 92.59% validation accuracy, 0.906 macro F1")
    print("="*80)

    # Load test data
    print(f"\nLoading test data from: {test_file}")
    try:
        df = pd.read_csv(test_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Detect text column
    text_column = None
    for col in ['text', 'Text', 'sentence', 'Sentence', 'content', 'Content']:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        text_column = df.columns[0]
        print(f"⚠ No standard text column found. Using '{text_column}'")

    # Detect label column (if present)
    label_column = None
    has_labels = False
    for col in ['label', 'Label', 'emotion', 'Emotion']:
        if col in df.columns:
            label_column = col
            has_labels = True
            break

    print(f"✓ Loaded {len(df)} test samples")
    print(f"  Text column: '{text_column}'")
    if has_labels:
        print(f"  Label column: '{label_column}' (evaluation enabled)")

    # Initialize preprocessor (same as training)
    preprocessor = TextPreprocessor()

    # Load models and tokenizer
    print(f"\n{'='*80}")
    print("LOADING MODELS")
    print(f"{'='*80}")

    print("\n[1/3] Loading GRU model (best model)...")
    gru_model, tokenizer = load_model_and_tokenizer(gru_model_path, tokenizer_path)

    print("\n[2/3] Loading LSTM model (baseline)...")
    lstm_model, _ = load_model_and_tokenizer(lstm_model_path, tokenizer_path)

    print("\n[3/3] Models loaded successfully!")

    # Generate predictions with both models
    print(f"\n{'='*80}")
    print("GENERATING PREDICTIONS")
    print(f"{'='*80}")

    texts = df[text_column].tolist()

    print("\nProcessing with GRU model...")
    gru_labels, gru_probs = predict_with_model(
        texts, gru_model, tokenizer, preprocessor, max_len
    )
    print("✓ GRU predictions complete")

    print("\nProcessing with LSTM model...")
    lstm_labels, lstm_probs = predict_with_model(
        texts, lstm_model, tokenizer, preprocessor, max_len
    )
    print("✓ LSTM predictions complete")

    # Create results dataframe
    results_df = pd.DataFrame({
        'text': df[text_column],
        'gru_prediction': [emotion_map[label] for label in gru_labels],
        'gru_label': gru_labels,
        'gru_confidence': [gru_probs[i][gru_labels[i]] for i in range(len(gru_labels))],
        'lstm_prediction': [emotion_map[label] for label in lstm_labels],
        'lstm_label': lstm_labels,
        'lstm_confidence': [lstm_probs[i][lstm_labels[i]] for i in range(len(lstm_labels))],
        'models_agree': gru_labels == lstm_labels
    })

    # Add probability distributions for both models
    for label, emotion in emotion_map.items():
        results_df[f'gru_prob_{emotion}'] = gru_probs[:, label]
        results_df[f'lstm_prob_{emotion}'] = lstm_probs[:, label]

    # If labels present, evaluate performance
    if has_labels:
        # Map text labels to numeric
        if df[label_column].dtype == 'object':
            label_to_num = {v: k for k, v in emotion_map.items()}
            df['numeric_label'] = df[label_column].str.lower().map(label_to_num)
        else:
            df['numeric_label'] = df[label_column]

        if not df['numeric_label'].isna().any():
            true_labels = df['numeric_label'].values
            results_df['true_label'] = true_labels
            results_df['true_emotion'] = [emotion_map[label] for label in true_labels]
            results_df['gru_correct'] = gru_labels == true_labels
            results_df['lstm_correct'] = lstm_labels == true_labels

            # Evaluate both models
            print(f"\n{'='*80}")
            print("EVALUATION RESULTS")
            print(f"{'='*80}")

            print("\n" + "─"*80)
            print("GRU MODEL (Best Model - LR=0.01)")
            print("─"*80)
            gru_metrics = evaluate_predictions(true_labels, gru_labels, emotion_map)
            print(f"Accuracy: {gru_metrics['accuracy']:.4f} ({gru_metrics['accuracy']*100:.2f}%)")
            print(f"Macro F1: {gru_metrics['macro_f1']:.4f}")
            print("\nPer-class F1 scores:")
            for emotion in emotion_map.values():
                f1 = gru_metrics['classification_report'][emotion]['f1-score']
                print(f"  {emotion.capitalize():<10}: {f1:.4f}")

            print("\n" + "─"*80)
            print("LSTM MODEL (Baseline)")
            print("─"*80)
            lstm_metrics = evaluate_predictions(true_labels, lstm_labels, emotion_map)
            print(f"Accuracy: {lstm_metrics['accuracy']:.4f} ({lstm_metrics['accuracy']*100:.2f}%)")
            print(f"Macro F1: {lstm_metrics['macro_f1']:.4f}")
            print("\nPer-class F1 scores:")
            for emotion in emotion_map.values():
                f1 = lstm_metrics['classification_report'][emotion]['f1-score']
                print(f"  {emotion.capitalize():<10}: {f1:.4f}")

            # Model agreement analysis
            agreement_rate = (gru_labels == lstm_labels).mean()
            print(f"\n{'='*80}")
            print(f"Model Agreement: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
            print(f"{'='*80}")

            # Cases where both are wrong
            both_wrong = ((gru_labels != true_labels) & (lstm_labels != true_labels)).sum()
            print(f"Both models wrong: {both_wrong}/{len(true_labels)} samples ({both_wrong/len(true_labels)*100:.1f}%)")

            # Cases where GRU right, LSTM wrong
            gru_right_lstm_wrong = ((gru_labels == true_labels) & (lstm_labels != true_labels)).sum()
            print(f"GRU correct, LSTM wrong: {gru_right_lstm_wrong} samples")

            # Cases where LSTM right, GRU wrong
            lstm_right_gru_wrong = ((lstm_labels == true_labels) & (gru_labels != true_labels)).sum()
            print(f"LSTM correct, GRU wrong: {lstm_right_gru_wrong} samples")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions.csv')

    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    print(f"Saving predictions to: {output_file}")
    results_df.to_csv(output_file, index=False)
    print("✓ Predictions saved successfully")

    # Save summary statistics
    summary_file = os.path.join(output_dir, 'prediction_summary.json')
    summary = {
        'total_samples': len(results_df),
        'gru_predictions': {
            emotion: int((gru_labels == label).sum())
            for label, emotion in emotion_map.items()
        },
        'lstm_predictions': {
            emotion: int((lstm_labels == label).sum())
            for label, emotion in emotion_map.items()
        },
        'model_agreement_rate': float(agreement_rate) if has_labels else float((gru_labels == lstm_labels).mean())
    }

    if has_labels:
        summary['gru_accuracy'] = float(gru_metrics['accuracy'])
        summary['gru_macro_f1'] = float(gru_metrics['macro_f1'])
        summary['lstm_accuracy'] = float(lstm_metrics['accuracy'])
        summary['lstm_macro_f1'] = float(lstm_metrics['macro_f1'])

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saving summary to: {summary_file}")
    print("✓ Summary saved successfully")

    # Display sample predictions
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS (First 5)")
    print(f"{'='*80}")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n{'─'*80}")
        print(f"Sample {i+1}:")
        print(f"  Text: {row['text'][:100]}...")
        print(f"  GRU:  {row['gru_prediction']} (confidence: {row['gru_confidence']:.2%})")
        print(f"  LSTM: {row['lstm_prediction']} (confidence: {row['lstm_confidence']:.2%})")
        print(f"  Models agree: {'Yes ✓' if row['models_agree'] else 'No ✗'}")
        if 'true_emotion' in row:
            print(f"  True: {row['true_emotion']}")
            print(f"  GRU:  {'✓ Correct' if row['gru_correct'] else '✗ Wrong'}")
            print(f"  LSTM: {'✓ Correct' if row['lstm_correct'] else '✗ Wrong'}")

    print(f"\n{'='*80}")
    print("PREDICTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples processed: {len(results_df)}")
    print(f"Output files:")
    print(f"  • {output_file}")
    print(f"  • {summary_file}")
    print(f"{'='*80}\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Predict emotions using best GRU and LSTM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_predictions.py --test_file test.csv
  python test_predictions.py --test_file test.csv --output_dir results/
  python test_predictions.py --test_file test.csv --max_len 60
        """
    )

    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test CSV file (must contain text column)')
    parser.add_argument('--gru_model', type=str,
                       default='checkpoints/best_gru_model.h5',
                       help='Path to best GRU model (default: checkpoints/best_gru_model.h5)')
    parser.add_argument('--lstm_model', type=str,
                       default='checkpoints/best_lstm_model.h5',
                       help='Path to best LSTM model (default: checkpoints/best_lstm_model.h5)')
    parser.add_argument('--tokenizer', type=str,
                       default='tokenizer.pkl',
                       help='Path to tokenizer file (default: tokenizer.pkl)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--max_len', type=int, default=60,
                       help='Maximum sequence length (default: 60)')

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        sys.exit(1)

    # Run prediction
    try:
        process_test_csv(
            test_file=args.test_file,
            gru_model_path=args.gru_model,
            lstm_model_path=args.lstm_model,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            max_len=args.max_len
        )
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

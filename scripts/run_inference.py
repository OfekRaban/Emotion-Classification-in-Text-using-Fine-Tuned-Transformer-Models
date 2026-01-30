"""
Inference script for the best-performing DeBERTa emotion classification model.

This script:
1. Loads the trained DeBERTa-base model checkpoint
2. Applies the same preprocessing pipeline used during training
3. Runs inference on raw input data (single text, list of texts, or CSV file)
4. Outputs predictions with confidence scores

Usage:
    # Single text prediction
    python scripts/run_inference.py --text "I am feeling so happy today!"

    # Batch prediction from CSV
    python scripts/run_inference.py --input_csv data/processed/validation.csv --output_csv outputs/predictions.csv

    # Interactive mode
    python scripts/run_inference.py --interactive
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# Configuration (matching training setup)
MODEL_NAME = "microsoft/deberta-base"
MODEL_CHECKPOINT = "models/deberta_best.pt"
NUM_LABELS = 6
MAX_LENGTH = 128

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_MAP = {i: name for i, name in enumerate(LABEL_NAMES)}


# Preprocessing Pipeline
def preprocess_text(text: str) -> str:
    """
    Apply the same minimal preprocessing used during training.

    Matches the preprocessing in scripts/preprocess.py:
    - Normalize whitespace (collapse multiple spaces, remove tabs/newlines)
    - No stemming, lemmatization, stop word removal, or casing changes

    This preserves stylistic cues important for emotion classification.
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize whitespace: replace tabs/newlines with spaces, collapse multiple spaces
    text = " ".join(text.split())

    return text



# Model Loading
def load_model(checkpoint_path: str, device: torch.device):
    """
    Load the trained DeBERTa model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: torch device (cuda or cpu)

    Returns:
        model: Loaded model in eval mode
        tokenizer: Corresponding tokenizer
    """
    print(f"Loading model from: {checkpoint_path}")

    # Initialize model architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    # Load trained weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Model loaded successfully on {device}")
    return model, tokenizer



# Inference Functions
def predict_single(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    return_probs: bool = True
) -> dict:
    """
    Predict emotion for a single text.

    Args:
        text: Raw input text
        model: Trained model
        tokenizer: Tokenizer
        device: torch device
        return_probs: Whether to return probability distribution

    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    text = preprocess_text(text)

    # Tokenize
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_label].item()

    result = {
        "text": text,
        "predicted_label": pred_label,
        "predicted_emotion": LABEL_MAP[pred_label],
        "confidence": confidence,
    }

    if return_probs:
        result["probabilities"] = {
            LABEL_MAP[i]: probs[0, i].item() for i in range(NUM_LABELS)
        }

    return result


def predict_batch(
    texts: list,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int = 32
) -> list:
    """
    Predict emotions for a batch of texts.

    Args:
        texts: List of raw input texts
        model: Trained model
        tokenizer: Tokenizer
        device: torch device
        batch_size: Batch size for inference

    Returns:
        List of prediction dictionaries
    """
    # Preprocess all texts
    texts = [preprocess_text(t) for t in texts]

    all_preds = []
    all_probs = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        encoding = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    # Format results
    results = []
    for text, pred, probs in zip(texts, all_preds, all_probs):
        results.append({
            "text": text,
            "predicted_label": pred,
            "predicted_emotion": LABEL_MAP[pred],
            "confidence": probs[pred],
            "probabilities": {LABEL_MAP[i]: probs[i] for i in range(NUM_LABELS)}
        })

    return results


def predict_from_csv(
    input_path: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    text_column: str = "text",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Run inference on a CSV file.

    Args:
        input_path: Path to input CSV
        model: Trained model
        tokenizer: Tokenizer
        device: torch device
        text_column: Name of the text column
        batch_size: Batch size for inference

    Returns:
        DataFrame with predictions
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")

    texts = df[text_column].astype(str).tolist()
    print(f"Running inference on {len(texts)} samples...")

    results = predict_batch(texts, model, tokenizer, device, batch_size)

    # Add predictions to dataframe
    df["predicted_label"] = [r["predicted_label"] for r in results]
    df["predicted_emotion"] = [r["predicted_emotion"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]

    # Add per-class probabilities
    for label_name in LABEL_NAMES:
        df[f"prob_{label_name}"] = [r["probabilities"][label_name] for r in results]

    return df



# Evaluation (when ground truth is available)
def evaluate_predictions(df: pd.DataFrame, label_column: str = "label"):
    """
    Evaluate predictions against ground truth labels.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    if label_column not in df.columns:
        print("No ground truth labels found, skipping evaluation.")
        return None

    y_true = df[label_column].astype(int).tolist()
    y_pred = df["predicted_label"].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print(" Evaluation Results")
    print("=" * 60)
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Macro-F1:     {macro_f1:.4f}")
    print(f"  Weighted-F1:  {weighted_f1:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


# Interactive Mode
def interactive_mode(model, tokenizer, device):
    """
    Run interactive prediction mode.
    """
    print("\n" + "=" * 60)
    print(" Interactive Emotion Classification")
    print(" Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if text.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break

        if not text:
            continue

        result = predict_single(text, model, tokenizer, device)

        print(f"\n  Predicted emotion: {result['predicted_emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  All probabilities:")
        for emotion, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"    {emotion:<10} {prob:>6.2%} {bar}")
        print()



def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the trained DeBERTa emotion classifier"
    )
    parser.add_argument(
        "--text", type=str, help="Single text to classify"
    )
    parser.add_argument(
        "--input_csv", type=str, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_csv", type=str, help="Path to save predictions CSV"
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Name of text column in CSV"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu). Auto-detected if not specified."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=MODEL_CHECKPOINT, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate against ground truth labels"
    )

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, tokenizer = load_model(args.checkpoint, device)

    # Run inference based on mode
    if args.interactive:
        interactive_mode(model, tokenizer, device)

    elif args.text:
        result = predict_single(args.text, model, tokenizer, device)
        print(f"\nText: {result['text']}")
        print(f"Predicted emotion: {result['predicted_emotion'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nProbabilities:")
        for emotion, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {emotion:<10} {prob:.4f}")

    elif args.input_csv:
        df = predict_from_csv(
            args.input_csv, model, tokenizer, device,
            text_column=args.text_column,
            batch_size=args.batch_size
        )

        if args.evaluate:
            evaluate_predictions(df)

        if args.output_csv:
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False)
            print(f"\nPredictions saved to: {args.output_csv}")
        else:
            print("\nSample predictions:")
            print(df[["text", "predicted_emotion", "confidence"]].head(10).to_string())

    else:
        parser.print_help()
        print("\nExample usage:")
        print('  python scripts/run_inference.py --text "I am so happy today!"')
        print('  python scripts/run_inference.py --input_csv data/processed/validation.csv --evaluate')
        print('  python scripts/run_inference.py --interactive')


if __name__ == "__main__":
    main()

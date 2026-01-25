import argparse
import pandas as pd
import yaml
from collections import Counter
from pathlib import Path


def compute_class_statistics(train_csv: str, output_path: str):
    """
    Compute class distribution and inverse-frequency class weights
    from the training dataset only, and save them as a YAML config file.
    """

    df = pd.read_csv(train_csv)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the training CSV")

    labels = df["label"].astype(int).tolist()
    num_samples = len(labels)

    class_counts = Counter(labels)
    num_classes = len(class_counts)

    # Inverse frequency weights
    class_weights = {
        cls: num_samples / count
        for cls, count in class_counts.items()
    }

    config = {
        "num_classes": num_classes,
        "num_samples": num_samples,
        "class_distribution": {
            int(cls): int(count) for cls, count in class_counts.items()
        },
        "class_weights": {
            int(cls): round(weight, 4) for cls, weight in class_weights.items()
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=True)

    print("✅ Class statistics computed successfully")
    print(f"Train samples: {num_samples}")
    print("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count}")
    print(f"\nConfig saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute class distribution and weights from training data"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to training CSV file (must contain 'label' column)"
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="configs/data.yaml",
        help="Path to output YAML config file (default: configs/data.yaml)"
    )

    args = parser.parse_args()

    compute_class_statistics(
        train_csv=args.train_csv,
        output_path=args.output_config
    )

import itertools
import csv
import time
from pathlib import Path
import torch

from src.data.dataset import EmotionDataset
from src.data.dataloader import create_dataloaders
from src.models.transformer_classifier import load_model
from src.training.train import train


# Grid configuration
GRID = {
    "learning_rate": [2e-5, 3e-5],
    "batch_size": [16, 32],
    "weight_decay": [0.0, 0.01],
    "freeze_encoder": [True, False],
    "epochs": [3, 4]
}

MODEL_NAME = "microsoft/deberta-base"
NUM_LABELS = 6
MAX_LENGTH = 128

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"

RESULTS_DIR = Path("outputs/grid_results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Main grid runner
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_path = RESULTS_DIR / "deberta_grid.csv"

    best_macro_f1 = -1.0
    best_config = None

    # Prepare CSV
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "learning_rate",
            "batch_size",
            "weight_decay",
            "freeze_encoder",
            "epochs",
            "val_accuracy",
            "val_macro_f1",
            "val_weighted_f1",
            "train_time_sec",
            "peak_gpu_mb",
            "infer_ms_per_sample"
        ])

    # Cartesian product of grid
    for lr, batch_size, weight_decay, freeze_encoder, epochs in itertools.product(
        GRID["learning_rate"],
        GRID["batch_size"],
        GRID["weight_decay"],
        GRID["freeze_encoder"],
        GRID["epochs"]
    ):

        print("\n" + "=" * 20)
        print(f"Running config: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, freeze_encoder={freeze_encoder}, epochs={epochs}")
        print("=" * 20)

        # Reset GPU stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Dataset & DataLoader
        train_dataset = EmotionDataset(TRAIN_CSV, MODEL_NAME, MAX_LENGTH)
        val_dataset = EmotionDataset(VAL_CSV, MODEL_NAME, MAX_LENGTH)

        train_loader, val_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size
        )

        # Model
        model = load_model(MODEL_NAME, NUM_LABELS)

        # -------------------------
        # Train + measure time
        # -------------------------
        (
            model,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1,
            infer_ms_per_sample,
            train_time_sec,
            peak_gpu_mb
        ) = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            freeze_encoder=freeze_encoder
        )

        # Save results
        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                lr,
                batch_size,
                weight_decay,
                freeze_encoder,
                epochs,
                round(val_accuracy, 4),
                round(val_macro_f1, 4),
                round(val_weighted_f1, 4),
                round(train_time_sec, 2),
                round(peak_gpu_mb, 1) if peak_gpu_mb else None,
                round(infer_ms_per_sample, 2)
            ])

        # Save best model
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_config = (lr, batch_size, weight_decay, freeze_encoder, epochs)

            model_path = MODELS_DIR / "deberta_best.pt"
            torch.save(model.state_dict(), model_path)

            print(f"\n New best model saved! Macro-F1 = {best_macro_f1:.4f}")

    print("\n" + "=" * 20)
    print("Grid search completed.")
    print(f"Best config: lr={best_config[0]}, batch_size={best_config[1]}, weight_decay={best_config[2]}, freeze_encoder={best_config[3]}, epochs={best_config[4]}")
    print(f"Best Macro-F1: {best_macro_f1:.4f}")
    print("=" * 20)


if __name__ == "__main__":
    main()

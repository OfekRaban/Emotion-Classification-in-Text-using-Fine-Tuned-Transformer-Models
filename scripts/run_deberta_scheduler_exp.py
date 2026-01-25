import time
import csv
from pathlib import Path
import torch

from transformers import get_linear_schedule_with_warmup

from src.data.dataset import EmotionDataset
from src.data.dataloader import create_dataloaders
from src.models.transformer_classifier import load_model
from src.training.train import train


# =========================
# Fixed BEST configuration
# =========================
MODEL_NAME = "microsoft/deberta-base"
NUM_LABELS = 6
MAX_LENGTH = 128

LR = 2e-5
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
FREEZE_ENCODER = False
EPOCHS = 3

WARMUP_RATIO = 0.1  # 10% warmup (best practice)

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"

RESULTS_DIR = Path("outputs/scheduler_exp")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_path = RESULTS_DIR / "deberta_scheduler_results.csv"

    # -------------------------
    # Prepare CSV
    # -------------------------
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "learning_rate",
            "batch_size",
            "weight_decay",
            "epochs",
            "warmup_ratio",
            "val_accuracy",
            "val_macro_f1",
            "val_weighted_f1",
            "train_time_sec",
            "peak_gpu_mb",
            "infer_ms_per_sample"
        ])

    print("\n" + "=" * 30)
    print("Running DeBERTa + LR Scheduler Experiment")
    print("=" * 30)

    # Reset GPU stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    train_dataset = EmotionDataset(TRAIN_CSV, MODEL_NAME, MAX_LENGTH)
    val_dataset = EmotionDataset(VAL_CSV, MODEL_NAME, MAX_LENGTH)

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE
    )

    # -------------------------
    # Model
    # -------------------------
    model = load_model(MODEL_NAME, NUM_LABELS)
    model.to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # -------------------------
    # Scheduler (WARMUP + DECAY)
    # -------------------------
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # -------------------------
    # Train
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
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        freeze_encoder=FREEZE_ENCODER,
        scheduler=scheduler,
        optimizer=optimizer
    )

    # -------------------------
    # Save results
    # -------------------------
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            LR,
            BATCH_SIZE,
            WEIGHT_DECAY,
            EPOCHS,
            WARMUP_RATIO,
            round(val_accuracy, 4),
            round(val_macro_f1, 4),
            round(val_weighted_f1, 4),
            round(train_time_sec, 2),
            round(peak_gpu_mb, 1) if peak_gpu_mb else None,
            round(infer_ms_per_sample, 2)
        ])

    # -------------------------
    # Save model
    # -------------------------
    model_path = MODELS_DIR / "deberta_scheduler_best.pt"
    torch.save(model.state_dict(), model_path)

    print("\n Scheduler experiment completed")
    print(f"Macro-F1: {val_macro_f1:.4f}")
    print(f"Model saved to: {model_path}")
    print("=" * 30)


if __name__ == "__main__":
    main()

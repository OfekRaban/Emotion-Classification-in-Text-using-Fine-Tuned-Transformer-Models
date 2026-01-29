import torch
from torch.optim import AdamW

from src.data.dataset import EmotionDataset
from src.data.dataloader import create_dataloaders
from src.models.transformer_classifier import load_model
from src.compression.distillation import DistillationTrainer
from src.utils.config_loader import load_class_weights
from src.utils.seed import set_seed
from src.utils.metrics import (
    count_parameters,
    get_model_size_mb,
    measure_inference_latency,
    get_peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    compute_comprehensive_metrics,
    print_comprehensive_metrics,
)


# =========================
# Configuration
# =========================
MODEL_TEACHER = "microsoft/deberta-base"
MODEL_STUDENT = "microsoft/deberta-v3-xsmall"

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"

TEACHER_CHECKPOINT = "models/deberta_best.pt"
OUTPUT_STUDENT = "models/deberta_student_distilled.pt"

NUM_LABELS = 6
MAX_LENGTH = 128

BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.0

ALPHA = 0.3
TEMPERATURE = 2.0


def main():
    # =========================
    # Setup
    # =========================
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Dataset & DataLoader
    # Use teacher's tokenizer (shared tokenizer approach)
    # =========================
    train_dataset = EmotionDataset(
        TRAIN_CSV,
        MODEL_TEACHER,
        MAX_LENGTH
    )
    val_dataset = EmotionDataset(
        VAL_CSV,
        MODEL_TEACHER,
        MAX_LENGTH
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE
    )

    # =========================
    # Teacher (frozen)
    # =========================
    teacher = load_model(MODEL_TEACHER, NUM_LABELS)
    teacher.load_state_dict(
        torch.load(TEACHER_CHECKPOINT, map_location=device)
    )
    teacher.to(device)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    # =========================
    # Student (trainable)
    # =========================
    student = load_model(MODEL_STUDENT, NUM_LABELS)

    # ============ DIAGNOSTIC: Verify Student Architecture ============
    print("\n" + "=" * 60)
    print(" STUDENT MODEL VERIFICATION")
    print("=" * 60)
    print(f"  MODEL_STUDENT variable: {MODEL_STUDENT}")
    print(f"  Model class: {student.__class__.__name__}")
    print(f"  Config model_type: {student.config.model_type}")
    print(f"  hidden_size: {student.config.hidden_size}")
    print(f"  num_hidden_layers: {student.config.num_hidden_layers}")
    print(f"  num_attention_heads: {student.config.num_attention_heads}")
    print(f"  vocab_size: {student.config.vocab_size}")
    student_params_fresh = sum(p.numel() for p in student.parameters())
    print(f"  Parameters (fresh load): {student_params_fresh:,}")
    print("=" * 60)

    # Expected values for deberta-v3-small:
    #   hidden_size: 768, num_hidden_layers: 6, vocab_size: 128100, params: ~44M
    # If you see num_hidden_layers: 12 or vocab_size: 50265, WRONG MODEL!
    if student_params_fresh > 100_000_000:
        print("  WARNING: Parameter count > 100M suggests this is NOT v3-small!")
        print("  deberta-v3-small should have ~44M parameters")
    # =================================================================

    student.to(device)

    # =========================
    # Class weights (same as baseline)
    # =========================
    class_weights = load_class_weights(
        config_path="configs/data.yaml",
        device=device
    )

    # =========================
    # Optimizer (same style as baseline)
    # =========================
    optimizer = AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # =========================
    # Distillation Trainer
    # =========================
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        alpha=ALPHA,
        temperature=TEMPERATURE,
        class_weights=class_weights,
        max_grad_norm=1.0
    )

    # =========================
    # Training loop
    # =========================
    best_f1 = -1.0
    reset_peak_gpu_memory()  # Track training GPU memory

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        train_stats = trainer.train_epoch()
        val_metrics = trainer.evaluate()

        print(
            f"Train | loss={train_stats['loss']:.4f} "
            f"(CE={train_stats['ce']:.4f}, KL={train_stats['kl']:.4f}) "
            f"time={train_stats['time_sec']:.1f}s"
        )

        print(
            f"Val   | acc={val_metrics['accuracy']:.4f} "
            f"macro-F1={val_metrics['f1_macro']:.4f} "
            f"weighted-F1={val_metrics['f1_weighted']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(student.state_dict(), OUTPUT_STUDENT)
            print("✔ Saved new best distilled student model")

    # Capture training peak GPU memory
    training_peak_gpu = get_peak_gpu_memory_mb()

    # =========================
    # Final Comprehensive Evaluation
    # =========================
    print("\n" + "=" * 60)
    print(" Final Evaluation - Distilled Student Model")
    print("=" * 60)

    # Load best model
    saved_state_dict = torch.load(OUTPUT_STUDENT, map_location=device)

    # ============ DIAGNOSTIC: Check saved checkpoint ============
    print("\n[Checkpoint Verification]")
    print(f"  Checkpoint path: {OUTPUT_STUDENT}")
    print(f"  Checkpoint keys (first 5): {list(saved_state_dict.keys())[:5]}")
    model_keys = set(student.state_dict().keys())
    saved_keys = set(saved_state_dict.keys())
    if model_keys != saved_keys:
        print(f"  WARNING: Key mismatch!")
        print(f"    Missing in checkpoint: {model_keys - saved_keys}")
        print(f"    Extra in checkpoint: {saved_keys - model_keys}")
    else:
        print(f"  Keys match: {len(model_keys)} keys")
    # Check embedding size as proxy for vocab_size
    if "deberta.embeddings.word_embeddings.weight" in saved_state_dict:
        emb_shape = saved_state_dict["deberta.embeddings.word_embeddings.weight"].shape
        print(f"  Saved embedding shape: {emb_shape}")
        print(f"  -> vocab_size={emb_shape[0]}, hidden_size={emb_shape[1]}")
    # =============================================================

    student.load_state_dict(saved_state_dict)
    student.eval()

    # Collect predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(all_labels, all_preds)

    # Model statistics
    total_params, _ = count_parameters(student)
    model_size = get_model_size_mb(OUTPUT_STUDENT)

    # Measure inference latency on CPU for fair comparison with quantization
    cpu_device = torch.device("cpu")
    student_cpu = student.to(cpu_device)
    latency_cpu = measure_inference_latency(student_cpu, val_loader, cpu_device)

    # Also measure GPU latency if available
    latency_gpu = None
    if torch.cuda.is_available():
        student.to(device)
        latency_gpu = measure_inference_latency(student, val_loader, device)

    # Print everything
    print_comprehensive_metrics(
        metrics=metrics,
        model_name="Distilled Student (DeBERTa-v3-small)",
        num_params=total_params,
        model_size_mb=model_size,
        latency_ms=latency_cpu,  # CPU latency for fair comparison
        peak_gpu_mb=training_peak_gpu,
    )

    # Also show GPU latency if available
    if latency_gpu is not None:
        print(f"  [GPU Inference Latency: {latency_gpu:.2f} ms/sample]")

    print("\nKnowledge Distillation Complete!")
    print(f"Model saved to: {OUTPUT_STUDENT}")


if __name__ == "__main__":
    main()

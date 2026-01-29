import torch
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig

from src.data.dataset import EmotionDataset
from src.data.dataloader import create_dataloaders
from src.utils.metrics import (
    count_parameters,
    get_model_size_mb,
    measure_inference_latency,
    compute_comprehensive_metrics,
    print_comprehensive_metrics,
)


MODEL_NAME = "microsoft/deberta-base"
NUM_LABELS = 6
MAX_LENGTH = 128
BATCH_SIZE = 32

VAL_CSV = "data/processed/validation.csv"
FP32_MODEL_PATH = "models/deberta_best.pt"
QUANT_MODEL_PATH = "models/deberta_quantized.pt"


def get_model_memory_size_mb(model):
    """
    Calculate actual model size in memory (not file size).
    Handles both regular and quantized models.
    """
    param_size = 0
    buffer_size = 0

    # Regular parameters
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    # Buffers
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    # For quantized models, also count packed params
    for name, module in model.named_modules():
        if hasattr(module, '_packed_params'):
            try:
                packed = module._packed_params._packed_params
                if isinstance(packed, tuple):
                    weight = packed[0]
                    # Quantized weights are stored as int8
                    param_size += weight.numel() * 1  # 1 byte per int8
            except Exception:
                pass

    return (param_size + buffer_size) / (1024 ** 2)


def print_model_architecture(model_name):
    """Print model architecture details for verification."""
    config = AutoConfig.from_pretrained(model_name)
    print(f"\n[Architecture: {model_name}]")
    print(f"  hidden_size:       {config.hidden_size}")
    print(f"  num_layers:        {config.num_hidden_layers}")
    print(f"  attention_heads:   {config.num_attention_heads}")
    print(f"  vocab_size:        {config.vocab_size}")


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def main():
    device = torch.device("cpu")  # Quantization requires CPU

    # ---------------------------
    # Verify architecture
    # ---------------------------
    print_model_architecture(MODEL_NAME)

    # ---------------------------
    # Dataset
    # ---------------------------
    val_dataset = EmotionDataset(VAL_CSV, MODEL_NAME, MAX_LENGTH)
    _, val_loader = create_dataloaders(
        train_dataset=val_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE
    )

    # ---------------------------
    # Load FP32 model
    # ---------------------------
    print("\nLoading FP32 model...")
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    model_fp32.load_state_dict(torch.load(FP32_MODEL_PATH, map_location="cpu"))
    model_fp32.eval()

    # Count FP32 params BEFORE quantization (this is the true param count)
    fp32_params, _ = count_parameters(model_fp32)
    fp32_memory_size = get_model_memory_size_mb(model_fp32)
    fp32_file_size = get_model_size_mb(FP32_MODEL_PATH)

    print(f"  FP32 Parameters: {fp32_params:,}")
    print(f"  FP32 Memory:     {fp32_memory_size:.2f} MB")
    print(f"  FP32 File:       {fp32_file_size:.2f} MB")

    # ---------------------------
    # Quantization
    # ---------------------------
    print("\nApplying dynamic INT8 quantization...")
    model_quant = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Save quantized model
    torch.save(model_quant.state_dict(), QUANT_MODEL_PATH)

    # Measure quantized model size
    # Note: param count stays same, but memory/storage is reduced
    quant_memory_size = get_model_memory_size_mb(model_quant)
    quant_file_size = get_model_size_mb(QUANT_MODEL_PATH)

    print(f"  INT8 Parameters: {fp32_params:,} (same as FP32)")
    print(f"  INT8 Memory:     {quant_memory_size:.2f} MB")
    print(f"  INT8 File:       {quant_file_size:.2f} MB")

    # ---------------------------
    # Evaluate Both Models
    # ---------------------------
    print("\nEvaluating FP32 model...")
    fp32_preds, fp32_labels = evaluate_model(model_fp32, val_loader, device)
    fp32_metrics = compute_comprehensive_metrics(fp32_labels, fp32_preds)

    print("Evaluating quantized model...")
    quant_preds, quant_labels = evaluate_model(model_quant, val_loader, device)
    quant_metrics = compute_comprehensive_metrics(quant_labels, quant_preds)

    # ---------------------------
    # Latency Measurement (CPU only, fair comparison)
    # ---------------------------
    print("\nMeasuring inference latency (CPU)...")
    fp32_latency = measure_inference_latency(model_fp32, val_loader, device)
    quant_latency = measure_inference_latency(model_quant, val_loader, device)

    # ---------------------------
    # Print Comprehensive Results
    # ---------------------------
    print_comprehensive_metrics(
        metrics=fp32_metrics,
        model_name="FP32 Baseline (DeBERTa-base)",
        num_params=fp32_params,
        model_size_mb=fp32_file_size,  # Report file size for reproducibility
        latency_ms=fp32_latency,
        peak_gpu_mb=None,  # CPU-only, don't show
    )

    print_comprehensive_metrics(
        metrics=quant_metrics,
        model_name="Quantized INT8 (DeBERTa-base)",
        num_params=fp32_params,  # Same param count, different precision
        model_size_mb=quant_file_size,
        latency_ms=quant_latency,
        peak_gpu_mb=None,  # CPU-only
    )

    # ---------------------------
    # Compression Summary
    # ---------------------------
    print("=" * 60)
    print(" Quantization Compression Summary")
    print("=" * 60)
    print(f"  Parameters:       {fp32_params:,} (unchanged)")
    print(f"  FP32 File Size:   {fp32_file_size:.2f} MB")
    print(f"  INT8 File Size:   {quant_file_size:.2f} MB")
    print(f"  Compression:      {fp32_file_size / quant_file_size:.2f}x")
    print(f"  Size Reduction:   {(1 - quant_file_size / fp32_file_size) * 100:.1f}%")
    print(f"  FP32 Latency:     {fp32_latency:.2f} ms/sample")
    print(f"  INT8 Latency:     {quant_latency:.2f} ms/sample")
    print(f"  Speedup:          {fp32_latency / quant_latency:.2f}x")
    print(f"  Accuracy Drop:    {(fp32_metrics['accuracy'] - quant_metrics['accuracy']) * 100:.2f}%")
    print("=" * 60)
    print(f"\nQuantized model saved to: {QUANT_MODEL_PATH}")


if __name__ == "__main__":
    main()

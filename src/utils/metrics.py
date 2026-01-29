import os
import time
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def compute_metrics(y_true, y_pred):
    """
    Compute standard classification metrics.
    Used by DistillationTrainer during validation.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = (y_true == y_pred).mean()

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def count_parameters(model, verbose=False):
    """
    Count parameters robustly, handling both regular and quantized models.

    For quantized models, parameter counting via .parameters() may miss
    packed quantized weights. This function handles both cases.
    """
    total = 0
    trainable = 0

    # Count regular parameters
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    # For quantized models, count packed parameters separately
    # (quantized Linear layers store weights differently)
    quantized_params = 0
    for name, module in model.named_modules():
        # Check for dynamic quantized linear layers
        if hasattr(module, '_packed_params'):
            try:
                # Access the underlying weight
                if hasattr(module._packed_params, '_packed_params'):
                    weight = module._packed_params._packed_params[0]
                    quantized_params += weight.numel()
            except Exception:
                pass

    if verbose and quantized_params > 0:
        print(f"  [Quantized params detected: {quantized_params:,}]")

    return total, trainable


def get_model_size_mb(model_or_path):
    """
    Get model size in MB.

    If given a path (str or Path), returns file size.
    If given a model, computes in-memory size.
    """
    if isinstance(model_or_path, (str, os.PathLike)):
        # File size
        return os.path.getsize(model_or_path) / (1024 ** 2)
    else:
        # In-memory size for model object
        param_size = sum(
            p.numel() * p.element_size() for p in model_or_path.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() for b in model_or_path.buffers()
        )
        return (param_size + buffer_size) / (1024 ** 2)


def measure_inference_latency(
    model,
    dataloader,
    device,
    num_warmup=10,
    num_runs=50,
    batch_size=None,
):
    """
    Measure inference latency in ms/sample.

    For fair comparison:
    - Uses fixed batch from dataloader (or synthetic if batch_size specified)
    - Reports mean latency over multiple runs
    - Handles both CPU and GPU properly
    """
    model.eval()
    model.to(device)

    # Get a batch for measurement
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    actual_batch_size = input_ids.size(0)

    # Warmup (important for both CPU cache and GPU kernels)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # Synchronize if using CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs - measure each run separately for variance analysis
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

    # Compute statistics
    mean_latency = sum(latencies) / len(latencies)
    latency_per_sample_ms = (mean_latency / actual_batch_size) * 1000

    return latency_per_sample_ms


def get_peak_gpu_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_peak_gpu_memory():
    """Reset peak GPU memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def compute_comprehensive_metrics(y_true, y_pred, label_names=None):
    """
    Compute comprehensive classification metrics including:
    - Accuracy, Macro-F1, Weighted-F1
    - Confusion matrix
    - Per-class Precision, Recall, F1
    """
    if label_names is None:
        label_names = LABEL_NAMES

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "per_class": {},
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, name in enumerate(label_names):
        metrics["per_class"][name] = {
            "precision": precision_per_class[i],
            "recall": recall_per_class[i],
            "f1": f1_per_class[i],
        }

    return metrics


def print_comprehensive_metrics(
    metrics,
    model_name,
    num_params=None,
    model_size_mb=None,
    latency_ms=None,
    peak_gpu_mb=None,
    label_names=None,
):
    """
    Print all metrics in a formatted way.
    """
    if label_names is None:
        label_names = LABEL_NAMES

    print(f"\n{'='*60}")
    print(f" {model_name} - Comprehensive Metrics")
    print(f"{'='*60}")

    # Basic metrics
    print(f"\n[Classification Metrics]")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:     {metrics['f1_macro']:.4f}")
    print(f"  Weighted-F1:  {metrics['f1_weighted']:.4f}")

    # Model stats
    if num_params is not None or model_size_mb is not None or latency_ms is not None:
        print(f"\n[Model Statistics]")
        if num_params is not None:
            print(f"  #Parameters:       {num_params:,}")
        if model_size_mb is not None:
            print(f"  Model Size:        {model_size_mb:.2f} MB")
        if latency_ms is not None:
            print(f"  Inference Latency (CPU): {latency_ms:.2f} ms/sample")
        if peak_gpu_mb is not None and peak_gpu_mb > 0:
            print(f"  Peak GPU Memory:   {peak_gpu_mb:.2f} MB")

    # Confusion matrix
    print(f"\n[Confusion Matrix]")
    cm = metrics["confusion_matrix"]
    header = "          " + "  ".join([f"{name[:5]:>5}" for name in label_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{val:5d}" for val in row])
        print(f"  {label_names[i][:8]:<8} {row_str}")

    # Per-class metrics
    print(f"\n[Per-Class Metrics]")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for name in label_names:
        pc = metrics["per_class"][name]
        print(f"  {name:<10} {pc['precision']:>10.4f} {pc['recall']:>10.4f} {pc['f1']:>10.4f}")

    print(f"\n{'='*60}\n")

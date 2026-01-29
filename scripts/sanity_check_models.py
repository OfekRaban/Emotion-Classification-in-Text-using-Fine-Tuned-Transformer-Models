"""
Sanity check script to verify model architectures and measurements.
Run this BEFORE any compression experiments.
"""
import torch
import time
from transformers import AutoModelForSequenceClassification, AutoConfig


def print_model_info(model_name: str):
    """Print detailed model architecture info."""
    print(f"\n{'='*60}")
    print(f" Model: {model_name}")
    print(f"{'='*60}")

    # Load config first
    config = AutoConfig.from_pretrained(model_name)
    print(f"\n[Config]")
    print(f"  hidden_size:        {config.hidden_size}")
    print(f"  num_hidden_layers:  {config.num_hidden_layers}")
    print(f"  num_attention_heads:{config.num_attention_heads}")
    print(f"  intermediate_size:  {config.intermediate_size}")
    print(f"  vocab_size:         {config.vocab_size}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=6
    )

    # Count parameters properly
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[Parameters]")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Calculate model size in memory (not file size!)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    print(f"\n[Memory Size]")
    print(f"  Parameters: {param_size / (1024**2):.2f} MB")
    print(f"  Buffers:    {buffer_size / (1024**2):.2f} MB")
    print(f"  Total:      {total_size_mb:.2f} MB")

    return model, config


def measure_latency_fair(model, batch_size=32, seq_len=128, num_warmup=10, num_runs=50):
    """
    Fair CPU-only latency measurement.
    Uses synthetic data to avoid dataloader overhead.
    """
    model.eval()
    model.to("cpu")

    # Create synthetic batch
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

    avg_latency = sum(latencies) / len(latencies)
    latency_per_sample_ms = (avg_latency / batch_size) * 1000

    return latency_per_sample_ms, latencies


def count_parameters_robust(model):
    """
    Count parameters robustly, handling quantized models.
    """
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()

    # For quantized models, also count packed parameters
    for name, module in model.named_modules():
        if hasattr(module, '_packed_params'):
            # Quantized linear layer
            weight, bias = module._packed_params._packed_params
            total += weight.numel()
            if bias is not None:
                total += bias.numel()

    return total


def main():
    print("\n" + "=" * 70)
    print(" SANITY CHECK: Model Architectures")
    print("=" * 70)

    models_to_check = [
        "microsoft/deberta-base",
        "microsoft/deberta-v3-small",
    ]

    results = {}

    for model_name in models_to_check:
        model, config = print_model_info(model_name)

        print(f"\n[CPU Latency Measurement]")
        latency_ms, _ = measure_latency_fair(model, batch_size=32, seq_len=128)
        print(f"  Latency: {latency_ms:.2f} ms/sample (batch=32, seq=128)")

        results[model_name] = {
            "params": sum(p.numel() for p in model.parameters()),
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
            "latency_ms": latency_ms,
        }

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print comparison
    print("\n" + "=" * 70)
    print(" COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Params':>12} {'Hidden':>8} {'Layers':>8} {'Vocab':>10} {'Latency':>10}")
    print("-" * 80)
    for name, r in results.items():
        short_name = name.split("/")[-1]
        print(f"{short_name:<30} {r['params']:>12,} {r['hidden_size']:>8} {r['layers']:>8} {r['vocab_size']:>10} {r['latency_ms']:>8.2f}ms")

    # Expected values check
    print("\n" + "=" * 70)
    print(" EXPECTED VALUES (from HuggingFace)")
    print("=" * 70)
    print("""
    deberta-base:     ~139M params, hidden=768,  layers=12, vocab=50265
    deberta-v3-small: ~44M params,  hidden=768,  layers=6,  vocab=128100

    If your measurements differ significantly, there's a bug!
    """)


if __name__ == "__main__":
    main()

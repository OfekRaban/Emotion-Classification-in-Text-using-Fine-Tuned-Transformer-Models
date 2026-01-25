import time
import torch
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from src.utils.config_loader import load_class_weights



# Train for a single epoch (weighted loss)

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scheduler=None):
    model.train()  # Set model to training mode - dropout, batchnorm, etc.
    total_loss = 0.0
    # Iterate over batches - one full epoch
    for batch in dataloader: 
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass - logits , loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



# Evaluation + inference latency (no class weights)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    # Validation loss should reflect real performance (no weighting)
    loss_fn = torch.nn.CrossEntropyLoss()

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    end_time = time.perf_counter()

    avg_loss = total_loss / len(dataloader)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    # Inference latency (ms per sample)
    infer_ms_per_sample = (end_time - start_time) * 1000 / len(all_labels)

    return avg_loss, accuracy, macro_f1, weighted_f1, infer_ms_per_sample



# Full training loop (returns BEST epoch metrics)

def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.0,
    freeze_encoder: bool = False,
    scheduler=None,
    optimizer=None
):
    model.to(device)
    # for freeze bert , we stop gradients to bert parameters
    if freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Use provided optimizer or create default (backward compatible with grid search)
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load class weights ONCE (do not recompute per run)
    class_weights = load_class_weights(
        config_path="configs/data.yaml",
        device=device
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) # Weighted loss because of class imbalance

    best_macro_f1 = -1.0
    best_state_dict = None
    best_metrics = None

    total_train_time = 0.0
    peak_gpu_mb = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reset GPU peak stats per epoch (best practice)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start_train = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler
        )

        epoch_train_time = time.perf_counter() - start_train
        total_train_time += epoch_train_time

        if device.type == "cuda":
            epoch_peak_gpu = torch.cuda.max_memory_allocated() / 1024**2
            peak_gpu_mb = epoch_peak_gpu if peak_gpu_mb is None else max(peak_gpu_mb, epoch_peak_gpu)

        # ---- Validation ----
        val_loss, val_accuracy, macro_f1, weighted_f1, infer_ms = evaluate(
            model=model,
            dataloader=val_loader,
            device=device
        )

        print(f"Train loss:        {train_loss:.4f}")
        print(f"Val loss:          {val_loss:.4f}")
        print(f"Val accuracy:      {val_accuracy:.4f}")
        print(f"Val Macro-F1:      {macro_f1:.4f}")
        print(f"Val Weighted-F1:   {weighted_f1:.4f}")
        print(f"Train time:        {epoch_train_time:.2f} sec / epoch")

        if peak_gpu_mb is not None:
            print(f"Peak GPU memory:   {epoch_peak_gpu:.1f} MB")

        print(f"Inference latency: {infer_ms:.2f} ms / sample")

        # Track BEST epoch by Macro-F1 (critical for imbalanced data)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = (
                val_accuracy,
                macro_f1,
                weighted_f1,
                infer_ms
            )

    # Restore best model weights before returning
    model.load_state_dict(best_state_dict)
    model.to(device)

    val_accuracy, macro_f1, weighted_f1, infer_ms = best_metrics

    return (
        model,
        val_accuracy,
        macro_f1,
        weighted_f1,
        infer_ms,
        total_train_time,
        peak_gpu_mb
    )

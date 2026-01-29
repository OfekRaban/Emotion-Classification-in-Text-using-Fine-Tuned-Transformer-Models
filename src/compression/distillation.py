import time
from typing import Dict, Any, Optional
import torch
from src.compression.losses import DistillationLoss
from src.utils.metrics import compute_metrics
from torch.nn.utils import clip_grad_norm_ # gradient clipping for stability

class DistillationTrainer:
    """
    Trainer for Knowledge Distillation:
    - Teacher is frozen (eval + no_grad)
    - Student is trained using:
        alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(student || teacher)
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        alpha: float = 0.3,
        temperature: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        max_grad_norm: float = 1.0,
    ):
        # Move models to device
        self.teacher = teacher.to(device) 
        self.student = student.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm

        # Distillation loss (CE + KL)
        self.criterion = DistillationLoss(
            alpha=alpha,
            temperature=temperature,
            class_weights=class_weights,
        )

        # Freeze teacher: no training, no grads, stable outputs
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Expected batch keys (common in HF-style datasets):
        - input_ids
        - attention_mask
        - labels  (or sometimes 'label')
        """
        # Support both 'labels' and 'label' - alignment for different datasets
        if "labels" not in batch and "label" in batch:
            batch["labels"] = batch["label"]

        required = ["input_ids", "attention_mask", "labels"]
        for k in required:
            if k not in batch:
                raise KeyError(
                    f"Batch is missing key '{k}'. Found keys: {list(batch.keys())}"
                )

        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train the student for one epoch.
        Returns average losses and epoch time.
        """
        self.student.train() # Set student to training mode(droput,layernorm.. ) . teacher is already eval

        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0

        start_time = time.time() # metric for epoch time
        n_batches = 0

        for batch in self.train_loader:
            n_batches += 1
            self.optimizer.zero_grad(set_to_none=True)  # Reset gradients (dont even build computational graph- efficiency)

            batch = self._move_batch_to_device(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Teacher forward pass (frozen)
            with torch.no_grad():
                teacher_out = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_out.logits

            # Student forward pass (trainable)
            student_out = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_out.logits

            # Distillation loss computation
            loss, ce, kl = self.criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
            )
            # Backpropagation and optimization step
            loss.backward()

            # Stabilize training (especially because KL term) - if grad norm is too high - scale it down
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                clip_grad_norm_(self.student.parameters(), self.max_grad_norm)

            self.optimizer.step()  # Update student parameters

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_kl += float(kl.item())

        elapsed = time.time() - start_time

        # Avoid division by zero (shouldn't happen, but for safety!)
        denom = max(1, n_batches)
        return {
            "loss": total_loss / denom,
            "ce": total_ce / denom,
            "kl": total_kl / denom,
            "time_sec": elapsed,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the student on validation set.
        Uses compute_metrics(y_true, y_pred) from src/utils/metrics.py
        """
        self.student.eval()

        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            out = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = out.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(all_labels, all_preds)
        # Expecting compute_metrics to return dict like:
        # {"accuracy": ..., "f1_macro": ..., "f1_weighted": ...}
        return metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation loss:
    L = alpha * CE(student, labels)
        + (1 - alpha) * T^2 * KL(student || teacher)

    - alpha controls the balance between hard labels and soft teacher guidance
    - temperature (T) softens probability distributions
    """

    def __init__(
        self,
        alpha: float = 0.3, # weight for the True labels
        temperature: float = 2.0, #softening factor for teacher
        class_weights: Optional[torch.Tensor] = None, # optional class weights for CE loss
    ):
        super().__init__()

        self.alpha = alpha
        self.temperature = temperature

        # Cross-Entropy for hard labels
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        # KL divergence for soft labels (teacher guidance) - "how different the distribution is "
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Args:
            student_logits: Tensor of shape (B, C)
            teacher_logits: Tensor of shape (B, C)
            labels: Tensor of shape (B,)

        Returns:
            total_loss: scalar tensor
            ce_loss: detached CE loss (for logging)
            kl_loss: detached KL loss (for logging)
        """

        # Hard-label loss (standard supervised learning)
        ce = self.ce_loss(student_logits, labels)  #  labels- ground truth

        # Soft-label loss (knowledge distillation)
        T = self.temperature

        log_probs_student = F.log_softmax(student_logits / T, dim=1) # student is log-prob, teacher is prob
        probs_teacher = F.softmax(teacher_logits / T, dim=1)

        kl = self.kl_loss(log_probs_student, probs_teacher) * (T * T)

        # Combine losses - weighted sum: loss on ground truth + loss on teacher guidance(KL)
        total_loss = self.alpha * ce + (1.0 - self.alpha) * kl

        return total_loss, ce.detach(), kl.detach()

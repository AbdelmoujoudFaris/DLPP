import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. formulation, reimplemented here):

        FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    where p_t is the softmax probability the model assigns to the true
    class. Down-weights easy, already-confident examples (large p_t) so
    training focuses on hard/misclassified ones; gamma=0 recovers plain
    cross-entropy.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        loss = -((1 - p_t) ** self.gamma) * log_p_t
        return loss.mean()


def _build_classification_loss(kind: str, focal_gamma: float, label_smoothing: float) -> nn.Module:
    if kind == "cross_entropy":
        return nn.CrossEntropyLoss()
    if kind == "focal":
        return FocalLoss(gamma=focal_gamma)
    if kind == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss kind '{kind}'")


class MultiTaskLoss(nn.Module):
    """Weighted sum of structure and function classification losses.

    total = weight_structure * L_structure(structure_logits, structure_label)
          + weight_function  * L_function(function_logits, function_label)

    Each task's loss is independently selectable (section 12): plain
    cross-entropy, focal loss, label-smoothed cross-entropy, or (for
    multi-label function taxonomies) BCEWithLogitsLoss against a multi-hot
    target of shape [batch, num_function_classes].
    """

    def __init__(
        self,
        weight_structure: float = 1.0,
        weight_function: float = 1.0,
        structure_loss: str = "cross_entropy",
        function_loss: str = "cross_entropy",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.weight_structure = weight_structure
        self.weight_function = weight_function
        self.function_loss_kind = function_loss
        self.structure_loss = _build_classification_loss(structure_loss, focal_gamma, label_smoothing)
        self.function_loss = _build_classification_loss(function_loss, focal_gamma, label_smoothing)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        structure_label: torch.Tensor,
        function_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_structure = self.structure_loss(outputs["structure_logits"], structure_label)
        if self.function_loss_kind == "bce":
            target = function_label.float()
            if target.dim() == 1:
                # single-label index target: convert to one-hot multi-label form
                num_classes = outputs["function_logits"].shape[1]
                target = F.one_hot(function_label, num_classes=num_classes).float()
            loss_function = self.function_loss(outputs["function_logits"], target)
        else:
            loss_function = self.function_loss(outputs["function_logits"], function_label)
        total = self.weight_structure * loss_structure + self.weight_function * loss_function
        return {"total": total, "structure": loss_structure, "function": loss_function}

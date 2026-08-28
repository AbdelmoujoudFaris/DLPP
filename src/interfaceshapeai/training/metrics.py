import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 over `num_classes` single-label classes.

    Computed directly from per-class TP/FP/FN counts (no sklearn dependency
    needed for this small, batch-level metric); classes absent from both
    predictions and labels contribute an F1 of 0 to the macro average, which
    is the standard scikit-learn convention for undefined per-class F1.
    """
    predictions = logits.argmax(dim=1)
    f1_scores = []
    for cls in range(num_classes):
        true_positive = ((predictions == cls) & (labels == cls)).sum().item()
        false_positive = ((predictions == cls) & (labels != cls)).sum().item()
        false_negative = ((predictions != cls) & (labels == cls)).sum().item()
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def task_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    """Accuracy + macro F1 for one task's logits/labels (section 12/15)."""
    return {
        "accuracy": accuracy(logits, labels),
        "macro_f1": macro_f1(logits, labels, num_classes),
    }

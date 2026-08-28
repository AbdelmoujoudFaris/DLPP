import math

import torch

from interfaceshapeai.training.losses import FocalLoss, MultiTaskLoss


def _dummy_outputs(n_structure=3, n_function=5, batch=4):
    return {
        "structure_logits": torch.randn(batch, n_structure, requires_grad=True),
        "function_logits": torch.randn(batch, n_function, requires_grad=True),
    }


def test_focal_loss_is_finite_and_differentiable():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.randint(0, 3, (4,))
    loss = loss_fn(logits, labels)
    assert math.isfinite(loss.item())
    loss.backward()
    assert logits.grad is not None


def test_multitask_loss_focal_and_label_smoothing_variants():
    for structure_loss, function_loss in [
        ("cross_entropy", "cross_entropy"),
        ("focal", "focal"),
        ("label_smoothing", "label_smoothing"),
    ]:
        outputs = _dummy_outputs()
        loss_fn = MultiTaskLoss(structure_loss=structure_loss, function_loss=function_loss)
        structure_label = torch.randint(0, 3, (4,))
        function_label = torch.randint(0, 5, (4,))
        losses = loss_fn(outputs, structure_label, function_label)
        assert math.isfinite(losses["total"].item())
        losses["total"].backward()


def test_multitask_loss_bce_multilabel_function_head():
    outputs = _dummy_outputs()
    loss_fn = MultiTaskLoss(function_loss="bce")
    structure_label = torch.randint(0, 3, (4,))
    function_label = torch.randint(0, 5, (4,))  # single-label index target, converted internally
    losses = loss_fn(outputs, structure_label, function_label)
    assert math.isfinite(losses["total"].item())
    losses["total"].backward()


def test_weighted_total_combines_both_tasks():
    outputs = _dummy_outputs()
    structure_label = torch.randint(0, 3, (4,))
    function_label = torch.randint(0, 5, (4,))
    loss_fn = MultiTaskLoss(weight_structure=2.0, weight_function=0.5)
    losses = loss_fn(outputs, structure_label, function_label)
    expected = 2.0 * losses["structure"] + 0.5 * losses["function"]
    assert torch.allclose(losses["total"], expected)

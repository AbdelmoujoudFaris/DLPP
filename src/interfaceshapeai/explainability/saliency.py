"""Vanilla-gradient 3D saliency (Simonyan et al. 2013 style, reimplemented
here from first principles - no code copied from any paper or repository).

For a chosen output logit y (a class score from one of the model's task
heads), saliency measures the local sensitivity of y to each input voxel:

    saliency(v) = | dy / dv |

A large value at voxel v means an infinitesimal change to that voxel's
feature values would most strongly change the prediction, i.e. the network
is most sensitive to that region of the interface. The raw gradient is
min-max normalized to [0, 1] purely for display/ranking purposes.
"""

import torch

from interfaceshapeai.models.cnn3d import CNN3D


def compute_saliency(
    model: CNN3D,
    voxel: torch.Tensor,
    target: str = "structure",
    target_class: int | None = None,
) -> torch.Tensor:
    """Returns a [C, D, H, W] saliency map matching `voxel`'s shape.

    target: which head's logits to differentiate ("structure" or "function").
    target_class: class index to explain; defaults to the model's own
        top prediction for that head.
    """
    if voxel.dim() == 4:
        voxel = voxel.unsqueeze(0)

    was_training = model.training
    model.eval()
    try:
        voxel_input = voxel.clone().detach().requires_grad_(True)
        outputs = model(voxel_input)
        logits = outputs[f"{target}_logits"]
        class_index = target_class if target_class is not None else int(logits[0].argmax())
        score = logits[0, class_index]

        (grad,) = torch.autograd.grad(score, voxel_input)
        saliency = grad[0].abs()

        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = torch.zeros_like(saliency)
        return saliency.detach()
    finally:
        model.train(was_training)

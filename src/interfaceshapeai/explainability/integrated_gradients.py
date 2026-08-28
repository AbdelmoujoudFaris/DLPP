"""Integrated Gradients (Sundararajan, Taly & Yan 2017 method, reimplemented
here from first principles - no code copied from any paper or repository).

Attribution for voxel v is the path integral of the gradient of the chosen
output score y along a straight line from a baseline b (default: an
all-zero "no interface" tensor) to the actual input x:

    IG(v) = (x_v - b_v) * integral_{a=0}^{1} [ dy/dv (b + a*(x - b)) ] da

The integral is approximated by a Riemann sum over `steps` equally spaced
points along that path (the standard IG approximation), which is exact in
the limit steps -> infinity. Unlike raw saliency, IG satisfies
"completeness": the sum of all voxel attributions equals y(x) - y(b).
"""

import torch

from interfaceshapeai.models.cnn3d import CNN3D


def integrated_gradients(
    model: CNN3D,
    voxel: torch.Tensor,
    baseline: torch.Tensor | None = None,
    target: str = "structure",
    target_class: int | None = None,
    steps: int = 32,
) -> torch.Tensor:
    """Returns a signed [C, D, H, W] attribution map matching `voxel`'s shape."""
    if voxel.dim() == 4:
        voxel = voxel.unsqueeze(0)
    if baseline is None:
        baseline = torch.zeros_like(voxel)
    elif baseline.dim() == 4:
        baseline = baseline.unsqueeze(0)

    was_training = model.training
    model.eval()
    try:
        voxel = voxel.detach()
        baseline = baseline.detach()
        accumulated_grad = torch.zeros_like(voxel)
        class_index = target_class

        for step in range(1, steps + 1):
            alpha = step / steps
            interpolated = (baseline + alpha * (voxel - baseline)).requires_grad_(True)
            outputs = model(interpolated)
            logits = outputs[f"{target}_logits"]
            if class_index is None:
                class_index = int(logits[0].argmax())
            score = logits[0, class_index]
            (grad,) = torch.autograd.grad(score, interpolated)
            accumulated_grad += grad

        average_grad = accumulated_grad / steps
        attributions = (voxel - baseline) * average_grad
        return attributions[0].detach()
    finally:
        model.train(was_training)

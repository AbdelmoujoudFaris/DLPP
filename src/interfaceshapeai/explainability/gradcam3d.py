"""3D Grad-CAM (Selvaraju et al. 2017 method, extended to 3D and
reimplemented here from first principles - no code copied from any paper or
repository).

Hooks the model's designated target convolutional layer
(`model.get_gradcam_target_layer()`, e.g. CNN3D.encoder's last block), runs
a forward + backward pass for the chosen output class, and computes a
class-discriminative localization map:

    alpha_k = mean over spatial dims (d, h, w) of dY/dA_k       # channel weight
    L        = ReLU( sum_k alpha_k * A_k )                       # weighted activation
    L        = trilinear-upsample(L, size=input spatial shape)

where A_k is the k-th channel of the target layer's activation map (shape
[K, d, h, w], d/h/w smaller than the input grid because of the encoder's
MaxPool3d layers) and Y is the selected class logit. The result is
min-max normalized to [0, 1] for display/ranking.
"""

import torch
import torch.nn.functional as F

from interfaceshapeai.models.cnn3d import CNN3D


def grad_cam_3d(
    model: CNN3D,
    voxel: torch.Tensor,
    target: str = "structure",
    target_class: int | None = None,
) -> torch.Tensor:
    """Returns a [D, H, W] localization map matching `voxel`'s spatial shape."""
    if voxel.dim() == 4:
        voxel = voxel.unsqueeze(0)

    was_training = model.training
    model.eval()

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    layer = model.get_gradcam_target_layer()

    def _forward_hook(_module, _inputs, output):
        activations["value"] = output

    def _backward_hook(_module, _grad_inputs, grad_outputs):
        gradients["value"] = grad_outputs[0]

    forward_handle = layer.register_forward_hook(_forward_hook)
    backward_handle = layer.register_full_backward_hook(_backward_hook)
    try:
        voxel_input = voxel.clone().detach().requires_grad_(True)
        outputs = model(voxel_input)
        logits = outputs[f"{target}_logits"]
        class_index = target_class if target_class is not None else int(logits[0].argmax())
        score = logits[0, class_index]

        model.zero_grad(set_to_none=True)
        score.backward()

        activation = activations["value"][0]  # [K, d, h, w]
        gradient = gradients["value"][0]  # [K, d, h, w]
        channel_weights = gradient.mean(dim=(1, 2, 3))  # [K]
        cam = torch.relu((channel_weights[:, None, None, None] * activation).sum(dim=0))  # [d, h, w]

        cam = F.interpolate(
            cam[None, None], size=voxel_input.shape[-3:], mode="trilinear", align_corners=False
        )[0, 0]

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.detach()
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.train(was_training)

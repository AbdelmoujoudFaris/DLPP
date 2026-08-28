from pathlib import Path

import torch
import torch.nn.functional as F

from interfaceshapeai.models.factory import build_model
from interfaceshapeai.utils.config import Config
from interfaceshapeai.utils.device import resolve_device


class Predictor:
    """Runs a trained (or freshly-initialized demo-mode) model on a voxel tensor.

    IMPORTANT: when no checkpoint is supplied, the model is randomly
    initialized ("demo mode"). Its outputs exercise the architecture and
    I/O pipeline only - they are NOT scientifically meaningful predictions
    (see README "Scientific disclaimer" and master-spec section 41).
    """

    def __init__(self, config: Config, in_channels: int, checkpoint_path: str | Path | None = None):
        self.config = config
        self.device = resolve_device(config.device.type)
        self.model = build_model(config, in_channels=in_channels).to(self.device)
        self.demo_mode = checkpoint_path is None

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

    @torch.no_grad()
    def predict(self, voxel: torch.Tensor) -> dict:
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(0)
        voxel = voxel.to(self.device)

        outputs = self.model(voxel)
        structure_probs = F.softmax(outputs["structure_logits"], dim=1)[0]
        function_probs = F.softmax(outputs["function_logits"], dim=1)[0]

        return {
            "demo_mode": self.demo_mode,
            "structure_prediction": dict(
                zip(self.config.model.structure_classes, structure_probs.tolist())
            ),
            "function_prediction": dict(
                zip(self.config.model.function_classes, function_probs.tolist())
            ),
        }

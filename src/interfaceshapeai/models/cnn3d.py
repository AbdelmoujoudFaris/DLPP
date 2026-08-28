import torch
from torch import nn


def build_conv3d_encoder(in_channels: int, base_filters: int = 16) -> tuple[nn.Sequential, int]:
    """Shared [Conv3D-BN-ReLU(-MaxPool)] x3 convolutional stack.

    Factored out of CNN3D so MultiResolutionCNN can instantiate two
    independent encoders (high-res, low-res) without duplicating the block
    definition. Returns the encoder and its output channel count.
    """
    f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4
    encoder = nn.Sequential(
        nn.Conv3d(in_channels, f1, kernel_size=3, padding=1),
        nn.BatchNorm3d(f1),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.Conv3d(f1, f2, kernel_size=3, padding=1),
        nn.BatchNorm3d(f2),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.Conv3d(f2, f3, kernel_size=3, padding=1),
        nn.BatchNorm3d(f3),
        nn.ReLU(inplace=True),
    )
    return encoder, f3


class CNN3D(nn.Module):
    """Basic single-resolution 3D CNN with a shared encoder and two heads.

    Architecture: [Conv3D-BN-ReLU-MaxPool] x3 -> global average pool ->
    shared FC embedding -> {structure_head, function_head}.

    Multi-resolution variant: see models.multires_cnn.MultiResolutionCNN.
    Residual/attention variants remain on the roadmap.
    """

    def __init__(
        self,
        in_channels: int,
        num_structure_classes: int,
        num_function_classes: int,
        base_filters: int = 16,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder, encoder_out_channels = build_conv3d_encoder(in_channels, base_filters)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.embedding = nn.Sequential(
            nn.Linear(encoder_out_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.structure_head = nn.Linear(embedding_dim, num_structure_classes)
        self.function_head = nn.Linear(embedding_dim, num_function_classes)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Last convolutional block, used by explainability.gradcam3d."""
        return self.encoder

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        pooled = self.global_pool(features).flatten(1)
        embedding = self.embedding(pooled)
        return {
            "structure_logits": self.structure_head(embedding),
            "function_logits": self.function_head(embedding),
            "embedding": embedding,
        }

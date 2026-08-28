"""Two-stage multi-resolution 3D CNN.

Two independent Conv3D encoders (see models.cnn3d.build_conv3d_encoder)
each process a voxelization of the *same* interface at a different
resolution (e.g. a fine 1.0 A grid capturing local topology and a coarse
2.5 A grid capturing global interface shape, see configs/default.yaml
voxelization.high_resolution / low_resolution). Each encoder's output is
global-average-pooled to a fixed-size embedding, and the two embeddings are
combined by a configurable fusion module before the shared structure/function
heads (same two-head pattern as CNN3D).

Fusion methods:
    concatenation (default): out = MLP([h_high ; h_low])
    gated:      out = g * W_high h_high + (1-g) * W_low h_low,
                g = sigmoid(W_gate [h_high ; h_low])  (per-dimension gate)
    attention:  softmax attention over the two resolution embeddings treated
                as a length-2 sequence: out = W (sum_i softmax(score_i) * h_i)
                Requires h_high and h_low to share the same dimensionality
                (true here, since both encoders use the same base_filters).
"""

import torch
from torch import nn


class ConcatFusion(nn.Module):
    def __init__(self, dim_high: int, dim_low: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim_high + dim_low, out_dim), nn.ReLU(inplace=True))

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([high, low], dim=1))


class GatedFusion(nn.Module):
    def __init__(self, dim_high: int, dim_low: int, out_dim: int):
        super().__init__()
        self.proj_high = nn.Linear(dim_high, out_dim)
        self.proj_low = nn.Linear(dim_low, out_dim)
        self.gate = nn.Sequential(nn.Linear(dim_high + dim_low, out_dim), nn.Sigmoid())

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([high, low], dim=1))
        return gate * self.proj_high(high) + (1 - gate) * self.proj_low(low)


class AttentionFusion(nn.Module):
    def __init__(self, dim_high: int, dim_low: int, out_dim: int):
        super().__init__()
        if dim_high != dim_low:
            raise ValueError(
                "attention fusion requires equal high/low embedding dimensions, "
                f"got {dim_high} and {dim_low}"
            )
        self.score = nn.Linear(dim_high, 1)
        self.proj = nn.Linear(dim_high, out_dim)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([high, low], dim=1)  # [B, 2, D]
        weights = torch.softmax(self.score(stacked), dim=1)  # [B, 2, 1]
        fused = (weights * stacked).sum(dim=1)  # [B, D]
        return self.proj(fused)


FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "concatenation": ConcatFusion,
    "gated": GatedFusion,
    "attention": AttentionFusion,
}


def _import_build_conv3d_encoder():
    # local import to avoid a module-level circular import between cnn3d and
    # multires_cnn if either is imported first from models/__init__.py
    from interfaceshapeai.models.cnn3d import build_conv3d_encoder

    return build_conv3d_encoder


class MultiResolutionCNN(nn.Module):
    """High-res encoder + low-res encoder -> fusion -> {structure, function} heads."""

    def __init__(
        self,
        in_channels: int,
        num_structure_classes: int,
        num_function_classes: int,
        base_filters: int = 16,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        fusion: str = "concatenation",
    ):
        super().__init__()
        build_conv3d_encoder = _import_build_conv3d_encoder()
        self.encoder_high, out_high = build_conv3d_encoder(in_channels, base_filters)
        self.encoder_low, out_low = build_conv3d_encoder(in_channels, base_filters)
        self.pool_high = nn.AdaptiveAvgPool3d(1)
        self.pool_low = nn.AdaptiveAvgPool3d(1)

        if fusion not in FUSION_REGISTRY:
            raise ValueError(f"Unknown fusion method '{fusion}'. Available: {list(FUSION_REGISTRY)}")
        self.fusion = FUSION_REGISTRY[fusion](out_high, out_low, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.structure_head = nn.Linear(embedding_dim, num_structure_classes)
        self.function_head = nn.Linear(embedding_dim, num_function_classes)

    def get_gradcam_target_layer(self) -> nn.Module:
        """High-resolution encoder: it carries the finer spatial detail that
        maps most usefully back onto individual interface residues."""
        return self.encoder_high

    def forward(self, voxel_high: torch.Tensor, voxel_low: torch.Tensor) -> dict[str, torch.Tensor]:
        feat_high = self.pool_high(self.encoder_high(voxel_high)).flatten(1)
        feat_low = self.pool_low(self.encoder_low(voxel_low)).flatten(1)
        embedding = self.dropout(self.fusion(feat_high, feat_low))
        return {
            "structure_logits": self.structure_head(embedding),
            "function_logits": self.function_head(embedding),
            "embedding": embedding,
        }

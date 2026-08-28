from torch import nn

from interfaceshapeai.models.cnn3d import CNN3D
from interfaceshapeai.models.multires_cnn import MultiResolutionCNN
from interfaceshapeai.utils.config import Config

# residual3d / attention3d (single-resolution variants) remain on the
# roadmap and are intentionally not stubbed here - adding a fake class would
# violate the "no fake ML" rule (see CHANGELOG/README).
MODEL_REGISTRY = {
    "cnn3d": CNN3D,
    "multires_cnn": MultiResolutionCNN,
}


def build_model(config: Config, in_channels: int) -> nn.Module:
    arch = config.model.architecture
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY)}"
        )
    model_cls = MODEL_REGISTRY[arch]
    kwargs = dict(
        in_channels=in_channels,
        num_structure_classes=len(config.model.structure_classes),
        num_function_classes=len(config.model.function_classes),
        base_filters=config.model.base_filters,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
    )
    if arch == "multires_cnn":
        kwargs["fusion"] = config.model.fusion
    return model_cls(**kwargs)

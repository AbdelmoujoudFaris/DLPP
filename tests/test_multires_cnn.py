import pytest
import torch

from interfaceshapeai.models.factory import MODEL_REGISTRY, build_model
from interfaceshapeai.models.multires_cnn import MultiResolutionCNN
from interfaceshapeai.utils.config import Config


@pytest.mark.parametrize("fusion", ["concatenation", "gated", "attention"])
def test_multires_cnn_forward_shapes(fusion):
    model = MultiResolutionCNN(
        in_channels=4, num_structure_classes=3, num_function_classes=5,
        base_filters=8, embedding_dim=16, fusion=fusion,
    )
    high = torch.rand(2, 4, 16, 16, 16)
    low = torch.rand(2, 4, 8, 8, 8)
    out = model(high, low)
    assert out["structure_logits"].shape == (2, 3)
    assert out["function_logits"].shape == (2, 5)
    assert out["embedding"].shape == (2, 16)


def test_multires_cnn_backward_produces_gradients():
    model = MultiResolutionCNN(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)
    out = model(torch.rand(1, 4, 16, 16, 16), torch.rand(1, 4, 8, 8, 8))
    loss = out["structure_logits"].sum() + out["function_logits"].sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_unknown_fusion_raises():
    with pytest.raises(ValueError):
        MultiResolutionCNN(in_channels=4, num_structure_classes=3, num_function_classes=5, fusion="nope")


def test_model_registry_contains_multires_cnn():
    assert "multires_cnn" in MODEL_REGISTRY


def test_build_model_multires_from_config():
    config = Config(model={"architecture": "multires_cnn", "fusion": "gated"})
    model = build_model(config, in_channels=4)
    grid = config.voxelization.grid_size
    out = model(torch.rand(1, 4, grid, grid, grid), torch.rand(1, 4, grid, grid, grid))
    assert out["structure_logits"].shape == (1, len(config.model.structure_classes))
    assert out["function_logits"].shape == (1, len(config.model.function_classes))

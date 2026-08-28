import torch

from interfaceshapeai.models.cnn3d import CNN3D
from interfaceshapeai.models.factory import MODEL_REGISTRY, build_model
from interfaceshapeai.utils.config import Config


def test_cnn3d_forward_shapes():
    model = CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=32)
    x = torch.rand(2, 4, 16, 16, 16)
    out = model(x)
    assert out["structure_logits"].shape == (2, 3)
    assert out["function_logits"].shape == (2, 5)
    assert out["embedding"].shape == (2, 32)


def test_cnn3d_backward_produces_gradients():
    model = CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=32)
    x = torch.rand(2, 4, 16, 16, 16)
    out = model(x)
    loss = out["structure_logits"].sum() + out["function_logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_model_registry_contains_cnn3d():
    assert "cnn3d" in MODEL_REGISTRY


def test_build_model_from_config():
    config = Config()
    model = build_model(config, in_channels=4)
    x = torch.rand(1, 4, config.voxelization.grid_size, config.voxelization.grid_size, config.voxelization.grid_size)
    out = model(x)
    assert out["structure_logits"].shape == (1, len(config.model.structure_classes))
    assert out["function_logits"].shape == (1, len(config.model.function_classes))

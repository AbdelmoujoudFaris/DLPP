import torch

from interfaceshapeai.explainability.integrated_gradients import integrated_gradients
from interfaceshapeai.models.cnn3d import CNN3D


def _tiny_model():
    return CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)


def test_integrated_gradients_shape():
    model = _tiny_model()
    voxel = torch.rand(4, 16, 16, 16)
    attributions = integrated_gradients(model, voxel, target="function", steps=8)
    assert attributions.shape == voxel.shape


def test_integrated_gradients_zero_input_zero_baseline_gives_zero_attribution():
    model = _tiny_model()
    voxel = torch.zeros(4, 16, 16, 16)
    attributions = integrated_gradients(model, voxel, target="structure", steps=4)
    # (x - baseline) == 0 everywhere when input equals the zero baseline,
    # so attributions must be exactly zero regardless of the gradient term.
    assert torch.allclose(attributions, torch.zeros_like(attributions))


def test_integrated_gradients_custom_baseline():
    model = _tiny_model()
    voxel = torch.rand(4, 16, 16, 16)
    baseline = torch.rand(4, 16, 16, 16)
    attributions = integrated_gradients(model, voxel, baseline=baseline, target="function", steps=4)
    assert attributions.shape == voxel.shape

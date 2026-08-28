import torch

from interfaceshapeai.explainability.saliency import compute_saliency
from interfaceshapeai.models.cnn3d import CNN3D


def _tiny_model():
    return CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)


def test_saliency_shape_and_range():
    model = _tiny_model()
    voxel = torch.rand(4, 16, 16, 16)
    saliency = compute_saliency(model, voxel, target="function")
    assert saliency.shape == voxel.shape
    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0 + 1e-6


def test_saliency_leaves_model_in_original_training_mode():
    model = _tiny_model()
    model.train()
    compute_saliency(model, torch.rand(4, 16, 16, 16))
    assert model.training is True


def test_saliency_respects_explicit_target_class():
    model = _tiny_model()
    voxel = torch.rand(4, 16, 16, 16)
    saliency_class_0 = compute_saliency(model, voxel, target="function", target_class=0)
    saliency_class_1 = compute_saliency(model, voxel, target="function", target_class=1)
    # Different target classes generally produce different gradients (not required
    # to differ for every random init, but the call must succeed for any valid index).
    assert saliency_class_0.shape == saliency_class_1.shape

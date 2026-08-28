import torch

from interfaceshapeai.explainability.gradcam3d import grad_cam_3d
from interfaceshapeai.models.cnn3d import CNN3D


def test_gradcam_shape_matches_input_spatial_resolution():
    model = CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)
    voxel = torch.rand(4, 16, 16, 16)
    cam = grad_cam_3d(model, voxel, target="structure")
    assert cam.shape == (16, 16, 16)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6


def test_gradcam_hooks_are_removed_after_call():
    model = CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)
    grad_cam_3d(model, torch.rand(4, 16, 16, 16))
    assert len(model.encoder._forward_hooks) == 0
    assert len(model.encoder._backward_hooks) == 0


def test_gradcam_works_for_both_heads():
    model = CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)
    voxel = torch.rand(4, 16, 16, 16)
    cam_structure = grad_cam_3d(model, voxel, target="structure")
    cam_function = grad_cam_3d(model, voxel, target="function")
    assert cam_structure.shape == cam_function.shape == (16, 16, 16)

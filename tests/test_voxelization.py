import numpy as np
import torch

from interfaceshapeai.geometry.burial_depth import geometric_burial_depth
from interfaceshapeai.geometry.voxelization import voxelize_interface


def _sample_atoms():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, 10.0, 10.0]]
    )
    residue_names = ["TYR", "ARG", "ASP", "GLY"]
    return coords, residue_names


def test_voxel_tensor_shape():
    coords, residue_names = _sample_atoms()
    depth = geometric_burial_depth(coords)
    channels = ["occupancy", "burial_depth", "hydrophobicity", "charge"]
    voxel = voxelize_interface(coords, residue_names, depth, voxel_size=1.0, grid_size=16, channels=channels)

    assert isinstance(voxel, torch.Tensor)
    assert voxel.shape == (len(channels), 16, 16, 16)


def test_occupancy_channel_is_binary():
    coords, residue_names = _sample_atoms()
    depth = geometric_burial_depth(coords)
    voxel = voxelize_interface(coords, residue_names, depth, grid_size=16, channels=["occupancy"])
    unique_values = torch.unique(voxel)
    assert set(unique_values.tolist()).issubset({0.0, 1.0})
    assert voxel.sum() > 0


def test_empty_atoms_returns_zero_grid():
    voxel = voxelize_interface(
        np.zeros((0, 3)), [], np.zeros(0), grid_size=8, channels=["occupancy"]
    )
    assert voxel.shape == (1, 8, 8, 8)
    assert voxel.sum() == 0

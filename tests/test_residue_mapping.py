import numpy as np
import pytest
import torch

from interfaceshapeai.explainability.residue_mapping import map_voxel_importance_to_residues


def test_residue_mapping_ranks_by_score_descending():
    grid_size = 16
    importance = torch.zeros(grid_size, grid_size, grid_size)
    center = grid_size // 2
    importance[center, center, center] = 1.0  # atom at the origin -> centroid -> center voxel
    importance[center + 2, center, center] = 0.2

    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    residue_ids = [("A", 1), ("A", 2)]
    residue_names = ["TYR", "ASP"]
    depth = np.array([0.9, 0.1])

    rows = map_voxel_importance_to_residues(
        importance, coords, residue_ids, residue_names, depth, voxel_size=1.0, grid_size=grid_size
    )

    assert len(rows) == 2
    assert rows[0]["rank"] == 1
    assert rows[0]["residue_number"] == 1
    assert rows[0]["score"] >= rows[1]["score"]
    assert rows[0]["feature"] == "aromatic"
    assert rows[1]["feature"] == "negative"


def test_residue_mapping_collapses_multichannel_importance():
    grid_size = 8
    importance = torch.zeros(3, grid_size, grid_size, grid_size)  # [C, D, H, W]
    center = grid_size // 2
    importance[1, center, center, center] = 0.7

    coords = np.array([[0.0, 0.0, 0.0]])
    rows = map_voxel_importance_to_residues(
        importance, coords, [("A", 1)], ["ALA"], np.array([0.5]), voxel_size=1.0, grid_size=grid_size
    )
    assert len(rows) == 1
    assert rows[0]["score"] == pytest.approx(0.7)


def test_residue_mapping_empty_coords_returns_empty_list():
    rows = map_voxel_importance_to_residues(
        torch.zeros(8, 8, 8), np.zeros((0, 3)), [], [], np.zeros(0), voxel_size=1.0, grid_size=8
    )
    assert rows == []

import torch

from interfaceshapeai.inference.predictor import Predictor
from interfaceshapeai.utils.config import Config


def test_predictor_demo_mode_returns_valid_distributions():
    config = Config()
    grid = config.voxelization.grid_size
    in_channels = len(config.voxelization.channels)
    predictor = Predictor(config, in_channels=in_channels)

    voxel = torch.rand(in_channels, grid, grid, grid)
    result = predictor.predict(voxel)

    assert result["demo_mode"] is True
    structure_probs = list(result["structure_prediction"].values())
    function_probs = list(result["function_prediction"].values())
    assert abs(sum(structure_probs) - 1.0) < 1e-4
    assert abs(sum(function_probs) - 1.0) < 1e-4
    assert set(result["structure_prediction"]) == set(config.model.structure_classes)

import math

import torch
from torch.utils.data import DataLoader, Dataset

from interfaceshapeai.models.cnn3d import CNN3D
from interfaceshapeai.training.losses import MultiTaskLoss
from interfaceshapeai.training.trainer import build_optimizer, train_one_epoch
from interfaceshapeai.utils.config import Config


class _TinySyntheticDataset(Dataset):
    def __init__(self, n=6, grid=8, channels=4, n_structure=3, n_function=5):
        self.n = n
        self.grid = grid
        self.channels = channels
        self.n_structure = n_structure
        self.n_function = n_function

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "voxel": torch.rand(self.channels, self.grid, self.grid, self.grid),
            "structure_label": torch.randint(0, self.n_structure, (1,)).squeeze(0),
            "function_label": torch.randint(0, self.n_function, (1,)).squeeze(0),
        }


def test_train_one_epoch_runs_and_returns_finite_loss():
    dataset = _TinySyntheticDataset()
    loader = DataLoader(dataset, batch_size=2)
    model = CNN3D(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)
    config = Config()
    optimizer = build_optimizer(model, config)
    loss_fn = MultiTaskLoss(config.training.loss_weight_structure, config.training.loss_weight_function)

    metrics = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["structure_accuracy"] <= 1.0
    assert 0.0 <= metrics["function_accuracy"] <= 1.0

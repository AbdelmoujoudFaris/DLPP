import json
import math

import torch
from torch.utils.data import DataLoader

from interfaceshapeai.datasets.voxel_dataset import MultiResVoxelDataset
from interfaceshapeai.models.multires_cnn import MultiResolutionCNN
from interfaceshapeai.training.losses import MultiTaskLoss
from interfaceshapeai.training.trainer import build_optimizer, train_one_epoch
from interfaceshapeai.utils.config import Config


def _make_multires_dataset_dir(tmp_path):
    records = []
    for i in range(3):
        high = torch.rand(4, 16, 16, 16)
        low = torch.rand(4, 8, 8, 8)
        high_name, low_name = f"high_{i}.pt", f"low_{i}.pt"
        torch.save(high, tmp_path / high_name)
        torch.save(low, tmp_path / low_name)
        records.append(
            {
                "voxel_file_high": high_name,
                "voxel_file_low": low_name,
                "structure_label": i % 3,
                "function_label": i % 2,
            }
        )
    (tmp_path / "labels.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


def test_multires_voxel_dataset_len_and_getitem(tmp_path):
    root = _make_multires_dataset_dir(tmp_path)
    dataset = MultiResVoxelDataset(root)
    assert len(dataset) == 3

    sample = dataset[0]
    assert sample["voxel_high"].shape == (4, 16, 16, 16)
    assert sample["voxel_low"].shape == (4, 8, 8, 8)
    assert sample["structure_label"].dtype == torch.long
    assert sample["function_label"].dtype == torch.long


def test_missing_labels_file_raises(tmp_path):
    try:
        MultiResVoxelDataset(tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_train_one_epoch_dispatches_to_multires_model(tmp_path):
    root = _make_multires_dataset_dir(tmp_path)
    loader = DataLoader(MultiResVoxelDataset(root), batch_size=2)
    model = MultiResolutionCNN(in_channels=4, num_structure_classes=3, num_function_classes=5, base_filters=8, embedding_dim=16)
    config = Config()
    optimizer = build_optimizer(model, config)
    loss_fn = MultiTaskLoss(config.training.loss_weight_structure, config.training.loss_weight_function)

    metrics = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["structure_accuracy"] <= 1.0
    assert 0.0 <= metrics["function_accuracy"] <= 1.0

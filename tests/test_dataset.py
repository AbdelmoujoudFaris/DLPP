import json

import torch

from interfaceshapeai.datasets.voxel_dataset import VoxelDataset


def _make_dataset_dir(tmp_path):
    records = []
    for i in range(3):
        voxel = torch.rand(4, 8, 8, 8)
        file_name = f"sample_{i}.pt"
        torch.save(voxel, tmp_path / file_name)
        records.append({"voxel_file": file_name, "structure_label": i % 3, "function_label": i % 2})
    (tmp_path / "labels.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


def test_voxel_dataset_len_and_getitem(tmp_path):
    root = _make_dataset_dir(tmp_path)
    dataset = VoxelDataset(root)
    assert len(dataset) == 3

    sample = dataset[0]
    assert sample["voxel"].shape == (4, 8, 8, 8)
    assert sample["structure_label"].dtype == torch.long
    assert sample["function_label"].dtype == torch.long


def test_missing_labels_file_raises(tmp_path):
    try:
        VoxelDataset(tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

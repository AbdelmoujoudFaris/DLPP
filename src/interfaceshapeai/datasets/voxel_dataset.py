import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class VoxelDataset(Dataset):
    """Loads precomputed [C, D, H, W] voxel tensors and multi-task labels.

    Expects `root_dir/labels.json`: a list of records
        {"voxel_file": "<relative path>.pt", "structure_label": int, "function_label": int}
    Tensors are loaded lazily (on __getitem__) so large datasets do not need
    to fit in memory at once.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        labels_path = self.root_dir / "labels.json"
        if not labels_path.is_file():
            raise FileNotFoundError(f"labels.json not found in {self.root_dir}")
        self.records: list[dict] = json.loads(labels_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        voxel_path = self.root_dir / record["voxel_file"]
        voxel = torch.load(voxel_path, weights_only=True)
        return {
            "voxel": voxel,
            "structure_label": torch.tensor(record["structure_label"], dtype=torch.long),
            "function_label": torch.tensor(record["function_label"], dtype=torch.long),
        }


class MultiResVoxelDataset(Dataset):
    """Loads paired high/low-resolution voxel tensors for MultiResolutionCNN.

    Expects `root_dir/labels.json`: a list of records
        {"voxel_file_high": "<path>.pt", "voxel_file_low": "<path>.pt",
         "structure_label": int, "function_label": int}
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        labels_path = self.root_dir / "labels.json"
        if not labels_path.is_file():
            raise FileNotFoundError(f"labels.json not found in {self.root_dir}")
        self.records: list[dict] = json.loads(labels_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        voxel_high = torch.load(self.root_dir / record["voxel_file_high"], weights_only=True)
        voxel_low = torch.load(self.root_dir / record["voxel_file_low"], weights_only=True)
        return {
            "voxel_high": voxel_high,
            "voxel_low": voxel_low,
            "structure_label": torch.tensor(record["structure_label"], dtype=torch.long),
            "function_label": torch.tensor(record["function_label"], dtype=torch.long),
        }

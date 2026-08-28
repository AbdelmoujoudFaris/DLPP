import platform
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from interfaceshapeai import __version__
from interfaceshapeai.training.losses import MultiTaskLoss
from interfaceshapeai.training.metrics import accuracy
from interfaceshapeai.utils.config import Config


def software_versions() -> dict[str, str]:
    return {
        "interfaceshapeai": __version__,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def _forward(model: torch.nn.Module, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Architecture-aware forward pass: single-resolution models consume
    batch["voxel"]; MultiResolutionCNN consumes the paired
    batch["voxel_high"]/batch["voxel_low"] produced by MultiResVoxelDataset.
    """
    if "voxel_high" in batch:
        return model(batch["voxel_high"].to(device), batch["voxel_low"].to(device))
    return model(batch["voxel"].to(device))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss, total_structure_acc, total_function_acc, n_batches = 0.0, 0.0, 0.0, 0
    for batch in loader:
        structure_label = batch["structure_label"].to(device)
        function_label = batch["function_label"].to(device)

        optimizer.zero_grad()
        outputs = _forward(model, batch, device)
        losses = loss_fn(outputs, structure_label, function_label)
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        total_structure_acc += accuracy(outputs["structure_logits"], structure_label)
        total_function_acc += accuracy(outputs["function_logits"], function_label)
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {
        "loss": total_loss / n_batches,
        "structure_accuracy": total_structure_acc / n_batches,
        "function_accuracy": total_function_acc / n_batches,
    }


def build_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    optimizer_cls = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam, "SGD": torch.optim.SGD}[
        config.training.optimizer
    ]
    return optimizer_cls(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: Config,
    seed: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config.model_dump(),
        "random_seed": seed,
        "software_versions": software_versions(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

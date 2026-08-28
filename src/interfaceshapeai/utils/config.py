from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "InterfaceShapeAI"
    seed: int = 42


class DeviceConfig(BaseModel):
    type: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class InterfaceConfig(BaseModel):
    distance_cutoff: float = 5.0
    definition: Literal["heavy_atom", "ca"] = "heavy_atom"
    min_contacts: int = 1
    include_heteroatoms: bool = False


class BurialDepthConfig(BaseModel):
    method: Literal["geometric"] = "geometric"
    probe_radius: float = 8.0
    normalize: bool = True


class VoxelizationConfig(BaseModel):
    voxel_size: float = 1.0
    grid_size: int = 32
    channels: list[str] = Field(
        default_factory=lambda: ["occupancy", "burial_depth", "hydrophobicity", "charge"]
    )
    normalization: Literal["global", "per_interface"] = "per_interface"
    # Used only when model.architecture == "multires_cnn": voxel_size/grid_size
    # above are then interpreted as the *high*-resolution grid, and this block
    # sets the second, coarser grid over the same interface (section 7).
    low_resolution_voxel_size: float = 2.5
    low_resolution_grid_size: int = 32


class ModelConfig(BaseModel):
    architecture: str = "cnn3d"
    base_filters: int = 16
    embedding_dim: int = 128
    dropout: float = 0.2
    fusion: Literal["concatenation", "gated", "attention"] = "concatenation"
    structure_classes: list[str] = Field(default_factory=lambda: ["helix", "sheet", "coil"])
    function_classes: list[str] = Field(
        default_factory=lambda: ["enzyme", "dna_binding", "signalling", "immune", "structural"]
    )


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 10


class TrainingConfig(BaseModel):
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    optimizer: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    loss_weight_structure: float = 1.0
    loss_weight_function: float = 1.0
    # classification loss per task; "focal" and "label_smoothing" apply to
    # single-label CrossEntropy-style targets, "bce" treats function_label as
    # a multi-hot multi-label target (section 12).
    structure_loss: Literal["cross_entropy", "focal", "label_smoothing"] = "cross_entropy"
    function_loss: Literal["cross_entropy", "focal", "label_smoothing", "bce"] = "cross_entropy"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)
    burial_depth: BurialDepthConfig = Field(default_factory=BurialDepthConfig)
    voxelization: VoxelizationConfig = Field(default_factory=VoxelizationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load a Config from a YAML file, falling back to documented defaults."""
    if path is None:
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)

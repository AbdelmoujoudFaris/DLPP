"""3D voxelization of a protein-protein interface.

Coordinate -> voxel-index transform:
    1. Compute the centroid of the interface atom coordinates.
    2. Shift coordinates so the centroid sits at the grid center.
    3. index = floor(shifted_coord / voxel_size) + grid_size // 2

    This places the interface centroid at the middle voxel of the cubic
    grid (grid_size, grid_size, grid_size) regardless of the interface's
    absolute position in the PDB coordinate frame. Atoms whose index falls
    outside [0, grid_size) on any axis are dropped (the grid is too small
    to contain them at the chosen voxel_size/grid_size) - callers should
    increase grid_size or voxel_size if this drops a large fraction of atoms.

Multi-atom aggregation:
    When multiple atoms map to the same voxel, each channel takes the
    element-wise max of the contributing atoms' values. This is a standard
    choice for occupancy-style molecular grids: it preserves the strongest
    signal (e.g. the most buried / most hydrophobic atom) in a voxel rather
    than diluting it with an average.

Normalization ("voxelization.normalization" in config):
    - "global": feature values are used as-is. Valid because every channel
      in this module is already produced in a known fixed range upstream
      (occupancy in {0,1}; burial_depth, hydrophobicity in [0,1];
      charge in [-1,1]).
    - "per_interface": additionally min-max rescales each non-occupancy
      channel to [0,1] using only the occupied voxels of *this* interface,
      which can help contrast low-variance interfaces at inference time.
"""

import numpy as np
import torch

from interfaceshapeai.features import chemistry

OCCUPANCY = "occupancy"
BURIAL_DEPTH = "burial_depth"
HYDROPHOBICITY = "hydrophobicity"
CHARGE = "charge"
AROMATICITY = "aromaticity"

SUPPORTED_CHANNELS = (OCCUPANCY, BURIAL_DEPTH, HYDROPHOBICITY, CHARGE, AROMATICITY)


def _channel_values(
    channel: str, residue_names: list[str], burial_depth: np.ndarray, n_atoms: int
) -> np.ndarray:
    if channel == OCCUPANCY:
        return np.ones(n_atoms, dtype=np.float64)
    if channel == BURIAL_DEPTH:
        return burial_depth
    if channel == HYDROPHOBICITY:
        return np.array([chemistry.hydrophobicity(r) for r in residue_names])
    if channel == CHARGE:
        return np.array([chemistry.charge(r) for r in residue_names])
    if channel == AROMATICITY:
        return np.array([chemistry.is_aromatic(r) for r in residue_names])
    raise ValueError(f"Unsupported voxel channel '{channel}'. Supported: {SUPPORTED_CHANNELS}")


def voxelize_interface(
    coords: np.ndarray,
    residue_names: list[str],
    burial_depth: np.ndarray,
    voxel_size: float = 1.0,
    grid_size: int = 32,
    channels: list[str] | None = None,
    normalization: str = "per_interface",
) -> torch.Tensor:
    """Voxelize interface atoms into a [C, D, H, W] torch.Tensor.

    Args:
        coords: (N, 3) atom coordinates in Angstrom.
        residue_names: length-N list of 3-letter residue codes, one per atom.
        burial_depth: (N,) precomputed burial-depth values in [0, 1].
        voxel_size: edge length of one voxel, in Angstrom.
        grid_size: number of voxels per axis (cubic grid).
        channels: which feature channels to produce, in order.
        normalization: "global" or "per_interface" (see module docstring).
    """
    if channels is None:
        channels = [OCCUPANCY, BURIAL_DEPTH, HYDROPHOBICITY, CHARGE]

    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = coords.shape[0]
    grid = torch.zeros((len(channels), grid_size, grid_size, grid_size), dtype=torch.float32)

    if n_atoms == 0:
        return grid

    centroid = coords.mean(axis=0)
    shifted = coords - centroid
    idx = np.floor(shifted / voxel_size).astype(int) + grid_size // 2

    valid = np.all((idx >= 0) & (idx < grid_size), axis=1)
    idx = idx[valid]
    valid_residue_names = [name for name, keep in zip(residue_names, valid) if keep]
    valid_depth = burial_depth[valid] if len(burial_depth) == n_atoms else burial_depth
    n_valid = idx.shape[0]

    for c, channel in enumerate(channels):
        values = _channel_values(channel, valid_residue_names, valid_depth, n_valid)
        for atom_i in range(n_valid):
            x, y, z = idx[atom_i]
            grid[c, x, y, z] = max(grid[c, x, y, z].item(), float(values[atom_i]))

    if normalization == "per_interface":
        occupied = grid[0] > 0
        for c, channel in enumerate(channels):
            if channel == OCCUPANCY or not occupied.any():
                continue
            channel_values = grid[c][occupied]
            v_min, v_max = channel_values.min(), channel_values.max()
            if v_max > v_min:
                grid[c][occupied] = (grid[c][occupied] - v_min) / (v_max - v_min)

    return grid


def voxelize_interface_multires(
    coords: np.ndarray,
    residue_names: list[str],
    burial_depth: np.ndarray,
    high_resolution: tuple[float, int],
    low_resolution: tuple[float, int],
    channels: list[str] | None = None,
    normalization: str = "per_interface",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Voxelize the same interface atoms at two resolutions for
    models.multires_cnn.MultiResolutionCNN.

    high_resolution / low_resolution are (voxel_size, grid_size) pairs
    (section 7's high/low-resolution defaults). Both grids are centered on
    the same atom-coordinate centroid (see voxelize_interface's coordinate
    transform), so a given interface residue occupies corresponding, just
    differently-scaled, regions of each grid.
    """
    high_voxel_size, high_grid_size = high_resolution
    low_voxel_size, low_grid_size = low_resolution
    high = voxelize_interface(
        coords, residue_names, burial_depth,
        voxel_size=high_voxel_size, grid_size=high_grid_size,
        channels=channels, normalization=normalization,
    )
    low = voxelize_interface(
        coords, residue_names, burial_depth,
        voxel_size=low_voxel_size, grid_size=low_grid_size,
        channels=channels, normalization=normalization,
    )
    return high, low

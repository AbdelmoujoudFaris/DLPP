"""Map a per-voxel importance grid (saliency / Grad-CAM / integrated
gradients) back onto interface residues (master spec section 16).

Uses the *identical* coordinate -> voxel-index transform as
geometry.voxelization.voxelize_interface (same centroid-centering, same
floor-division indexing), so voxel (x, y, z) in `importance` corresponds
exactly to the atoms that were binned into it during voxelization:

    idx = floor((atom_coord - atom_centroid) / voxel_size) + grid_size // 2

For each atom that falls inside the grid, its importance score is read off
at its own voxel index; each residue's score is the max (or mean) over its
atoms' scores. Output rows are ranked by score, descending, matching the
rank/chain/residue/score/depth/feature table in the master spec.
"""

from collections import defaultdict

import numpy as np
import torch

from interfaceshapeai.features import chemistry


def map_voxel_importance_to_residues(
    importance: torch.Tensor,
    coords: np.ndarray,
    residue_ids: list[tuple[str, int]],
    residue_names: list[str],
    burial_depth: np.ndarray,
    voxel_size: float = 1.0,
    grid_size: int = 32,
    aggregate: str = "max",
) -> list[dict]:
    """Returns a list of dicts sorted by descending importance:
    {"rank", "chain", "residue_number", "residue_name", "score", "depth", "feature"}.
    """
    if importance.dim() == 4:  # [C, D, H, W] importance/attribution -> collapse channels
        importance = importance.abs().amax(dim=0)
    importance_np = importance.detach().cpu().numpy()

    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = coords.shape[0]
    if n_atoms == 0 or len(residue_ids) != n_atoms:
        return []

    centroid = coords.mean(axis=0)
    shifted = coords - centroid
    idx = np.floor(shifted / voxel_size).astype(int) + grid_size // 2
    valid = np.all((idx >= 0) & (idx < grid_size), axis=1)

    scores_by_residue: dict[tuple[str, int], list[float]] = defaultdict(list)
    depths_by_residue: dict[tuple[str, int], list[float]] = defaultdict(list)
    name_by_residue: dict[tuple[str, int], str] = {}

    for atom_i in range(n_atoms):
        key = residue_ids[atom_i]
        name_by_residue[key] = residue_names[atom_i]
        depths_by_residue[key].append(float(burial_depth[atom_i]))
        if valid[atom_i]:
            x, y, z = idx[atom_i]
            scores_by_residue[key].append(float(importance_np[x, y, z]))

    rows = []
    for key, name in name_by_residue.items():
        scores = scores_by_residue.get(key, [0.0])
        score = max(scores) if aggregate == "max" else float(np.mean(scores))
        rows.append(
            {
                "chain": key[0],
                "residue_number": key[1],
                "residue_name": name,
                "score": score,
                "depth": float(np.mean(depths_by_residue[key])),
                "feature": chemistry.dominant_feature(name),
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows

"""Geometric burial-depth estimation.

v0.1 defines burial depth WITHOUT any external SASA binary (DSSP/FreeSASA
integration is planned for v0.2, selectable via burial_depth.method).

Definition (method="geometric"):
    For an atom i with coordinate x_i, define its local neighbor density as
    the count of *other* atoms in the same coordinate set within a sphere of
    radius `probe_radius`:

        n(i) = |{ j != i : ||x_j - x_i|| <= probe_radius }|

    A solvent-exposed atom (e.g. at the tip of a loop pointing into bulk
    solvent) has few neighbors within the probe sphere; a deeply buried atom
    (interior of a folded domain, or the center of a large interface patch)
    is surrounded on all sides and has many. n(i) is therefore a monotonic,
    scale-free proxy for burial that requires only atomic coordinates.

    When normalize=True, depths are min-max rescaled across the atom set to
    the documented [0, 1] convention: 0 = most exposed atom in the set,
    1 = most buried atom in the set. This makes the exact probe_radius less
    critical since only the *relative* ordering within one interface matters
    for the voxel channel.

This is a coarse, monotone proxy for burial, not a physically calibrated
distance to solvent. It is documented as such and is fully configurable via
`configs/default.yaml: burial_depth`.
"""

import numpy as np


def geometric_burial_depth(
    coords: np.ndarray, probe_radius: float = 8.0, normalize: bool = True
) -> np.ndarray:
    """Compute per-atom geometric burial depth for an (N, 3) coordinate array."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords of shape (N, 3), got {coords.shape}")

    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return np.zeros(0, dtype=np.float64)
    if n_atoms == 1:
        return np.zeros(1, dtype=np.float64)

    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=probe_radius)
    depth = np.array([len(neighbors) - 1 for neighbors in neighbor_lists], dtype=np.float64)
    depth = np.clip(depth, a_min=0.0, a_max=None)

    if normalize:
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

    return depth

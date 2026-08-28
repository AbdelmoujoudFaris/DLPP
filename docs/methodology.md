# Scientific Methodology (v0.1 + v0.2)

## Interface detection

Two residues, one from chain A and one from chain B, are considered part of
the interface if any pair of their selected atoms lies within a distance
cutoff `d` (default 5.0 Å). Atom selection is configurable:

- `heavy_atom`: all non-hydrogen atoms (default).
- `ca`: only the Cα atom (coarser, faster).

A residue is retained as an interface residue if it has contacts with at
least `min_contacts` distinct residues on the other chain.

## Burial depth

v0.1 defines a purely geometric burial depth (no DSSP/FreeSASA dependency):

For atom *i* at coordinate **x**ᵢ, with the full interface atom set as
context:

```
n(i) = |{ j != i : ||x_j - x_i|| <= r }|
```

where `r` is `burial_depth.probe_radius` (default 8.0 Å). `n(i)` counts
neighbors within a sphere of radius `r`. Atoms in the interior of a densely
packed interface have many neighbors (high `n`); atoms at the exposed rim
have few. When `normalize=True`, depths are min-max rescaled across the atom
set to `[0, 1]`, with 0 = most exposed atom and 1 = most buried atom in that
set.

This is an explicit, documented, monotone proxy for burial - not a
physically calibrated solvent-accessible-surface-area measurement. SASA-based
and combined burial-depth definitions are planned for v0.2, gated behind
`burial_depth.method`.

## Voxelization

Given interface atom coordinates, residue names, and per-atom burial depth:

1. Compute the centroid of the atom coordinates.
2. Shift coordinates so the centroid is at the origin.
3. `index = floor(shifted_coord / voxel_size) + grid_size // 2`

This places the interface centroid at the middle of a cubic
`(grid_size, grid_size, grid_size)` grid. Atoms whose index falls outside the
grid on any axis are dropped (increase `grid_size` or `voxel_size` if this
drops many atoms for a given structure).

When multiple atoms map to the same voxel, each channel takes the
element-wise **max** across contributing atoms (not the mean), so the
strongest signal in a voxel (e.g. most buried / most hydrophobic atom) is
preserved.

### Channels (v0.1)

| Channel | Range | Source |
|---|---|---|
| `occupancy` | {0, 1} | 1 if any atom occupies the voxel |
| `burial_depth` | [0, 1] | geometric burial depth (above) |
| `hydrophobicity` | [0, 1] | Kyte & Doolittle (1982) hydropathy, rescaled |
| `charge` | [-1, 1] | formal side-chain charge at physiological pH |
| `aromaticity` | {0, 1} | aromatic side-chain flag |

### Normalization

- `global`: channel values are used as produced above (all channels are
  already in a known fixed range).
- `per_interface`: additionally min-max rescales each non-occupancy channel
  to `[0, 1]` using only the occupied voxels of the current interface.

## Model

A single-resolution 3D CNN (`models.cnn3d.CNN3D`): three
`Conv3d -> BatchNorm3d -> ReLU` blocks (the first two followed by
`MaxPool3d(2)`), global average pooling, a shared fully-connected embedding,
and two linear heads producing `structure_logits` and `function_logits`.

### Multi-resolution model (v0.2)

`models.multires_cnn.MultiResolutionCNN` runs two independent copies of the
same convolutional stack (`models.cnn3d.build_conv3d_encoder`) over a
high-resolution and a low-resolution voxelization of the *same* interface
(`geometry.voxelization.voxelize_interface_multires`; defaults 1.0 Å /
`grid_size` and 2.5 Å / `low_resolution_grid_size`, see
`configs/multires.yaml`). Each encoder output is global-average-pooled to an
embedding `h_high`, `h_low`, then combined by a configurable fusion module
(`model.fusion`) before the same two-head pattern as `CNN3D`:

- **concatenation** (default): `out = ReLU(W [h_high ; h_low] + b)`
- **gated**: `g = sigmoid(W_g [h_high ; h_low] + b_g)`,
  `out = g ⊙ (W_h h_high) + (1-g) ⊙ (W_l h_low)` (element-wise gate, so each
  embedding dimension can weight the two resolutions independently)
- **attention**: treats `[h_high, h_low]` as a length-2 sequence and computes
  `a_i = softmax_i(w^T h_i)`, `out = W (a_high h_high + a_low h_low)`
  (requires `h_high` and `h_low` to share dimensionality, true here since
  both encoders use the same `base_filters`)

## Loss

```
total = w_structure * L_structure(structure_logits, structure_label)
      + w_function  * L_function(function_logits, function_label)
```

with `w_structure`, `w_function` configurable
(`training.loss_weight_structure`, `training.loss_weight_function`).

`L_structure`/`L_function` are independently selectable
(`training.structure_loss` / `training.function_loss`, see
`training.losses.MultiTaskLoss`):

- `cross_entropy` (default): standard softmax cross-entropy.
- `label_smoothing`: cross-entropy against a smoothed target distribution
  (`training.label_smoothing`, default 0.1), which discourages
  over-confident predictions.
- `focal`: `FL(p_t) = -(1 - p_t)^gamma * log(p_t)` where `p_t` is the
  softmax probability assigned to the true class and `gamma`
  (`training.focal_gamma`, default 2.0) down-weights already-confident
  ("easy") examples so training focuses on hard ones.
- `bce` (function head only): treats `function_label` as a multi-label
  target - a single-label class index is converted to a one-hot vector and
  trained with `BCEWithLogitsLoss`, appropriate when a complex can belong to
  more than one functional class simultaneously.

## Explainability (v0.2)

All three methods differentiate a chosen output logit `y` (from either the
structure or function head, `--target`/`target_class`) with respect to the
input voxel tensor `x`, using real `torch.autograd` gradients on the
model - never randomly generated "importance" values.

**Saliency** (`explainability.saliency`, vanilla gradient):

```
saliency(v) = | dy/dv |
```

normalized to `[0, 1]` for display. Measures the local sensitivity of the
prediction to each voxel.

**Integrated Gradients** (`explainability.integrated_gradients`):

```
IG(v) = (x_v - b_v) * (1/steps) * sum_{k=1}^{steps} dy/dv (b + (k/steps)(x - b))
```

a Riemann-sum approximation of the path integral of the gradient from a
baseline `b` (default: an all-zero "no interface" tensor) to `x`. Unlike raw
saliency, IG satisfies *completeness*: summing all voxel attributions
recovers `y(x) - y(b)`.

**Grad-CAM 3D** (`explainability.gradcam3d`): hooks the model's last
convolutional block (`model.get_gradcam_target_layer()`) to obtain its
activation `A` (shape `[K, d, h, w]`) and gradient `dy/dA`:

```
alpha_k = mean over (d, h, w) of dy/dA_k        # per-channel weight
L       = ReLU( sum_k alpha_k * A_k )            # weighted activation map
L       = trilinear_upsample(L, input_shape)     # match input resolution
```

`L` is min-max normalized to `[0, 1]`.

**Residue mapping** (`explainability.residue_mapping`): reuses the exact
coordinate → voxel-index transform from voxelization
(`idx = floor((x - centroid) / voxel_size) + grid_size // 2`) to look up, for
every atom, the importance value at its own voxel; a residue's score is the
max (or mean) over its atoms. Rows are ranked by descending score, matching
the rank/chain/residue/score/depth/feature table used by
`interfaceshapeai explain` and the GUI's Explain tab.

## Roadmap

Residual/single-resolution-attention architectures, DSSP/FreeSASA-based
burial depth, a PyVista 3D saliency overlay, a full multi-epoch `train` CLI
subcommand, and embedding-space clustering are on the v0.3+ roadmap - see
the README.

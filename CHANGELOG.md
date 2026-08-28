# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - v0.2

### Added
- Two-stage multi-resolution 3D CNN with concatenation/gated/attention
  fusion (`models.multires_cnn`), sharing the `cnn3d` conv-block builder.
- Paired high/low-resolution voxelization (`geometry.voxelization.voxelize_interface_multires`)
  and dataset loader (`datasets.voxel_dataset.MultiResVoxelDataset`).
- Architecture-aware training dispatch (`training.trainer.train_one_epoch`
  handles both single- and multi-resolution batches).
- Focal loss, label-smoothed cross-entropy, and multi-label BCE options for
  `training.losses.MultiTaskLoss`; per-task macro-F1 (`training.metrics`).
- Real gradient-based explainability: vanilla-gradient saliency, 3D Grad-CAM,
  and Integrated Gradients (`explainability.*`), plus residue-importance
  mapping back onto interface residues.
- `interfaceshapeai explain` CLI subcommand and a GUI **Explain** tab.
- `configs/multires.yaml` example configuration.

### Known limitations (carried forward / new)
- Explainability supports the `cnn3d` architecture only.
- No full multi-epoch `train` CLI subcommand yet.
- No DSSP/FreeSASA-derived solvent accessibility yet.
- No interactive 3D (PyVista) viewer yet.

## [Unreleased] - v0.1

### Added
- PDB/mmCIF structure parsing via Biopython (`structure.parser`).
- Chain listing and chain-pair selection (`structure.chains`).
- Contact-based protein-protein interface detection (`structure.interface`).
- Geometric burial-depth calculation with documented formula (`geometry.burial_depth`).
- Configurable multi-channel 3D voxelization of interfaces (`geometry.voxelization`).
- Kyte-Doolittle-based residue chemistry feature table (`features.chemistry`).
- Basic 3D CNN with shared encoder and two task heads: secondary-structure and
  functional-class prediction (`models.cnn3d`).
- Model registry for future architecture extensions (`models.factory`).
- Multi-task training loop with checkpointing (`training.trainer`).
- Inference pipeline with explicit demo-mode labeling for untrained weights
  (`inference.predictor`).
- PySide6 desktop GUI with Structure / Interface / Voxelize / Predict tabs.
- CLI: `interfaceshapeai gui|interface|voxelize|predict`.
- Synthetic example-dataset generator (clearly marked as demo data).
- Unit tests covering the full pipeline with tiny synthetic fixtures.
- GitHub Actions CI (tests + lint), Dockerfile, conda environment.

### Known limitations
- No pretrained weights are shipped; predictions in demo mode are architecture
  smoke tests only, not scientifically meaningful.
- No multi-resolution/residual/attention architectures yet (planned v0.2).
- No explainability (saliency/Grad-CAM/Integrated Gradients) yet (planned v0.2).
- No DSSP/FreeSASA-derived solvent accessibility; burial depth is purely
  geometric in v0.1.
- No interactive 3D (PyVista) viewer yet; GUI shows tabular/text results.

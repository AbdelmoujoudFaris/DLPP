# Data directory

- `raw/` - untouched input structures you provide (gitignored).
- `processed/` - extracted interfaces / voxel tensors derived from `raw/` (gitignored).
- `examples/` - synthetic demo data produced by `scripts/generate_example_dataset.py`.
  **DEMO DATA - NOT FOR SCIENTIFIC BENCHMARKING.**

No real structural datasets are bundled with this repository. See
`docs/dataset.md` for how to assemble a real training set from public
structures (e.g. the PDB) under their own licensing terms.

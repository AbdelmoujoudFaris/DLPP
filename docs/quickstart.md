# Quickstart

## 1. Generate demo data (DEMO DATA - NOT FOR SCIENTIFIC BENCHMARKING)

```bash
python scripts/generate_example_dataset.py --output data/examples/synthetic --n-samples 20
```

## 2. Detect an interface from a structure you provide

```bash
interfaceshapeai interface --input complex.pdb --chain-a A --chain-b B
```

## 3. Voxelize it

```bash
interfaceshapeai voxelize --pdb complex.pdb --chain-a A --chain-b B --output interface.pt
```

## 4. Run inference (demo mode - untrained weights, architecture smoke test only)

```bash
interfaceshapeai predict --input complex.pdb --chain-a A --chain-b B
```

## 5. Launch the GUI

```bash
interfaceshapeai gui
```

Walk through the Structure -> Interface -> Voxelize -> Predict tabs in order;
each tab consumes the state produced by the previous one.

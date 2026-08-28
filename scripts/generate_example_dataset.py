"""Generate a small synthetic voxel dataset for pipeline smoke-testing.

DEMO DATA - NOT FOR SCIENTIFIC BENCHMARKING.

The tensors here are random noise shaped like real voxel grids; they let the
training/evaluation/CLI/GUI pipeline be exercised end-to-end without a real
structural dataset. See docs/dataset.md for how to build a real dataset.
"""

import argparse
import json
from pathlib import Path

import torch

CHANNELS = ["occupancy", "burial_depth", "hydrophobicity", "charge"]
NUM_STRUCTURE_CLASSES = 3  # helix, sheet, coil
NUM_FUNCTION_CLASSES = 5  # enzyme, dna_binding, signalling, immune, structural


def generate(output_dir: Path, n_samples: int, grid_size: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(n_samples):
        occupancy = (torch.rand((1, grid_size, grid_size, grid_size), generator=generator) > 0.85).float()
        feature_channels = torch.rand((3, grid_size, grid_size, grid_size), generator=generator) * occupancy
        voxel = torch.cat([occupancy, feature_channels], dim=0)

        file_name = f"sample_{i:04d}.pt"
        torch.save(voxel, output_dir / file_name)
        records.append(
            {
                "voxel_file": file_name,
                "structure_label": int(torch.randint(0, NUM_STRUCTURE_CLASSES, (1,), generator=generator)),
                "function_label": int(torch.randint(0, NUM_FUNCTION_CLASSES, (1,), generator=generator)),
            }
        )

    (output_dir / "labels.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# DEMO DATA - NOT FOR SCIENTIFIC BENCHMARKING\n\n"
        "Randomly generated voxel tensors and labels for exercising the "
        "InterfaceShapeAI pipeline only. Do not use for model evaluation "
        "or to draw biological conclusions.\n",
        encoding="utf-8",
    )
    print(f"Wrote {n_samples} synthetic samples to {output_dir}")
    print("DEMO DATA - NOT FOR SCIENTIFIC BENCHMARKING")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/examples/synthetic", type=Path)
    parser.add_argument("--n-samples", default=20, type=int)
    parser.add_argument("--grid-size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    generate(args.output, args.n_samples, args.grid_size, args.seed)


if __name__ == "__main__":
    main()

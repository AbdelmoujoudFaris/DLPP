import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from interfaceshapeai.geometry.burial_depth import geometric_burial_depth
from interfaceshapeai.geometry.voxelization import voxelize_interface
from interfaceshapeai.inference.predictor import Predictor
from interfaceshapeai.inference.preprocessing import extract_interface_atoms
from interfaceshapeai.structure.chains import select_chain_pair
from interfaceshapeai.structure.interface import detect_interface
from interfaceshapeai.structure.parser import StructureParsingError, load_structure
from interfaceshapeai.utils.config import Config, load_config
from interfaceshapeai.utils.logging import get_logger

logger = get_logger(__name__)


def _load_interface_atoms(input_path: str, chain_a_id: str, chain_b_id: str, config: Config):
    """Shared load -> select chains -> detect interface -> extract atoms
    pipeline used by cmd_predict, cmd_explain, and cmd_voxelize.

    Returns (coords, residue_names, residue_ids) or None (with an error
    already printed to stderr) if the structure/interface is unusable.
    """
    try:
        structure = load_structure(input_path)
        chain_a, chain_b = select_chain_pair(structure, chain_a_id, chain_b_id)
    except StructureParsingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None

    residues = detect_interface(
        chain_a, chain_b,
        distance_cutoff=config.interface.distance_cutoff,
        definition=config.interface.definition,
        min_contacts=config.interface.min_contacts,
        include_heteroatoms=config.interface.include_heteroatoms,
    )
    if not residues:
        print(
            f"No interface detected between chains {chain_a_id} and {chain_b_id}.\n"
            "Try:\n"
            "  - increasing the interface distance cutoff\n"
            "  - selecting another chain pair\n"
            "  - checking whether the structure contains a biological complex",
            file=sys.stderr,
        )
        return None

    coords_a, names_a, ids_a = extract_interface_atoms(chain_a, residues, config.interface.include_heteroatoms)
    coords_b, names_b, ids_b = extract_interface_atoms(chain_b, residues, config.interface.include_heteroatoms)
    coords = np.concatenate([coords_a, coords_b], axis=0)
    residue_names = names_a + names_b
    residue_ids = ids_a + ids_b
    return coords, residue_names, residue_ids


def cmd_interface(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        structure = load_structure(args.input)
        chain_a, chain_b = select_chain_pair(structure, args.chain_a, args.chain_b)
    except StructureParsingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    residues = detect_interface(
        chain_a,
        chain_b,
        distance_cutoff=args.distance_cutoff or config.interface.distance_cutoff,
        definition=config.interface.definition,
        min_contacts=config.interface.min_contacts,
        include_heteroatoms=config.interface.include_heteroatoms,
    )
    if not residues:
        print(
            f"No interface detected between chains {args.chain_a} and {args.chain_b}.\n"
            "Try:\n"
            "  - increasing the interface distance cutoff\n"
            "  - selecting another chain pair\n"
            "  - checking whether the structure contains a biological complex",
            file=sys.stderr,
        )
        return 1

    payload = [r.to_dict() for r in residues]
    output_text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        logger.info("Wrote %d interface residues to %s", len(residues), args.output)
    else:
        print(output_text)
    return 0


def cmd_voxelize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    extracted = _load_interface_atoms(args.pdb, args.chain_a, args.chain_b, config)
    if extracted is None:
        return 1
    coords, residue_names, _residue_ids = extracted

    depth = geometric_burial_depth(
        coords,
        probe_radius=config.burial_depth.probe_radius,
        normalize=config.burial_depth.normalize,
    )
    voxel = voxelize_interface(
        coords,
        residue_names,
        depth,
        voxel_size=config.voxelization.voxel_size,
        grid_size=config.voxelization.grid_size,
        channels=config.voxelization.channels,
        normalization=config.voxelization.normalization,
    )
    torch.save(voxel, args.output)
    logger.info("Wrote voxel tensor %s to %s", tuple(voxel.shape), args.output)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    extracted = _load_interface_atoms(args.input, args.chain_a, args.chain_b, config)
    if extracted is None:
        return 1
    coords, residue_names, _residue_ids = extracted

    depth = geometric_burial_depth(coords, probe_radius=config.burial_depth.probe_radius)
    voxel = voxelize_interface(
        coords, residue_names, depth,
        voxel_size=config.voxelization.voxel_size, grid_size=config.voxelization.grid_size,
        channels=config.voxelization.channels, normalization=config.voxelization.normalization,
    )

    predictor = Predictor(config, in_channels=len(config.voxelization.channels), checkpoint_path=args.model)
    result = predictor.predict(voxel)
    if result["demo_mode"]:
        print("MODEL STATUS: architecture available, no trained weights loaded (demo mode).")
        print("Predictions below are NOT scientifically meaningful.\n")
    print(json.dumps(result, indent=2))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.model.architecture != "cnn3d":
        print(
            f"Error: explainability is currently implemented for architecture 'cnn3d' only "
            f"(config has '{config.model.architecture}'). Multi-resolution explainability is "
            "on the roadmap.",
            file=sys.stderr,
        )
        return 1

    extracted = _load_interface_atoms(args.input, args.chain_a, args.chain_b, config)
    if extracted is None:
        return 1
    coords, residue_names, residue_ids = extracted

    depth = geometric_burial_depth(coords, probe_radius=config.burial_depth.probe_radius)
    voxel = voxelize_interface(
        coords, residue_names, depth,
        voxel_size=config.voxelization.voxel_size, grid_size=config.voxelization.grid_size,
        channels=config.voxelization.channels, normalization=config.voxelization.normalization,
    )

    from interfaceshapeai.explainability.gradcam3d import grad_cam_3d
    from interfaceshapeai.explainability.integrated_gradients import integrated_gradients
    from interfaceshapeai.explainability.residue_mapping import map_voxel_importance_to_residues
    from interfaceshapeai.explainability.saliency import compute_saliency
    from interfaceshapeai.models.factory import build_model
    from interfaceshapeai.utils.device import resolve_device

    device = resolve_device(config.device.type)
    model = build_model(config, in_channels=len(config.voxelization.channels)).to(device)
    demo_mode = args.model is None
    if not demo_mode:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_index = None
    if args.target is not None:
        try:
            target_index = config.model.function_classes.index(args.target)
        except ValueError:
            print(
                f"Error: unknown --target '{args.target}'. Available: {config.model.function_classes}",
                file=sys.stderr,
            )
            return 1

    voxel = voxel.to(device)
    if args.method == "saliency":
        importance = compute_saliency(model, voxel, target="function", target_class=target_index)
    elif args.method == "gradcam":
        importance = grad_cam_3d(model, voxel, target="function", target_class=target_index)
    else:  # integrated_gradients
        importance = integrated_gradients(model, voxel, target="function", target_class=target_index)

    table = map_voxel_importance_to_residues(
        importance, coords, residue_ids, residue_names, depth,
        voxel_size=config.voxelization.voxel_size, grid_size=config.voxelization.grid_size,
    )

    if demo_mode:
        print("MODEL STATUS: architecture available, no trained weights loaded (demo mode).")
        print("Explanations below reflect an untrained model and are NOT scientifically meaningful.\n")
    print(json.dumps(table, indent=2))
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from interfaceshapeai.app.main import run_app

    return run_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interfaceshapeai", description="InterfaceShapeAI CLI")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gui = subparsers.add_parser("gui", help="Launch the desktop GUI")
    p_gui.set_defaults(func=cmd_gui)

    p_iface = subparsers.add_parser("interface", help="Detect a protein-protein interface")
    p_iface.add_argument("--input", required=True)
    p_iface.add_argument("--chain-a", required=True)
    p_iface.add_argument("--chain-b", required=True)
    p_iface.add_argument("--distance-cutoff", type=float, default=None)
    p_iface.add_argument("--output", default=None)
    p_iface.set_defaults(func=cmd_interface)

    p_vox = subparsers.add_parser("voxelize", help="Voxelize an interface from a structure")
    p_vox.add_argument("--pdb", required=True)
    p_vox.add_argument("--chain-a", required=True)
    p_vox.add_argument("--chain-b", required=True)
    p_vox.add_argument("--output", required=True)
    p_vox.set_defaults(func=cmd_voxelize)

    p_pred = subparsers.add_parser("predict", help="Predict on a new complex")
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--chain-a", required=True)
    p_pred.add_argument("--chain-b", required=True)
    p_pred.add_argument("--model", default=None, help="Checkpoint path; omit for demo mode")
    p_pred.set_defaults(func=cmd_predict)

    p_explain = subparsers.add_parser("explain", help="Explain a prediction (saliency / Grad-CAM / integrated gradients)")
    p_explain.add_argument("--input", required=True)
    p_explain.add_argument("--chain-a", required=True)
    p_explain.add_argument("--chain-b", required=True)
    p_explain.add_argument("--model", default=None, help="Checkpoint path; omit for demo mode")
    p_explain.add_argument("--target", default=None, help="Function class to explain; default: model's top prediction")
    p_explain.add_argument(
        "--method", choices=["saliency", "gradcam", "integrated_gradients"], default="gradcam",
    )
    p_explain.set_defaults(func=cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

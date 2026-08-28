import numpy as np
from Bio.PDB.Chain import Chain

from interfaceshapeai.structure.interface import InterfaceResidue


def extract_interface_atoms(
    chain: Chain, interface_residues: list[InterfaceResidue], include_heteroatoms: bool = False
) -> tuple[np.ndarray, list[str], list[tuple[str, int]]]:
    """Collect heavy-atom coordinates, residue names, and (chain, residue_number)
    identifiers for one chain's interface residues, given the residues
    detected by `detect_interface`. The residue-id list lets
    explainability.residue_mapping trace a per-atom voxel index back to the
    specific interface residue it belongs to.
    """
    residue_keys = {(r.chain, r.residue_number) for r in interface_residues}
    coords, names, residue_ids = [], [], []
    for residue in chain:
        if not include_heteroatoms and residue.id[0] != " ":
            continue
        if (chain.id, residue.id[1]) not in residue_keys:
            continue
        for atom in residue:
            if atom.element == "H":
                continue
            coords.append(atom.coord)
            names.append(residue.resname)
            residue_ids.append((chain.id, residue.id[1]))
    coords_array = np.array(coords) if coords else np.zeros((0, 3))
    return coords_array, names, residue_ids

from dataclasses import dataclass, field

import numpy as np
from Bio.PDB.Chain import Chain
from scipy.spatial import cKDTree

HETERO_FLAG_INDEX = 0  # Bio.PDB residue.id = (hetero_flag, resseq, icode)


@dataclass
class ContactRecord:
    chain: str
    residue_number: int
    residue_name: str
    distance: float


@dataclass
class InterfaceResidue:
    chain: str
    residue_number: int
    residue_name: str
    contacts: list[ContactRecord] = field(default_factory=list)
    min_distance: float = float("inf")
    interface: bool = True

    def to_dict(self) -> dict:
        return {
            "chain": self.chain,
            "residue_number": self.residue_number,
            "residue_name": self.residue_name,
            "contacts": [
                {
                    "chain": c.chain,
                    "residue_number": c.residue_number,
                    "residue_name": c.residue_name,
                    "distance": c.distance,
                }
                for c in self.contacts
            ],
            "min_distance": self.min_distance,
            "interface": self.interface,
        }


def _residue_atoms(chain: Chain, definition: str, include_heteroatoms: bool):
    for residue in chain:
        if not include_heteroatoms and residue.id[HETERO_FLAG_INDEX] != " ":
            continue
        for atom in residue:
            if definition == "ca":
                if atom.get_name() == "CA":
                    yield residue, atom
            else:  # heavy_atom
                if atom.element != "H":
                    yield residue, atom


def detect_interface(
    chain_a: Chain,
    chain_b: Chain,
    distance_cutoff: float = 5.0,
    definition: str = "heavy_atom",
    min_contacts: int = 1,
    include_heteroatoms: bool = False,
) -> list[InterfaceResidue]:
    """Detect interface residues between two chains by inter-atomic distance.

    Two residues (one per chain) are in contact if any pair of their selected
    atoms (heavy atoms, or CA-only when definition="ca") lies within
    distance_cutoff Angstrom. A residue is reported as an interface residue
    if it has at least min_contacts contacting residues on the other chain.
    """
    atoms_a = list(_residue_atoms(chain_a, definition, include_heteroatoms))
    atoms_b = list(_residue_atoms(chain_b, definition, include_heteroatoms))
    if not atoms_a or not atoms_b:
        return []

    coords_a = np.array([atom.coord for _, atom in atoms_a])
    coords_b = np.array([atom.coord for _, atom in atoms_b])

    tree_b = cKDTree(coords_b)
    pairs = tree_b.query_ball_point(coords_a, r=distance_cutoff)

    residues: dict[tuple[str, str, int, str], InterfaceResidue] = {}

    def _key(chain_id: str, residue) -> tuple[str, str, int, str]:
        return (chain_id, residue.resname, residue.id[1], str(residue.id[2]))

    for i, hits in enumerate(pairs):
        if not hits:
            continue
        residue_a, atom_a = atoms_a[i]
        key_a = _key(chain_a.id, residue_a)
        for j in hits:
            residue_b, atom_b = atoms_b[j]
            distance = float(np.linalg.norm(atom_a.coord - atom_b.coord))
            key_b = _key(chain_b.id, residue_b)

            for key, own_residue, other_chain_id, other_residue in (
                (key_a, residue_a, chain_b.id, residue_b),
                (key_b, residue_b, chain_a.id, residue_a),
            ):
                entry = residues.setdefault(
                    key,
                    InterfaceResidue(
                        chain=own_residue.get_parent().id,
                        residue_number=own_residue.id[1],
                        residue_name=own_residue.resname,
                    ),
                )
                entry.contacts.append(
                    ContactRecord(
                        chain=other_chain_id,
                        residue_number=other_residue.id[1],
                        residue_name=other_residue.resname,
                        distance=distance,
                    )
                )
                entry.min_distance = min(entry.min_distance, distance)

    result = []
    for entry in residues.values():
        distinct_partners = {(c.chain, c.residue_number) for c in entry.contacts}
        if len(distinct_partners) >= min_contacts:
            result.append(entry)

    result.sort(key=lambda r: (r.chain, r.residue_number))
    return result

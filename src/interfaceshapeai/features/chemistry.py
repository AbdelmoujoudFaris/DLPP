"""Per-residue chemical property lookup tables.

Hydrophobicity values are the Kyte & Doolittle (1982) hydropathy scale,
J. Mol. Biol. 157:105-132, linearly rescaled from its native [-4.5, 4.5]
range to [0, 1] so it can sit alongside other normalized voxel channels.
Charge is the formal side-chain charge at physiological pH (+1 / -1 / 0).
Aromaticity is a binary flag for residues with an aromatic side chain.
Unknown/non-standard residue names fall back to neutral values (0.5, 0, 0).
"""

_KYTE_DOOLITTLE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}
_KD_MIN, _KD_MAX = -4.5, 4.5

_CHARGE = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.5}

_AROMATIC = {"PHE", "TYR", "TRP", "HIS"}


def hydrophobicity(residue_name: str) -> float:
    """Kyte-Doolittle hydropathy rescaled to [0, 1]; 0.5 for unknown residues."""
    raw = _KYTE_DOOLITTLE.get(residue_name.upper())
    if raw is None:
        return 0.5
    return (raw - _KD_MIN) / (_KD_MAX - _KD_MIN)


def charge(residue_name: str) -> float:
    """Formal side-chain charge at physiological pH: -1, 0, +1 (0.5 for His)."""
    return _CHARGE.get(residue_name.upper(), 0.0)


def is_aromatic(residue_name: str) -> float:
    return 1.0 if residue_name.upper() in _AROMATIC else 0.0


def dominant_feature(residue_name: str) -> str:
    """Single dominant chemical label for a residue (explainability tables,
    section 16): aromatic > charged > hydrophobic > polar, in that priority
    order, since e.g. TYR/HIS are both aromatic and (for HIS) weakly charged.
    """
    if is_aromatic(residue_name):
        return "aromatic"
    residue_charge = charge(residue_name)
    if residue_charge > 0:
        return "positive"
    if residue_charge < 0:
        return "negative"
    if hydrophobicity(residue_name) >= 0.6:
        return "hydrophobic"
    return "polar"

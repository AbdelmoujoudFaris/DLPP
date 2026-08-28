from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure

MAX_STRUCTURE_FILE_BYTES = 50 * 1024 * 1024  # 50 MB, untrusted-input guard


class StructureParsingError(Exception):
    """Raised when a structure file cannot be parsed or fails basic validation."""


def load_structure(path: str | Path, structure_id: str = "structure") -> Structure:
    """Parse a PDB or mmCIF file into a Biopython Structure.

    Treats the input as untrusted: only .pdb/.ent/.cif/.mmcif extensions are
    accepted, the file must exist and be under MAX_STRUCTURE_FILE_BYTES, and
    any Biopython parsing error is re-raised as a typed StructureParsingError
    so callers (CLI/GUI) can show a clear message instead of crashing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise StructureParsingError(f"Structure file not found: {file_path}")

    size = file_path.stat().st_size
    if size == 0:
        raise StructureParsingError(f"Structure file is empty: {file_path}")
    if size > MAX_STRUCTURE_FILE_BYTES:
        raise StructureParsingError(
            f"Structure file exceeds {MAX_STRUCTURE_FILE_BYTES} bytes: {file_path}"
        )

    suffix = file_path.suffix.lower()
    try:
        if suffix in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        elif suffix in (".pdb", ".ent"):
            parser = PDBParser(QUIET=True)
        else:
            raise StructureParsingError(
                f"Unsupported structure file extension '{suffix}'. Use .pdb, .ent, .cif or .mmcif."
            )
        structure = parser.get_structure(structure_id, str(file_path))
    except StructureParsingError:
        raise
    except Exception as exc:  # Biopython raises a variety of exception types
        raise StructureParsingError(f"Failed to parse structure {file_path}: {exc}") from exc

    if len(list(structure.get_chains())) == 0:
        raise StructureParsingError(f"No chains found in structure: {file_path}")

    return structure

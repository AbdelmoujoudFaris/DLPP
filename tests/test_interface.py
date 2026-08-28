from interfaceshapeai.structure.chains import ChainSelectionError, select_chain_pair
from interfaceshapeai.structure.interface import detect_interface
from interfaceshapeai.structure.parser import load_structure


def test_detect_interface_finds_close_residues(synthetic_pdb_path):
    structure = load_structure(synthetic_pdb_path)
    chain_a, chain_b = select_chain_pair(structure, "A", "B")

    residues = detect_interface(chain_a, chain_b, distance_cutoff=5.0)

    keys = {(r.chain, r.residue_number) for r in residues}
    assert ("A", 2) in keys  # TYR, close to chain B
    assert ("B", 1) in keys  # ARG, close to chain A
    assert ("A", 3) not in keys  # GLY, isolated far residue
    assert ("B", 3) not in keys  # SER, isolated far residue


def test_detect_interface_no_contacts_returns_empty(synthetic_pdb_path):
    structure = load_structure(synthetic_pdb_path)
    chain_a, chain_b = select_chain_pair(structure, "A", "B")

    residues = detect_interface(chain_a, chain_b, distance_cutoff=0.01)
    assert residues == []


def test_select_chain_pair_missing_chain_raises(synthetic_pdb_path):
    structure = load_structure(synthetic_pdb_path)
    try:
        select_chain_pair(structure, "A", "Z")
        assert False, "expected ChainSelectionError"
    except ChainSelectionError:
        pass


def test_interface_residue_to_dict_shape(synthetic_pdb_path):
    structure = load_structure(synthetic_pdb_path)
    chain_a, chain_b = select_chain_pair(structure, "A", "B")
    residues = detect_interface(chain_a, chain_b, distance_cutoff=5.0)
    payload = residues[0].to_dict()
    assert set(payload) == {"chain", "residue_number", "residue_name", "contacts", "min_distance", "interface"}

import pytest

from interfaceshapeai.structure.parser import StructureParsingError, load_structure


def test_load_structure_lists_chains(synthetic_pdb_path):
    structure = load_structure(synthetic_pdb_path)
    chain_ids = sorted(chain.id for chain in structure.get_chains())
    assert chain_ids == ["A", "B"]


def test_missing_file_raises():
    with pytest.raises(StructureParsingError):
        load_structure("does_not_exist.pdb")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "structure.txt"
    path.write_text("not a structure", encoding="utf-8")
    with pytest.raises(StructureParsingError):
        load_structure(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.pdb"
    path.write_text("", encoding="utf-8")
    with pytest.raises(StructureParsingError):
        load_structure(path)

from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure


class ChainSelectionError(Exception):
    """Raised when a requested chain is missing from a structure."""


def list_chain_ids(structure: Structure, model_index: int = 0) -> list[str]:
    model = list(structure)[model_index]
    return [chain.id for chain in model]


def get_chain(structure: Structure, chain_id: str, model_index: int = 0) -> Chain:
    model = list(structure)[model_index]
    if chain_id not in model:
        available = list_chain_ids(structure, model_index)
        raise ChainSelectionError(
            f"Chain '{chain_id}' not found. Available chains: {available}"
        )
    return model[chain_id]


def select_chain_pair(
    structure: Structure, chain_a_id: str, chain_b_id: str, model_index: int = 0
) -> tuple[Chain, Chain]:
    if chain_a_id == chain_b_id:
        raise ChainSelectionError("Chain A and Chain B must be different chains.")
    chain_a = get_chain(structure, chain_a_id, model_index)
    chain_b = get_chain(structure, chain_b_id, model_index)
    return chain_a, chain_b

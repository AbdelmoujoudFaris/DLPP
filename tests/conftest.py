import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# A tiny synthetic two-chain complex: chain A and chain B each have a few
# residues, with a handful of atoms placed close enough (< 5 A) between the
# chains to form a well-defined interface.
SYNTHETIC_PDB = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      12.000  10.000  10.000  1.00  0.00           C
ATOM      4  N   TYR A   2      13.000  10.000  10.000  1.00  0.00           N
ATOM      5  CA  TYR A   2      14.000  10.000  10.000  1.00  0.00           C
ATOM      6  CZ  TYR A   2      15.000  10.000  10.000  1.00  0.00           C
ATOM      7  N   GLY A   3      30.000  30.000  30.000  1.00  0.00           N
ATOM      8  CA  GLY A   3      31.000  30.000  30.000  1.00  0.00           C
TER
ATOM      9  N   ARG B   1      15.500  10.000  10.000  1.00  0.00           N
ATOM     10  CA  ARG B   1      16.500  10.000  10.000  1.00  0.00           C
ATOM     11  CZ  ARG B   1      17.500  10.000  10.000  1.00  0.00           C
ATOM     12  N   ASP B   2      14.500  11.500  10.000  1.00  0.00           N
ATOM     13  CA  ASP B   2      15.500  11.500  10.000  1.00  0.00           C
ATOM     14  N   SER B   3      50.000  50.000  50.000  1.00  0.00           N
ATOM     15  CA  SER B   3      51.000  50.000  50.000  1.00  0.00           C
TER
END
"""


@pytest.fixture
def synthetic_pdb_path(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.pdb"
    path.write_text(SYNTHETIC_PDB, encoding="utf-8")
    return path

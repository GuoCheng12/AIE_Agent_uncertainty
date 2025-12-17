"""Chemistry utilities for semantic entropy computation."""
from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger
# Disable RDKit error logging (suppress specifically error messages)
# This will stop the "SMILES Parse Error" from printing to stderr
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
# Matches common SMILES tokens; heuristic and not exhaustive
_SMILES_PATTERN = re.compile(r"[A-Za-z0-9@+\-\[\]\\/()=#$]{2,}")


def extract_smiles(text: str) -> Optional[str]:
    """Extract the first SMILES-like substring from free text."""
    match = _SMILES_PATTERN.search(text)
    return match.group(0) if match else None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES using RDKit or ``None`` if parsing fails."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        return None

    if molecule is None:
        return None

    canonical = Chem.MolToSmiles(molecule, canonical=True)
    return canonical or None

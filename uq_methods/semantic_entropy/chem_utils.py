"""Chemistry utilities for semantic entropy computation."""
from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger

# Suppress noisy RDKit parsing errors sent to stderr
RDLogger.DisableLog("rdApp.error")

_SMILES_TAG_PATTERN = re.compile(r"<SMILES>(.*?)</SMILES>", re.IGNORECASE | re.DOTALL)


def extract_smiles(text: str) -> str:
    """Extract SMILES content from `<SMILES>...</SMILES>` tags or return trimmed text."""
    match = _SMILES_TAG_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


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

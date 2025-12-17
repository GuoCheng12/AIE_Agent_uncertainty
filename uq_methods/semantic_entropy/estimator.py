"""Semantic entropy estimator for LLM-generated SMILES."""
from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from llm_interface import LLMGenerator
from uq_methods.base_uq import BaseUQEstimator
from uq_methods.semantic_entropy.chem_utils import canonicalize_smiles, extract_smiles

_INVALID_LABEL = "INVALID"
CHEMISTRY_SYSTEM_PROMPT = (
    "You are a specialized Chemistry Assistant Agent.\n"
    "Your primary task is to generate valid SMILES strings for molecular design tasks.\n"
    "Output Rules:\n"
    "1. When asked to generate a molecule, provide ONLY the SMILES string.\n"
    "2. Do not include markdown formatting, explanations, or chemical names unless explicitly asked.\n"
    "3. If you must provide context, enclose the SMILES string strictly within <SMILES> and </SMILES> tags."
)


class SemanticEntropyEstimator(BaseUQEstimator):
    """Estimate uncertainty via entropy over canonical SMILES clusters."""

    def __init__(self, n_samples: int = 5, temperature: float = 0.7) -> None:
        self.n_samples = n_samples
        self.temperature = temperature

    def estimate_uncertainty(self, prompt: str, llm: LLMGenerator) -> float:
        """Generate samples from ``llm`` and compute semantic entropy of the outputs."""
        chemistry_aligned_prompt = f"{CHEMISTRY_SYSTEM_PROMPT}\n\nUser request:\n{prompt}"
        raw_samples: List[str] = llm.generate(
            chemistry_aligned_prompt, self.n_samples, self.temperature
        )

        canonical_smiles: List[str] = []
        for sample in raw_samples:

            extracted = extract_smiles(sample)
            if extracted is None:
                canonical_smiles.append(_INVALID_LABEL)
                continue

            canonical = canonicalize_smiles(extracted)
            canonical_smiles.append(canonical if canonical is not None else _INVALID_LABEL)

        counts = Counter(canonical_smiles)
        total = sum(counts.values())
        if total == 0:
            return 0.0

        probabilities = np.array([count / total for count in counts.values()], dtype=float)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        return entropy

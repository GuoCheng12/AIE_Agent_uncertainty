"""LLM interface definition for molecular generation."""
from __future__ import annotations

from typing import List, Protocol


class LLMGenerator(Protocol):
    """Protocol for LLM-based SMILES generation."""

    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        """Generate ``n`` SMILES strings for the given prompt at specified temperature."""
        ...

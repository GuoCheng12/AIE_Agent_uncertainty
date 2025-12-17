"""Entry point demonstrating semantic entropy uncertainty estimation."""
from __future__ import annotations

from typing import List

import numpy as np

from llm_interface import LLMGenerator
from uq_methods.semantic_entropy.estimator import SemanticEntropyEstimator


class MockLLM(LLMGenerator):
    """Mock LLM returning hard-coded SMILES variants for testing."""

    def __init__(self, responses: List[str] | None = None) -> None:
        self._responses = responses or ["CCO", "OCC", "InvalidString", "CCO", "OCC"]

    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        # Cycle through the mock responses to simulate sampling behavior
        repeats = (n + len(self._responses) - 1) // len(self._responses)
        samples = (self._responses * repeats)[:n]
        return samples


def main() -> None:
    prompt = "Generate a simple alcohol molecule"
    estimator = SemanticEntropyEstimator(n_samples=5, temperature=0.7)
    llm = MockLLM()

    entropy = estimator.estimate_uncertainty(prompt, llm)
    print(f"Semantic entropy for prompt '{prompt}': {entropy:.4f}")


if __name__ == "__main__":
    main()

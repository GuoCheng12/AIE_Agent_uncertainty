"""Entry point demonstrating semantic entropy uncertainty estimation."""
from __future__ import annotations

from real_llm import LangChainGPT
from uq_methods.semantic_entropy.estimator import SemanticEntropyEstimator


def main() -> None:
    question = "Generate a simple alcohol molecule"
    llm = LangChainGPT()
    estimator = SemanticEntropyEstimator(n_samples=3, temperature=1.0, verbose=True)

    entropy = estimator.estimate_uncertainty(question, llm)
    print(f"Semantic entropy for prompt '{question}': {entropy:.4f}")


if __name__ == "__main__":
    main()

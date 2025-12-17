"""Abstract base class for uncertainty estimators."""
from __future__ import annotations

from abc import ABC, abstractmethod

from llm_interface import LLMGenerator


class BaseUQEstimator(ABC):
    """Base contract for uncertainty estimation strategies."""

    @abstractmethod
    def estimate_uncertainty(self, prompt: str, llm: LLMGenerator) -> float:
        """Estimate and return an uncertainty score for the given prompt using the provided LLM."""
        raise NotImplementedError

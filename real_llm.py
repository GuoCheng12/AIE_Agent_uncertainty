"""LangChain-based LLM driver implementing the LLMGenerator protocol."""
from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from llm_interface import LLMGenerator

# Load environment variables for OpenAI configuration
load_dotenv()


class LangChainGPT(LLMGenerator):
    """LLMGenerator implementation using langchain-openai ChatOpenAI backend."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._base_url = os.getenv("OPENAI_API_BASE")

    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        """Generate ``n`` SMILES samples using the configured chat model."""
        if n <= 0:
            return []

        chat = ChatOpenAI(
            model=self.model,
            temperature=temperature,
            api_key=self._api_key,
            base_url=self._base_url,
        )

        messages = [HumanMessage(content=prompt)]
        outputs: List[str] = []
        for _ in range(n):
            response = chat.invoke(messages)
            outputs.append(response.content)
        return outputs

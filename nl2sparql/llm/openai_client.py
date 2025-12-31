"""OpenAI LLM client implementation."""

import os
from typing import Optional

from .base import LLMClient

# Models that require max_completion_tokens instead of max_tokens
NEWER_MODELS = {"gpt-5.2","gpt-5.1", "gpt-5", "o1", "o1-mini", "o1-preview", "o3", "o3-mini"}


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)

        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install nl2sparql[openai]"
            )

        self._client = OpenAI(api_key=self.api_key)

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        # Check exact match or prefix match for newer models
        for newer_model in NEWER_MODELS:
            if self.model == newer_model or self.model.startswith(f"{newer_model}-"):
                return True
        return False

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        # Build request parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        # Use appropriate token limit parameter based on model
        if self._uses_max_completion_tokens():
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content

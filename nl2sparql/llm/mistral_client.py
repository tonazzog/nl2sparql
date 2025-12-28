"""Mistral AI LLM client implementation."""

import os
from typing import Optional

from .base import LLMClient


class MistralClient(LLMClient):
    """Mistral AI API client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)

        if self.api_key is None:
            self.api_key = os.environ.get("MISTRAL_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "Mistral package not installed. Install with: pip install nl2sparql[mistral]"
            )

        self._client = Mistral(api_key=self.api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        response = self._client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

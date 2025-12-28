"""Anthropic (Claude) LLM client implementation."""

import os
from typing import Optional

from .base import LLMClient


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)

        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install nl2sparql[anthropic]"
            )

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        # Anthropic handles system messages separately
        system_content = ""
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_content if system_content else None,
            messages=filtered_messages,
            temperature=temperature,
            **kwargs,
        )

        # Extract text from content blocks
        return response.content[0].text

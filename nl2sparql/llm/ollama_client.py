"""Ollama (local) LLM client implementation."""

from typing import Optional

from .base import LLMClient


class OllamaClient(LLMClient):
    """Ollama local LLM client."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        **kwargs,
    ):
        # Ollama doesn't need an API key
        super().__init__(model, api_key=None, **kwargs)
        self.host = host

        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Install with: pip install nl2sparql[ollama]"
            )

        self._client = ollama.Client(host=self.host)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        # Ollama uses same message format as OpenAI
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            **kwargs,
        )
        return response["message"]["content"]

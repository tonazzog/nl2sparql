"""Base LLM client abstraction for multiple providers."""

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM client.

        Args:
            model: The model identifier
            api_key: API key (uses environment variable if not provided)
            **kwargs: Additional provider-specific options
        """
        self.model = model
        self.api_key = api_key
        self.options = kwargs

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """
        Generate a completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options

        Returns:
            The generated text response
        """
        pass

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """
        Convenience method for simple system + user prompt pattern.

        Args:
            system_prompt: The system/context prompt
            user_prompt: The user's message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            The generated text response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.complete(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)


def get_client(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Factory function to create an LLM client for the specified provider.

    Args:
        provider: Provider name ("openai", "anthropic", "mistral", "gemini", "ollama")
        model: Model name (uses provider default if not specified)
        api_key: API key (uses environment variable if not provided)
        **kwargs: Additional provider-specific options

    Returns:
        An LLMClient instance

    Raises:
        ValueError: If the provider is not supported
        ImportError: If the provider's package is not installed
    """
    from ..config import AVAILABLE_PROVIDERS

    provider = provider.lower()

    if provider not in AVAILABLE_PROVIDERS:
        available = ", ".join(AVAILABLE_PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

    # Get default model if not specified
    if model is None:
        model = AVAILABLE_PROVIDERS[provider]["default_model"]

    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model=model, api_key=api_key, **kwargs)

    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model=model, api_key=api_key, **kwargs)

    elif provider == "mistral":
        from .mistral_client import MistralClient
        return MistralClient(model=model, api_key=api_key, **kwargs)

    elif provider == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient(model=model, api_key=api_key, **kwargs)

    elif provider == "ollama":
        from .ollama_client import OllamaClient
        return OllamaClient(model=model, **kwargs)

    else:
        raise ValueError(f"Provider '{provider}' not implemented")

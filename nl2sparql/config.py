"""Configuration management for nl2sparql."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Default SPARQL endpoint for LiITA
LIITA_ENDPOINT = "https://liita.it/sparql"

# Default embedding model for semantic search
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Retriever default weights (semantic, bm25, pattern)
DEFAULT_RETRIEVER_WEIGHTS = (0.4, 0.3, 0.3)

# Package data directory
PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"
DATASET_PATH = DATA_DIR / "sparql_queries_final.json"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    api_key: Optional[str] = None

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "mistral": "MISTRAL_API_KEY",
                "gemini": "GEMINI_API_KEY",
                # Ollama doesn't need an API key
            }
            env_var = env_var_map.get(self.provider)
            if env_var:
                self.api_key = os.environ.get(env_var)


@dataclass
class RetrieverConfig:
    """Configuration for the hybrid retriever."""

    dataset_path: Path = field(default_factory=lambda: DATASET_PATH)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    weights: tuple[float, float, float] = DEFAULT_RETRIEVER_WEIGHTS
    top_k: int = 5


@dataclass
class ValidationConfig:
    """Configuration for SPARQL validation."""

    endpoint: str = LIITA_ENDPOINT
    timeout: int = 30
    check_syntax: bool = True
    check_endpoint: bool = True
    check_semantic: bool = True


@dataclass
class NL2SPARQLConfig:
    """Main configuration for the NL2SPARQL system."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Generation options
    fix_errors: bool = True
    max_retries: int = 3

    @classmethod
    def from_provider(
        cls,
        provider: str,
        model: str,
        **kwargs
    ) -> "NL2SPARQLConfig":
        """Create config with specified LLM provider."""
        llm_config = LLMConfig(provider=provider, model=model)
        return cls(llm=llm_config, **kwargs)


# Available LLM providers and their default models
AVAILABLE_PROVIDERS = {
    "openai": {
        "default_model": "gpt-4.1-mini",
        "models": ["gpt-5.2","gpt-4.1","gpt-4.1-mini","gpt-4.1-nano","gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "anthropic": {
        "default_model": "claude-sonnet-4-20250514",
        "models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
    },
    "mistral": {
        "default_model": "mistral-large-latest",
        "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
    },
    "gemini": {
        "default_model": "gemini-pro",
        "models": ["gemini-pro", "gemini-pro-vision"],
    },
    "ollama": {
        "default_model": "llama3",
        "models": ["llama3", "mistral", "codellama", "phi3"],
    },
}

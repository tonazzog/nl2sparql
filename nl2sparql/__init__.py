"""
NL2SPARQL: Natural Language to SPARQL translation for the LiITA knowledge base.

This package provides tools to translate natural language questions (primarily in Italian)
into SPARQL queries for querying the LiITA (Linked Italian) linguistic knowledge base.

Basic usage:
    >>> from nl2sparql import translate
    >>> result = translate("Quali lemmi esprimono tristezza?")
    >>> print(result.sparql)

Advanced usage:
    >>> from nl2sparql import NL2SPARQL
    >>> translator = NL2SPARQL(provider="anthropic", model="claude-sonnet-4-20250514")
    >>> result = translator.translate("Trova traduzioni siciliane di casa")
"""

__version__ = "0.1.0"

from dataclasses import dataclass, field
from typing import Optional

from .config import (
    NL2SPARQLConfig,
    LLMConfig,
    RetrieverConfig,
    ValidationConfig,
    AVAILABLE_PROVIDERS,
    LIITA_ENDPOINT,
)


@dataclass
class QueryExample:
    """A natural language / SPARQL query pair from the dataset."""

    id: int
    sparql: str
    nl: str
    nl_variants: dict[str, str] = field(default_factory=dict)
    patterns: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from the hybrid retriever."""

    example: QueryExample
    score: float
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    pattern_score: float = 0.0


@dataclass
class ValidationResult:
    """Result of SPARQL validation."""

    syntax_valid: bool
    syntax_error: Optional[str] = None
    execution_success: Optional[bool] = None
    execution_error: Optional[str] = None
    result_count: Optional[int] = None
    sample_results: Optional[list] = None
    semantic_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if the query is fully valid."""
        return (
            self.syntax_valid
            and (self.execution_success is None or self.execution_success)
            and len(self.semantic_errors) == 0
        )


@dataclass
class TranslationResult:
    """Result of translating a natural language question to SPARQL."""

    question: str
    sparql: str
    validation: Optional[ValidationResult] = None
    retrieved_examples: list[RetrievalResult] = field(default_factory=list)
    detected_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0
    was_fixed: bool = False
    fix_attempts: int = 0


# Lazy imports to avoid loading heavy dependencies at import time
def _get_translator():
    """Lazy load the NL2SPARQL class."""
    from .generation.synthesizer import NL2SPARQL
    return NL2SPARQL


def translate(
    question: str,
    provider: str = "openai",
    model: Optional[str] = None,
    validate: bool = True,
    fix_errors: bool = True,
    max_retries: int = 3,
    endpoint: str = LIITA_ENDPOINT,
) -> TranslationResult:
    """
    Translate a natural language question to a SPARQL query.

    This is the main entry point for simple usage. For more control,
    use the NL2SPARQL class directly.

    Args:
        question: Natural language question (typically in Italian)
        provider: LLM provider ("openai", "anthropic", "mistral", "gemini", "ollama")
        model: Model name (uses provider default if not specified)
        validate: Whether to validate the generated query
        fix_errors: Whether to attempt to fix invalid queries
        max_retries: Maximum number of fix attempts
        endpoint: SPARQL endpoint URL for validation

    Returns:
        TranslationResult containing the SPARQL query and metadata

    Example:
        >>> result = translate("Quali lemmi esprimono tristezza?")
        >>> print(result.sparql)
    """
    if model is None:
        model = AVAILABLE_PROVIDERS.get(provider, {}).get("default_model", "gpt-4.1")

    NL2SPARQL = _get_translator()
    translator = NL2SPARQL(
        provider=provider,
        model=model,
        validate=validate,
        fix_errors=fix_errors,
        max_retries=max_retries,
        endpoint=endpoint,
    )
    return translator.translate(question)


# Re-export key classes and functions
__all__ = [
    # Main API
    "translate",
    "NL2SPARQL",
    # Types
    "QueryExample",
    "RetrievalResult",
    "ValidationResult",
    "TranslationResult",
    # Configuration
    "NL2SPARQLConfig",
    "LLMConfig",
    "RetrieverConfig",
    "ValidationConfig",
    "AVAILABLE_PROVIDERS",
    "LIITA_ENDPOINT",
    # Version
    "__version__",
]


# Lazy export for NL2SPARQL class
def __getattr__(name):
    if name == "NL2SPARQL":
        return _get_translator()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

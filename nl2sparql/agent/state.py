"""State definition for the NL2SPARQL LangGraph agent."""

from typing import TypedDict, Literal, Annotated
from operator import add


class NL2SPARQLState(TypedDict):
    """State for the NL2SPARQL agent workflow."""

    # Input
    question: str
    language: str  # "it" or "en"

    # LLM Configuration
    provider: str  # "openai", "anthropic", "mistral", etc.
    model: str | None  # Model name (uses provider default if None)
    api_key: str | None  # API key (uses environment variable if None)

    # Analysis
    detected_patterns: list[str]
    complexity: Literal["simple", "moderate", "complex"]
    requires_service: bool
    requires_translation: bool
    dialects_needed: list[str]  # ["sicilian", "parmigiano"]

    # Planning
    sub_tasks: list[str]
    current_task_index: int

    # Retrieval
    retrieved_examples: list[dict]
    relevant_constraints: str

    # Generation
    generated_sparql: str
    generation_attempts: int

    # Execution
    execution_result: list[dict] | None
    result_count: int
    execution_error: str | None

    # Verification
    is_valid: bool
    validation_errors: list[str]

    # Refinement - uses Annotated to accumulate history
    refinement_history: Annotated[list[dict], add]

    # Schema exploration
    discovered_properties: list[str]
    schema_explored: bool

    # Output
    final_sparql: str
    confidence: float
    explanation: str


def create_initial_state(
    question: str,
    language: str = "it",
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
) -> NL2SPARQLState:
    """Create initial state for a translation request."""
    return NL2SPARQLState(
        # Input
        question=question,
        language=language,
        # LLM Configuration
        provider=provider,
        model=model,
        api_key=api_key,
        # Analysis
        detected_patterns=[],
        complexity="simple",
        requires_service=False,
        requires_translation=False,
        dialects_needed=[],
        # Planning
        sub_tasks=[],
        current_task_index=0,
        # Retrieval
        retrieved_examples=[],
        relevant_constraints="",
        # Generation
        generated_sparql="",
        generation_attempts=0,
        # Execution
        execution_result=None,
        result_count=0,
        execution_error=None,
        # Verification
        is_valid=False,
        validation_errors=[],
        # Refinement
        refinement_history=[],
        # Schema
        discovered_properties=[],
        schema_explored=False,
        # Output
        final_sparql="",
        confidence=0.0,
        explanation="",
    )

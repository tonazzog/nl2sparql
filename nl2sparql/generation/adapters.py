"""Query adaptation and synthesis functions."""

import re
from typing import TYPE_CHECKING

from ..constraints.prompt_builder import (
    build_synthesis_prompt,
    build_fix_prompt,
    build_adaptation_prompt,
)

if TYPE_CHECKING:
    from .. import QueryExample
    from ..llm.base import LLMClient


def extract_sparql_from_response(response: str) -> str:
    """
    Extract SPARQL query from LLM response.

    Handles responses with:
    - Code blocks (```sparql ... ```)
    - Plain SPARQL starting with PREFIX or SELECT
    - Mixed text with embedded queries

    Args:
        response: Raw LLM response

    Returns:
        Extracted SPARQL query
    """
    # Try to extract from code block
    code_block_match = re.search(
        r"```(?:sparql)?\s*(.*?)```",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    # Look for query starting with PREFIX or SELECT
    query_match = re.search(
        r"(PREFIX\s+.+?(?:SELECT|ASK|CONSTRUCT|DESCRIBE).+?)(?:\n\n|\Z)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if query_match:
        return query_match.group(1).strip()

    # Look for SELECT without PREFIX
    select_match = re.search(
        r"(SELECT\s+.+?)(?:\n\n|\Z)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if select_match:
        return select_match.group(1).strip()

    # Return the whole response stripped
    return response.strip()


def adapt_query(
    example: "QueryExample",
    user_question: str,
    client: "LLMClient",
    detected_patterns: list[str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Adapt an existing query to answer a new question.

    This is used when the retrieval finds a highly similar example
    that can be modified rather than synthesized from scratch.

    Args:
        example: The example query to adapt
        user_question: The new question to answer
        client: LLM client for generation
        detected_patterns: Detected query patterns
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Adapted SPARQL query
    """
    system_prompt, user_prompt = build_adaptation_prompt(
        example=example,
        user_question=user_question,
        detected_patterns=detected_patterns,
    )

    response = client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return extract_sparql_from_response(response)


def synthesize_query(
    user_question: str,
    exemplars: list["QueryExample"],
    client: "LLMClient",
    detected_patterns: list[str],
    schema_context: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Synthesize a new SPARQL query from exemplars.

    This is used when no single example is similar enough,
    and we need to combine patterns from multiple examples.

    Args:
        user_question: The question to answer
        exemplars: Retrieved example queries for few-shot learning
        client: LLM client for generation
        detected_patterns: Detected query patterns
        schema_context: Additional schema information
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Synthesized SPARQL query
    """
    system_prompt, user_prompt = build_synthesis_prompt(
        user_question=user_question,
        exemplars=exemplars,
        detected_patterns=detected_patterns,
        schema_context=schema_context,
    )

    response = client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return extract_sparql_from_response(response)


def fix_query(
    sparql: str,
    error: str,
    client: "LLMClient",
    detected_patterns: list[str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Attempt to fix an invalid SPARQL query.

    Args:
        sparql: The invalid SPARQL query
        error: The error message
        client: LLM client for generation
        detected_patterns: Detected query patterns
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Fixed SPARQL query
    """
    system_prompt, user_prompt = build_fix_prompt(
        sparql=sparql,
        error=error,
        detected_patterns=detected_patterns,
    )

    response = client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return extract_sparql_from_response(response)

"""Constraint modules for LiITA SPARQL query generation.

This package contains the domain-specific prompts, patterns, and validation
rules for generating valid SPARQL queries for the LiITA knowledge base.
"""

from .base import SPARQL_PREFIXES, SYSTEM_PROMPT
from .emotion import EMOTION_MANDATORY_PATTERNS, validate_emotion_query
from .translation import TRANSLATION_MANDATORY_PATTERNS, validate_translation_query
from .semantic import SEMANTIC_MANDATORY_PATTERNS, validate_semantic_query
from .multi_entry import MULTI_ENTRY_CRITICAL_PATTERN, validate_multi_entry_pattern
from .compositional import COMPOSITIONAL_PATTERNS, detect_compositional_pattern
from .prompt_builder import build_synthesis_prompt, get_constraints_for_patterns

__all__ = [
    # Base
    "SPARQL_PREFIXES",
    "SYSTEM_PROMPT",
    # Patterns
    "EMOTION_MANDATORY_PATTERNS",
    "TRANSLATION_MANDATORY_PATTERNS",
    "SEMANTIC_MANDATORY_PATTERNS",
    "MULTI_ENTRY_CRITICAL_PATTERN",
    "COMPOSITIONAL_PATTERNS",
    # Validation
    "validate_emotion_query",
    "validate_translation_query",
    "validate_semantic_query",
    "validate_multi_entry_pattern",
    "detect_compositional_pattern",
    # Prompt building
    "build_synthesis_prompt",
    "get_constraints_for_patterns",
]

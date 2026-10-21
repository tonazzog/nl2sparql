"""Dynamic prompt construction for SPARQL synthesis.

This module builds prompts by combining the base system prompt with
relevant constraint sections based on detected query patterns.
"""

import os
from typing import TYPE_CHECKING

from .base import SPARQL_PREFIXES, SYSTEM_PROMPT, PATTERN_CATEGORIES
from .emotion import EMOTION_MANDATORY_PATTERNS
from .translation import TRANSLATION_MANDATORY_PATTERNS
from .semantic import SEMANTIC_MANDATORY_PATTERNS
from .multi_entry import MULTI_ENTRY_CRITICAL_PATTERN
from .compositional import COMPOSITIONAL_PATTERNS
from .lexical_relation import LEXICAL_RELATION_MANDATORY_PATTERNS

if TYPE_CHECKING:
    from .. import QueryExample


def get_constraints_for_patterns(patterns: list[str]) -> str:
    """
    Get appropriate constraint sections based on detected query patterns.

    Args:
        patterns: List of pattern names (e.g., ["EMOTION_LEXICON", "TRANSLATION"])

    Returns:
        Concatenated constraint documentation string
    """
    # Experiment toggle: inject every constraint block regardless of detected
    # patterns, so a missed or wrong pattern detection can't leave the LLM
    # without a rule it needs (e.g. the inverted partMeronym direction). Costs
    # longer prompts and possibly distracting rules. Off by default.
    if os.getenv("NL2SPARQL_ALL_CONSTRAINTS", "").lower() in ("1", "true", "yes"):
        patterns = list(PATTERN_CATEGORIES)

    # Map patterns to categories
    categories = set()
    for pattern in patterns:
        category = PATTERN_CATEGORIES.get(pattern)
        if category:
            categories.add(category)

    constraints = []

    has_emotion = "emotion" in categories
    has_translation = "translation" in categories
    has_semantic = "sense_definition" in categories or "semantic_relation" in categories
    has_lexical_relation = "lexical_relation" in categories
    has_morphology = "morphology" in categories
    has_compositional = "COMPOSITIONAL" in patterns

    # Multi-entry pattern needed if emotion + anything else
    if has_emotion and (has_translation or has_semantic or has_morphology or has_lexical_relation):
        constraints.append("\n## MULTI-ENTRY PATTERN (CRITICAL):\n")
        constraints.append(MULTI_ENTRY_CRITICAL_PATTERN)

    if has_emotion:
        constraints.append("\n## EMOTION CONSTRAINTS:\n")
        constraints.append(EMOTION_MANDATORY_PATTERNS)

    if has_translation:
        constraints.append("\n## TRANSLATION CONSTRAINTS:\n")
        constraints.append(TRANSLATION_MANDATORY_PATTERNS)

    if has_semantic:
        constraints.append("\n## SEMANTIC/SENSE CONSTRAINTS:\n")
        constraints.append(SEMANTIC_MANDATORY_PATTERNS)

    # Lexical relations (synonyms, antonyms) - separate from hierarchical semantic relations
    if has_lexical_relation:
        constraints.append("\n## LEXICAL RELATION CONSTRAINTS (SYNONYM/ANTONYM):\n")
        constraints.append(LEXICAL_RELATION_MANDATORY_PATTERNS)

    # Compositional reasoning for complex multi-step queries
    if has_compositional or has_semantic or has_lexical_relation:
        constraints.append("\n## COMPOSITIONAL QUERY REASONING:\n")
        constraints.append(COMPOSITIONAL_PATTERNS)

    return "\n".join(constraints) if constraints else ""


def format_exemplars(exemplars: list["QueryExample"]) -> str:
    """
    Format exemplars as few-shot examples for the prompt.

    Args:
        exemplars: List of QueryExample objects

    Returns:
        Formatted examples string
    """
    formatted = []
    for i, ex in enumerate(exemplars, 1):
        # Get the most appropriate NL variant
        nl = ex.nl
        if ex.nl_variants:
            nl = ex.nl_variants.get("generic", ex.nl)

        # Extract top patterns
        top_patterns = sorted(ex.patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        pattern_str = ", ".join(p[0] for p in top_patterns if p[1] > 0.5)

        formatted.append(
            f"### Example {i}" + (f" (Patterns: {pattern_str})" if pattern_str else "") + f":\n"
            f"Question: {nl}\n"
            f"SPARQL:\n```sparql\n{ex.sparql}\n```\n"
        )

    return "\n".join(formatted)


def format_known_failures(known_failures: list[str]) -> str:
    """
    Format remembered past mistakes (for similar past questions) as a
    prompt section warning the LLM away from repeating them.

    Args:
        known_failures: Error messages from remembered failed attempts

    Returns:
        Formatted "known past mistakes" section, or "" if none
    """
    if not known_failures:
        return ""
    lines = "\n".join(f"- {err}" for err in known_failures[:3])
    return f"\n## KNOWN PAST MISTAKES FOR SIMILAR QUESTIONS (avoid these):\n{lines}\n"


def build_synthesis_prompt(
    user_question: str,
    exemplars: list["QueryExample"],
    detected_patterns: list[str],
    schema_context: str = "",
    known_failures: list[str] | None = None,
) -> tuple[str, str]:
    """
    Build a complete synthesis prompt for SPARQL generation.

    Args:
        user_question: The natural language question to translate
        exemplars: Retrieved example queries for few-shot learning
        detected_patterns: Patterns detected in the user question
        schema_context: Optional additional schema information
        known_failures: Optional list of remembered past-mistake error
            messages for similar past questions (from translation memory)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Get relevant constraints
    constraints = get_constraints_for_patterns(detected_patterns)

    # Format exemplars
    examples_text = format_exemplars(exemplars) if exemplars else ""

    # Format remembered past mistakes, if any
    failures_text = format_known_failures(known_failures or [])

    # Build system prompt
    system_prompt = f"""{SYSTEM_PROMPT}

{constraints}

{schema_context}
"""

    # Build user prompt
    user_prompt = f"""## SPARQL PREFIXES:
{SPARQL_PREFIXES}

## EXAMPLE QUERIES:
{examples_text if examples_text else "No examples available."}
{failures_text}

## YOUR TASK:

Translate the following natural language question into a valid SPARQL query for the LiITA knowledge base.

**Question**: {user_question}

**Requirements**:
1. Include all necessary PREFIX declarations
2. Follow the mandatory patterns for the query type
3. Use correct graph locations for each property
4. Use appropriate LIMIT (10-50 results)
5. Output ONLY the SPARQL query, no explanations

**SPARQL Query**:
"""

    return system_prompt, user_prompt


def build_fix_prompt(sparql: str, error: str, detected_patterns: list[str]) -> tuple[str, str]:
    """
    Build a prompt for fixing an invalid SPARQL query.

    Args:
        sparql: The invalid SPARQL query
        error: The error message
        detected_patterns: Patterns detected in the original question

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    constraints = get_constraints_for_patterns(detected_patterns)

    system_prompt = f"""{SYSTEM_PROMPT}

{constraints}
"""

    user_prompt = f"""## FIX SPARQL QUERY

The following SPARQL query has an error:

```sparql
{sparql}
```

**Error**: {error}

**Instructions**:
1. Analyze the error message
2. Check the query against the mandatory patterns above
3. Fix the structural issues
4. Return ONLY the corrected SPARQL query

**Corrected Query**:
"""

    return system_prompt, user_prompt


def build_adaptation_prompt(
    example: "QueryExample",
    user_question: str,
    detected_patterns: list[str],
    known_failures: list[str] | None = None,
) -> tuple[str, str]:
    """
    Build a prompt for adapting an existing query to a new question.

    Args:
        example: The example query to adapt
        user_question: The new question to answer
        detected_patterns: Detected patterns
        known_failures: Optional list of remembered past-mistake error
            messages for similar past questions (from translation memory)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    constraints = get_constraints_for_patterns(detected_patterns)
    failures_text = format_known_failures(known_failures or [])

    system_prompt = f"""{SYSTEM_PROMPT}

{constraints}
"""

    user_prompt = f"""## ADAPT QUERY

You have a working SPARQL query that answers a similar question. Adapt it to answer the new question.

**Original Question**: {example.nl}

**Original Query**:
```sparql
{example.sparql}
```

**New Question**: {user_question}
{failures_text}
**Instructions**:
1. Identify what needs to change (entities, filters, properties)
2. Keep the structural pattern intact
3. Modify only the necessary parts
4. Output ONLY the adapted SPARQL query

**Adapted Query**:
"""

    return system_prompt, user_prompt

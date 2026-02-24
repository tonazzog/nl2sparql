"""Semantic validation using constraint modules."""

import re
from typing import Optional

from ..constraints.emotion import validate_emotion_query
from ..constraints.translation import validate_translation_query
from ..constraints.semantic import validate_semantic_query
from ..constraints.multi_entry import validate_multi_entry_pattern


# Non-canonical variable patterns to detect
# Format: (regex_pattern, canonical_replacement, severity)
# severity: "warning" (informational) or "error" (blocks validation)
NON_CANONICAL_PATTERNS = [
    # Italian lemma variants
    (r'\?liitaLemma\b', '?italianLemma', 'warning'),
    (r'\?itaLemma\b', '?italianLemma', 'warning'),

    # Italian word variants
    (r'\?wrIta\b', '?italianWord', 'warning'),
    (r'\?wrIT\b', '?italianWord', 'warning'),
    (r'\?wrsIT\b', '?italianWord', 'warning'),
    (r'\?itaWord\b', '?italianWord', 'warning'),

    # Italian lexical entry variants
    (r'\?leIta\b', '?italianLexEntry', 'warning'),
    (r'\?leITA\b', '?italianLexEntry', 'warning'),

    # Sicilian variants
    (r'\?lemmaSiciliano\b', '?sicilianLemma', 'warning'),
    (r'\?wrSicilian\b', '?sicilianWord', 'warning'),
    (r'\?leSicilian\b', '?sicilianLexEntry', 'warning'),

    # Parmigiano variants
    (r'\?lePar\b', '?parmigianoLexEntry', 'warning'),
    (r'\?leParmigiano\b', '?parmigianoLexEntry', 'warning'),
    (r'\?parWord\b', '?parmigianoWord', 'warning'),

    # Definition variants
    (r'\?_definition\b', '?definition', 'warning'),
    (r'\?senseDefinition\b', '?definition', 'warning'),

    # Count variants
    (r'\?lemmaCount\b', '?count', 'warning'),
    (r'\?wordCount\b', '?count', 'warning'),
    (r'\?verbCount\b', '?count', 'warning'),
]


def check_variable_naming(sparql: str) -> tuple[bool, list[str]]:
    """
    Check for non-canonical variable names in a SPARQL query.

    This validation is informational (warnings) and does not block
    query execution. It helps track naming consistency.

    Args:
        sparql: The SPARQL query to check

    Returns:
        Tuple of (all_canonical, list_of_warnings)
    """
    warnings = []

    for pattern, canonical, severity in NON_CANONICAL_PATTERNS:
        if re.search(pattern, sparql):
            # Extract variable name from pattern
            var_match = re.search(r'\?(\w+)', pattern)
            if var_match:
                found_var = '?' + var_match.group(1)
                msg = f"NAMING {severity.upper()}: Non-canonical variable {found_var} found (use {canonical})"
                warnings.append(msg)

    return (len(warnings) == 0, warnings)


def validate_semantic(
    sparql: str,
    detected_patterns: Optional[list[str]] = None,
    check_naming: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate a SPARQL query against LiITA semantic constraints.

    This checks for common structural errors based on the query type:
    - Emotion queries: Correct graph usage, canonicalForm linking
    - Translation queries: Correct dialect resource patterns
    - Semantic queries: Correct SERVICE usage
    - Multi-entry patterns: Separate variables for different entry types
    - Variable naming: Canonical variable names (optional, warnings only)

    Args:
        sparql: The SPARQL query to validate
        detected_patterns: Optional list of detected patterns to check
        check_naming: Whether to include variable naming checks (default True)

    Returns:
        Tuple of (is_valid, list_of_errors_and_warnings)
    """
    all_errors = []

    # Run emotion validation
    is_valid, errors = validate_emotion_query(sparql)
    all_errors.extend(errors)

    # Run translation validation
    is_valid, errors = validate_translation_query(sparql)
    all_errors.extend(errors)

    # Run semantic validation
    is_valid, errors = validate_semantic_query(sparql)
    all_errors.extend(errors)

    # Run multi-entry validation
    is_valid, errors = validate_multi_entry_pattern(sparql)
    all_errors.extend(errors)

    # Run variable naming check (warnings only, doesn't affect validity)
    if check_naming:
        _, naming_warnings = check_variable_naming(sparql)
        all_errors.extend(naming_warnings)

    # Filter out duplicate errors
    unique_errors = list(dict.fromkeys(all_errors))

    # Determine validity (only CRITICAL/ERROR items, not warnings)
    critical_errors = [e for e in unique_errors if 'WARNING' not in e]

    return (len(critical_errors) == 0, unique_errors)


def get_validation_summary(sparql: str) -> dict:
    """
    Get a detailed validation summary for a query.

    Args:
        sparql: The SPARQL query to validate

    Returns:
        Dictionary with validation results from each module
    """
    summary = {}

    # Emotion validation
    is_valid, errors = validate_emotion_query(sparql)
    summary["emotion"] = {"valid": is_valid, "errors": errors}

    # Translation validation
    is_valid, errors = validate_translation_query(sparql)
    summary["translation"] = {"valid": is_valid, "errors": errors}

    # Semantic validation
    is_valid, errors = validate_semantic_query(sparql)
    summary["semantic"] = {"valid": is_valid, "errors": errors}

    # Multi-entry validation
    is_valid, errors = validate_multi_entry_pattern(sparql)
    summary["multi_entry"] = {"valid": is_valid, "errors": errors}

    # Variable naming check
    is_canonical, warnings = check_variable_naming(sparql)
    summary["variable_naming"] = {"canonical": is_canonical, "warnings": warnings}

    # Overall (excluding naming warnings from validity check)
    all_errors = []
    for key in ["emotion", "translation", "semantic", "multi_entry"]:
        all_errors.extend(summary[key]["errors"])

    summary["overall"] = {
        "valid": len(all_errors) == 0,
        "total_errors": len(all_errors),
        "naming_warnings": len(summary["variable_naming"]["warnings"]),
    }

    return summary

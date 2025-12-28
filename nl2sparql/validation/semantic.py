"""Semantic validation using constraint modules."""

from typing import Optional

from ..constraints.emotion import validate_emotion_query
from ..constraints.translation import validate_translation_query
from ..constraints.semantic import validate_semantic_query
from ..constraints.multi_entry import validate_multi_entry_pattern


def validate_semantic(
    sparql: str,
    detected_patterns: Optional[list[str]] = None,
) -> tuple[bool, list[str]]:
    """
    Validate a SPARQL query against LiITA semantic constraints.

    This checks for common structural errors based on the query type:
    - Emotion queries: Correct graph usage, canonicalForm linking
    - Translation queries: Correct dialect resource patterns
    - Semantic queries: Correct SERVICE usage
    - Multi-entry patterns: Separate variables for different entry types

    Args:
        sparql: The SPARQL query to validate
        detected_patterns: Optional list of detected patterns to check

    Returns:
        Tuple of (is_valid, list_of_errors)
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

    # Filter out duplicate errors
    unique_errors = list(dict.fromkeys(all_errors))

    return (len(unique_errors) == 0, unique_errors)


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

    # Overall
    all_errors = []
    for key in summary:
        all_errors.extend(summary[key]["errors"])

    summary["overall"] = {
        "valid": len(all_errors) == 0,
        "total_errors": len(all_errors),
    }

    return summary

"""SPARQL syntax validation using rdflib."""

from typing import Optional


def validate_syntax(sparql: str) -> tuple[bool, Optional[str]]:
    """
    Validate SPARQL syntax using rdflib's parser.

    Args:
        sparql: The SPARQL query to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    try:
        from rdflib.plugins.sparql import prepareQuery
    except ImportError:
        raise ImportError(
            "rdflib package not installed. Install with: pip install rdflib"
        )

    try:
        # Try to parse the query
        prepareQuery(sparql)
        return (True, None)
    except Exception as e:
        # Extract meaningful error message
        error_msg = str(e)

        # Try to provide more helpful error messages
        if "Expected" in error_msg:
            return (False, f"Syntax error: {error_msg}")
        elif "Unresolved prefix" in error_msg.lower():
            return (False, f"Missing PREFIX declaration: {error_msg}")
        else:
            return (False, f"Parse error: {error_msg}")

"""SPARQL endpoint validation."""

import warnings
from typing import Optional

from ..config import LIITA_ENDPOINT


def validate_endpoint(
    sparql: str,
    endpoint: str = LIITA_ENDPOINT,
    timeout: int = 30,
) -> tuple[bool, Optional[str], Optional[int], Optional[list]]:
    """
    Validate a SPARQL query by executing it against an endpoint.

    Args:
        sparql: The SPARQL query to validate
        endpoint: The SPARQL endpoint URL
        timeout: Query timeout in seconds

    Returns:
        Tuple of (success, error_message, result_count, sample_results)
        - success: Whether the query executed successfully
        - error_message: Error message if failed, None otherwise
        - result_count: Number of results returned
        - sample_results: First few results (up to 5)
    """
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        raise ImportError(
            "SPARQLWrapper package not installed. Install with: pip install SPARQLWrapper"
        )

    try:
        client = SPARQLWrapper(endpoint)
        client.setQuery(sparql)
        client.setReturnFormat(JSON)
        client.setTimeout(timeout)
        # Explicitly request JSON to avoid HTML error pages
        client.addCustomHttpHeader("Accept", "application/sparql-results+json")

        # Suppress SPARQLWrapper warning about unexpected content types
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="unknown response content type",
                category=RuntimeWarning,
                module="SPARQLWrapper"
            )
            results = client.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        result_count = len(bindings)

        # Get sample results (first 5)
        sample = []
        for binding in bindings[:5]:
            row = {}
            for var, value in binding.items():
                row[var] = value.get("value", "")
            sample.append(row)

        return (True, None, result_count, sample)

    except Exception as e:
        error_msg = str(e)

        # Categorize common errors
        if "timeout" in error_msg.lower():
            return (False, f"Query timeout after {timeout}s", None, None)
        elif "400" in error_msg or "Bad Request" in error_msg:
            return (False, f"Query rejected by endpoint: {error_msg}", None, None)
        elif "500" in error_msg or "Internal Server Error" in error_msg:
            return (False, f"Endpoint error: {error_msg}", None, None)
        elif "connection" in error_msg.lower():
            return (False, f"Connection error: {error_msg}", None, None)
        else:
            return (False, f"Execution error: {error_msg}", None, None)

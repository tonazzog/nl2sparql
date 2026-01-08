"""Tool handler implementations for the MCP server.

Each tool function receives arguments from the MCP client and returns
a JSON-serializable result.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..retrieval import HybridRetriever, OntologyRetriever
    from ..generation.synthesizer import NL2SPARQL


def handle_translate(
    arguments: dict[str, Any],
    translator: "NL2SPARQL",
) -> dict[str, Any]:
    """Handle translate tool - full NL to SPARQL translation.

    Args:
        arguments: {"question": str, "language": str (optional)}
        translator: NL2SPARQL instance

    Returns:
        Translation result with sparql, confidence, patterns, etc.
    """
    question = arguments["question"]
    # Language param is for documentation; NL2SPARQL auto-detects

    result = translator.translate(question)

    return {
        "sparql": result.sparql,
        "confidence": result.confidence,
        "detected_patterns": result.detected_patterns,
        "is_valid": result.validation.is_valid if result.validation else None,
        "result_count": result.validation.result_count if result.validation else None,
        "was_fixed": result.was_fixed,
        "fix_attempts": result.fix_attempts,
        "syntax_error": result.validation.syntax_error if result.validation else None,
        "execution_error": result.validation.execution_error if result.validation else None,
        "semantic_errors": result.validation.semantic_errors if result.validation else [],
    }


def handle_infer_patterns(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle infer_patterns tool - detect query patterns from NL.

    Args:
        arguments: {"question": str, "threshold": float (optional)}

    Returns:
        Detected patterns with scores and complexity assessment.
    """
    from ..retrieval.patterns import infer_patterns

    question = arguments["question"]
    threshold = arguments.get("threshold", 0.3)

    patterns = infer_patterns(question, threshold=threshold)

    # Get top patterns above threshold
    top_patterns = [p for p, score in patterns.items() if score >= threshold]
    top_patterns.sort(key=lambda p: patterns[p], reverse=True)

    # Determine complexity
    n_patterns = len(top_patterns)
    if n_patterns <= 1:
        complexity = "simple"
    elif n_patterns == 2:
        complexity = "moderate"
    else:
        complexity = "complex"

    return {
        "patterns": patterns,
        "top_patterns": top_patterns[:5],  # Top 5
        "complexity": complexity,
    }


def handle_retrieve_examples(
    arguments: dict[str, Any],
    retriever: "HybridRetriever",
) -> dict[str, Any]:
    """Handle retrieve_examples tool - get similar query examples.

    Args:
        arguments: {"question": str, "top_k": int (optional), "patterns": dict (optional)}
        retriever: HybridRetriever instance

    Returns:
        List of similar examples with scores.
    """
    question = arguments["question"]
    top_k = arguments.get("top_k", 5)
    patterns = arguments.get("patterns")

    results = retriever.retrieve(
        query=question,
        user_patterns=patterns,
        top_k=top_k,
    )

    examples = []
    for r in results:
        examples.append({
            "id": r.example.id,
            "nl": r.example.nl,
            "sparql": r.example.sparql,
            "patterns": r.example.patterns,
            "score": round(r.score, 4),
            "semantic_score": round(r.semantic_score, 4),
            "bm25_score": round(r.bm25_score, 4),
            "pattern_score": round(r.pattern_score, 4),
        })

    return {"examples": examples}


def handle_search_ontology(
    arguments: dict[str, Any],
    retriever: "OntologyRetriever",
) -> dict[str, Any]:
    """Handle search_ontology tool - semantic search over ontology catalog.

    Args:
        arguments: {"query": str, "top_k": int (optional), "entry_type": str (optional)}
        retriever: OntologyRetriever instance

    Returns:
        Matching ontology entries with descriptions and SPARQL patterns.
    """
    query = arguments["query"]
    top_k = arguments.get("top_k", 10)
    entry_type = arguments.get("entry_type", "all")

    if entry_type == "property":
        results = retriever.retrieve_properties(query, top_k=top_k)
    elif entry_type == "class":
        results = retriever.retrieve_classes(query, top_k=top_k)
    else:
        results = retriever.retrieve(query, top_k=top_k, entry_type="all")

    entries = []
    for r in results:
        entry = r.entry
        entries.append({
            "uri": entry.uri,
            "prefix_local": entry.prefix_local,
            "type": entry.type,
            "label": entry.label,
            "description": entry.short_text,
            "domain": entry.metadata.get("domain", []),
            "range": entry.metadata.get("range", []),
            "inverse": entry.metadata.get("inverse"),
            "sparql_pattern": entry.metadata.get("sparql_pattern", ""),
            "score": round(r.score, 4),
        })

    # Also provide formatted prompt text
    formatted_prompt = retriever.format_for_prompt(results, include_examples=True)

    return {
        "entries": entries,
        "formatted_prompt": formatted_prompt,
    }


def handle_get_constraints(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle get_constraints tool - get domain constraints for patterns.

    Args:
        arguments: {"patterns": list[str]}

    Returns:
        Constraint documentation, prefixes, and system prompt.
    """
    from ..constraints.prompt_builder import get_constraints_for_patterns
    from ..constraints.base import SPARQL_PREFIXES, SYSTEM_PROMPT

    patterns = arguments.get("patterns", [])

    constraints = get_constraints_for_patterns(patterns)

    return {
        "constraints": constraints,
        "prefixes": SPARQL_PREFIXES,
        "system_prompt": SYSTEM_PROMPT,
    }


def handle_validate_sparql(
    arguments: dict[str, Any],
    endpoint: str,
    timeout: int,
) -> dict[str, Any]:
    """Handle validate_sparql tool - comprehensive SPARQL validation.

    Args:
        arguments: {"sparql": str, "patterns": list (optional), check flags}
        endpoint: SPARQL endpoint URL
        timeout: Query timeout in seconds

    Returns:
        Validation results for syntax, semantic, and endpoint checks.
    """
    from ..validation.syntax import validate_syntax
    from ..validation.endpoint import validate_endpoint
    from ..validation.semantic import validate_semantic, get_validation_summary

    sparql = arguments["sparql"]
    patterns = arguments.get("patterns", [])
    check_syntax = arguments.get("check_syntax", True)
    check_semantic = arguments.get("check_semantic", True)
    check_endpoint = arguments.get("check_endpoint", True)
    query_timeout = arguments.get("timeout", timeout)

    result: dict[str, Any] = {"is_valid": True}

    # Syntax check
    if check_syntax:
        valid, error = validate_syntax(sparql)
        result["syntax"] = {"valid": valid, "error": error}
        if not valid:
            result["is_valid"] = False

    # Semantic check
    if check_semantic:
        summary = get_validation_summary(sparql)
        all_errors = []
        for key, val in summary.items():
            if key != "overall" and val.get("errors"):
                all_errors.extend(val["errors"])

        result["semantic"] = {
            "valid": summary["overall"]["valid"],
            "errors": all_errors,
            "summary": summary,
        }
        if not summary["overall"]["valid"]:
            result["is_valid"] = False

    # Endpoint check (only if syntax is valid)
    if check_endpoint and result.get("syntax", {}).get("valid", True):
        success, error, count, sample = validate_endpoint(
            sparql=sparql,
            endpoint=endpoint,
            timeout=query_timeout,
        )
        result["endpoint"] = {
            "success": success,
            "error": error,
            "result_count": count,
            "sample_results": sample,
        }
        if not success:
            result["is_valid"] = False

    return result


def handle_execute_sparql(
    arguments: dict[str, Any],
    endpoint: str,
    default_timeout: int,
) -> dict[str, Any]:
    """Handle execute_sparql tool - execute query against endpoint.

    Args:
        arguments: {"sparql": str, "timeout": int (optional), "limit": int (optional)}
        endpoint: SPARQL endpoint URL
        default_timeout: Default query timeout

    Returns:
        Execution results with result count and data.
    """
    from SPARQLWrapper import SPARQLWrapper, JSON

    sparql = arguments["sparql"]
    timeout = arguments.get("timeout", default_timeout)
    limit = arguments.get("limit", 50)

    # Add LIMIT if not present
    sparql_with_limit = sparql
    if "LIMIT" not in sparql.upper():
        sparql_with_limit = sparql.rstrip().rstrip(";") + f"\nLIMIT {limit}"

    try:
        wrapper = SPARQLWrapper(endpoint)
        wrapper.setQuery(sparql_with_limit)
        wrapper.setReturnFormat(JSON)
        wrapper.setTimeout(timeout)
        wrapper.setMethod("POST")

        response = wrapper.query().convert()

        bindings = response.get("results", {}).get("bindings", [])
        variables = response.get("head", {}).get("vars", [])

        # Convert bindings to simple dicts
        results = []
        for binding in bindings:
            row = {}
            for var in variables:
                if var in binding:
                    row[var] = binding[var].get("value", "")
            results.append(row)

        return {
            "success": True,
            "error": None,
            "result_count": len(results),
            "results": results,
            "variables": variables,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result_count": 0,
            "results": [],
            "variables": [],
        }


def handle_fix_case_sensitivity(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle fix_case_sensitivity tool - auto-fix case-sensitive filters.

    Args:
        arguments: {"sparql": str}

    Returns:
        Fixed SPARQL with change description.
    """
    sparql = arguments["sparql"]
    changes = []

    # Pattern: FILTER(STR(?var) = "string") -> FILTER(REGEX(STR(?var), "^string$", "i"))
    pattern = r'FILTER\s*\(\s*STR\s*\(\s*(\?\w+)\s*\)\s*=\s*"([^"]+)"\s*\)'

    def replace_filter(match):
        var = match.group(1)
        value = match.group(2)
        # Escape regex special characters
        escaped_value = re.escape(value)
        changes.append(f"Converted case-sensitive filter on {var} = \"{value}\" to case-insensitive REGEX")
        return f'FILTER(REGEX(STR({var}), "^{escaped_value}$", "i"))'

    fixed_sparql = re.sub(pattern, replace_filter, sparql, flags=re.IGNORECASE)

    # Also handle: FILTER(?var = "string") for literal comparisons
    pattern2 = r'FILTER\s*\(\s*(\?\w+)\s*=\s*"([^"]+)"\s*\)'

    def replace_filter2(match):
        var = match.group(1)
        value = match.group(2)
        escaped_value = re.escape(value)
        changes.append(f"Converted case-sensitive filter on {var} = \"{value}\" to case-insensitive REGEX")
        return f'FILTER(REGEX(STR({var}), "^{escaped_value}$", "i"))'

    fixed_sparql = re.sub(pattern2, replace_filter2, fixed_sparql, flags=re.IGNORECASE)

    return {
        "sparql": fixed_sparql,
        "was_modified": len(changes) > 0,
        "changes": changes,
    }


def handle_check_variable_reuse(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle check_variable_reuse tool - detect variable reuse bugs.

    Args:
        arguments: {"sparql": str}

    Returns:
        Whether issues were found and descriptions.
    """
    sparql = arguments["sparql"]
    issues = []

    # Find variables used as literal values (in writtenRep)
    literal_vars = set(re.findall(r'writtenRep\s+(\?\w+)', sparql))

    # Find variables used as subjects (before predicates)
    subject_pattern = r'(?:^|\{|\.\s*)\s*(\?\w+)\s+(?:a\s|ontolex:|lexinfo:|skos:|lila:|rdf:|rdfs:)'
    subject_vars = set(re.findall(subject_pattern, sparql, re.MULTILINE))

    # Check for overlap
    reused_vars = literal_vars & subject_vars
    if reused_vars:
        for var in reused_vars:
            issues.append(
                f"Variable {var} is used both as a URI (subject of triple) "
                f"and as a literal (in writtenRep). This will cause the query to fail."
            )

    # Check for variables bound in writtenRep used in FILTER with URI comparison
    for var in literal_vars:
        # Look for patterns like ?var lexinfo:something
        if re.search(rf'{re.escape(var)}\s+(?:lexinfo:|ontolex:|skos:)', sparql):
            if var not in reused_vars:  # Don't double-report
                issues.append(
                    f"Variable {var} appears to be a literal (from writtenRep) "
                    f"but is used as a subject for property access."
                )

    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
    }

"""SPARQL variable normalization for canonical naming.

This module normalizes SPARQL variables to canonical names for consistency
in generation output and evaluation.

Canonical naming conventions:
- Italian: ?italianLemma, ?italianWord, ?italianLexEntry
- Sicilian: ?sicilianLemma, ?sicilianWord, ?sicilianLexEntry
- Parmigiano: ?parmigianoLemma, ?parmigianoWord, ?parmigianoLexEntry
- Semantic: ?definition, ?sense, ?emotionLabel, ?polarityValue
- Aggregation: ?count
"""

import re


# Variable normalization mappings: (pattern, replacement)
# Order matters - more specific patterns should come first
VARIABLE_NORMALIZATIONS = [
    # Italian lemma variants
    (r'\?liitaLemma\b', '?italianLemma'),
    (r'\?itaLemma\b', '?italianLemma'),

    # Italian word variants
    (r'\?wrIta\b', '?italianWord'),
    (r'\?wrIT\b', '?italianWord'),
    (r'\?wrsIT\b', '?italianWord'),
    (r'\?itaWord\b', '?italianWord'),
    (r'\?italianWrittenRep\b', '?italianWord'),

    # Italian lexical entry variants
    (r'\?leIta\b', '?italianLexEntry'),
    (r'\?leITA\b', '?italianLexEntry'),
    (r'\?leItalian\b', '?italianLexEntry'),
    (r'\?italianLexicalEntry\b', '?italianLexEntry'),

    # Sicilian lemma variants
    (r'\?lemmaSiciliano\b', '?sicilianLemma'),

    # Sicilian word variants
    (r'\?wrSicilian\b', '?sicilianWord'),
    (r'\?sicilianWrittenRep\b', '?sicilianWord'),

    # Sicilian lexical entry variants
    (r'\?leSicilian\b', '?sicilianLexEntry'),
    (r'\?sicilianForm\b', '?sicilianLexEntry'),
    (r'\?sicilianLexicalEntry\b', '?sicilianLexEntry'),

    # Parmigiano lemma variants (none common)

    # Parmigiano word variants
    (r'\?parWord\b', '?parmigianoWord'),
    (r'\?parmigianoWrittenRep\b', '?parmigianoWord'),
    (r'\?wrPar\b', '?parmigianoWord'),

    # Parmigiano lexical entry variants
    (r'\?lePar\b', '?parmigianoLexEntry'),
    (r'\?leParmigiano\b', '?parmigianoLexEntry'),
    (r'\?leParLexiconPar\b', '?parmigianoLexEntry'),
    (r'\?parmigianoForm\b', '?parmigianoLexEntry'),
    (r'\?parmigianoLexicalEntry\b', '?parmigianoLexEntry'),

    # Definition variants
    (r'\?_definition\b', '?definition'),
    (r'\?senseDefinition\b', '?definition'),
    (r'\?def\b', '?definition'),

    # Count variants
    (r'\?lemmaCount\b', '?count'),
    (r'\?wordCount\b', '?count'),
    (r'\?senseCount\b', '?count'),
    (r'\?verbCount\b', '?count'),
    (r'\?linkCount\b', '?count'),
    (r'\?countItaWithSicilian\b', '?count'),

    # Polarity variants
    (r'\?avgPolarity\b', '?avgPolarityValue'),

    # Graph variable
    (r'\?g\b(?=\s*\{|\s*\?)(?!\w)', '?graph'),

    # Emotion entry (normalize expression to emotionEntry in emotion context)
    (r'\?expression\b(?=\s+elita:HasEmotion)', '?emotionEntry'),

    # Generic dialect handling
    (r'\?dialectForm\b', '?dialectLexEntry'),
    (r'\?dialEntry\b', '?dialectLexEntry'),
    (r'\?dialectWrittenRep\b', '?dialectWord'),
]


def normalize_variables(sparql: str) -> str:
    """
    Normalize SPARQL variables to canonical names.

    Applies a series of regex replacements to ensure consistent
    variable naming across generated queries.

    Args:
        sparql: Raw SPARQL query

    Returns:
        SPARQL with normalized variable names
    """
    result = sparql

    # Apply standard normalizations
    for pattern, replacement in VARIABLE_NORMALIZATIONS:
        result = re.sub(pattern, replacement, result)

    # Apply context-aware normalizations
    result = _normalize_context_aware(result)

    return result


def _normalize_context_aware(sparql: str) -> str:
    """
    Apply context-aware normalizations.

    Some variables like ?wr are ambiguous and need context to normalize correctly:
    - In Italian-only queries: ?wr → ?italianWord
    - In translation queries: ?wr might be dialect-specific
    """
    sparql_lower = sparql.lower()

    # Detect query context
    has_sicilian = 'siciliano' in sparql_lower or 'sicilian' in sparql_lower
    has_parmigiano = 'parmigiano' in sparql_lower
    has_translation = 'vartrans:translatableas' in sparql_lower
    has_service = 'service' in sparql_lower

    # Context 1: Italian-only query (no dialects, no translation)
    is_italian_only = (
        'liita.it/data' in sparql_lower and
        not has_sicilian and
        not has_parmigiano and
        not has_translation
    )

    if is_italian_only:
        # Check if ?wr appears in SELECT and there's no ?italianWord already
        select_match = re.search(r'SELECT\s+[^{]+', sparql, re.IGNORECASE)
        if select_match:
            select_clause = select_match.group()
            if '?wr' in select_clause and '?italianWord' not in select_clause:
                # Safe to rename all ?wr to ?italianWord
                sparql = re.sub(r'\?wr\b', '?italianWord', sparql)

    # Context 2: Translation queries - normalize based on position
    if has_translation:
        # Pattern: ?sicilianLemma ontolex:writtenRep ?wr → ?sicilianWord
        sparql = re.sub(
            r'(\?sicilianLemma\s+ontolex:writtenRep\s+)\?wr\b',
            r'\1?sicilianWord',
            sparql,
            flags=re.IGNORECASE
        )

        # Pattern: ?parmigianoLemma ontolex:writtenRep ?wr → ?parmigianoWord
        sparql = re.sub(
            r'(\?parmigianoLemma\s+ontolex:writtenRep\s+)\?wr\b',
            r'\1?parmigianoWord',
            sparql,
            flags=re.IGNORECASE
        )

        # Pattern: ?dialectLemma ontolex:writtenRep ?wr → ?dialectWord
        sparql = re.sub(
            r'(\?dialectLemma\s+ontolex:writtenRep\s+)\?wr\b',
            r'\1?dialectWord',
            sparql,
            flags=re.IGNORECASE
        )

    # Context 3: SERVICE queries - keep ?writtenRep or ?wr as generic
    # (No changes needed, these are intentionally generic inside SERVICE)

    return sparql


def detect_non_canonical_variables(sparql: str) -> list[tuple[str, str]]:
    """
    Detect non-canonical variable names in a SPARQL query.

    Returns a list of (found_variable, suggested_canonical) tuples.

    Args:
        sparql: SPARQL query to check

    Returns:
        List of (non_canonical, canonical) variable pairs found
    """
    issues = []

    for pattern, replacement in VARIABLE_NORMALIZATIONS:
        matches = re.findall(pattern, sparql)
        if matches:
            # Extract the variable name from the pattern
            var_match = re.search(r'\?(\w+)', pattern)
            if var_match:
                found_var = '?' + var_match.group(1)
                issues.append((found_var, replacement))

    return issues

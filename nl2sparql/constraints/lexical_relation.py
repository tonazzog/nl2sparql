"""Lexical relation constraints for synonym and antonym queries.

This module provides constraints for queries involving lexical relations
like synonymy and antonymy, which are distinct from hierarchical semantic
relations (hypernym/hyponym/meronym).
"""

import re

LEXICAL_RELATION_MANDATORY_PATTERNS = """
## CRITICAL LEXICAL RELATION CONSTRAINTS

Lexical relations (synonyms, antonyms) differ from hierarchical semantic relations.
They express equivalence or opposition rather than hierarchy.

### DATA SOURCES FOR LEXICAL RELATIONS:

1. **CompL-it SERVICE** - For sense-level relations
   - `lexinfo:approximateSynonym` - Links synonymous senses
   - `lexinfo:antonym` - Links antonymous senses

2. **VarTrans (NO GRAPH)** - For lexical entry-level equivalence
   - `vartrans:translatableAs` - Translation/variant equivalence
   - `vartrans:lexicalRel` - General lexical relation

---

## Pattern 1: Find SYNONYMS via CompL-it (Sense-Level)

**Example: Find synonyms of "felice" (happy)**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the target word
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] .
    FILTER(STR(?wr) = "felice")

    # Get its sense
    ?word ontolex:sense ?sense .

    # Find synonyms via approximateSynonym
    { ?sense lexinfo:approximateSynonym ?synonymSense . }
    UNION
    { ?synonymSense lexinfo:approximateSynonym ?sense . }

    # Get the synonym words
    ?synonymWord ontolex:sense ?synonymSense ;
                 ontolex:canonicalForm [ ontolex:writtenRep ?synonym ] .
}
```

**KEY**: Use UNION for bidirectional matching since synonymy is symmetric.

---

## Pattern 2: Find ANTONYMS via CompL-it (Sense-Level)

**Example: Find antonyms of "bene" (good)**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the target word
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] .
    FILTER(STR(?wr) = "bene")

    # Get its sense
    ?word ontolex:sense ?sense .

    # Find antonyms via lexinfo:antonym
    { ?sense lexinfo:antonym ?antonymSense . }
    UNION
    { ?antonymSense lexinfo:antonym ?sense . }

    # Get the antonym words
    ?antonymWord ontolex:sense ?antonymSense ;
                 ontolex:canonicalForm [ ontolex:writtenRep ?antonym ] .
}
```

**KEY**: Antonymy is also symmetric, use UNION for both directions.

---

## Pattern 3: Synonyms with Definitions

**Example: Find synonyms of "casa" with their definitions**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the target word
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] .
    FILTER(STR(?wr) = "casa")

    ?word ontolex:sense ?sense .

    # Find synonyms
    { ?sense lexinfo:approximateSynonym ?synonymSense . }
    UNION
    { ?synonymSense lexinfo:approximateSynonym ?sense . }

    # Get synonym word and definition
    ?synonymWord ontolex:sense ?synonymSense ;
                 ontolex:canonicalForm [ ontolex:writtenRep ?synonym ] .

    OPTIONAL { ?synonymSense skos:definition ?definition . }
}
```

---

## Pattern 4: Linking Synonyms/Antonyms to LiITA

**Example: Find synonyms with their POS from LiITA**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] .
    FILTER(STR(?wr) = "veloce")

    ?word ontolex:sense ?sense .

    { ?sense lexinfo:approximateSynonym ?synonymSense . }
    UNION
    { ?synonymSense lexinfo:approximateSynonym ?sense . }

    ?synonymWord ontolex:sense ?synonymSense .
}

# Link to LiITA using shared URI
?synonymWord ontolex:canonicalForm ?liitaLemma .

GRAPH <http://liita.it/data> {
    ?liitaLemma a lila:Lemma ;
               lila:hasPOS ?pos ;
               ontolex:writtenRep ?synonymWr .
}
```

---

## DIFFERENCES FROM SEMANTIC_RELATION PATTERNS

| Aspect | Semantic Relations | Lexical Relations |
|--------|-------------------|-------------------|
| Type | Hierarchical (hypernym/hyponym) | Non-hierarchical (synonym/antonym) |
| Property | `lexinfo:hypernym`, `lexinfo:hyponym` | `lexinfo:approximateSynonym`, `lexinfo:antonym` |
| Symmetry | Asymmetric (directed) | Symmetric (use UNION) |
| Meaning | More/less general, part/whole | Same/opposite meaning |

---

## COMMON MISTAKES TO AVOID

### Mistake 1: Forgetting UNION for symmetric relations
```sparql
# WRONG - Only finds synonyms in one direction
?sense lexinfo:approximateSynonym ?synonymSense .

# CORRECT - Finds synonyms in both directions
{ ?sense lexinfo:approximateSynonym ?synonymSense . }
UNION
{ ?synonymSense lexinfo:approximateSynonym ?sense . }
```

### Mistake 2: Confusing with translation equivalence
```sparql
# WRONG - vartrans:translatableAs is for translations, not synonyms
?entry vartrans:translatableAs ?synonymEntry .

# CORRECT - Use lexinfo:approximateSynonym for same-language synonyms
?sense lexinfo:approximateSynonym ?synonymSense .
```

### Mistake 3: Using wrong data source
```sparql
# WRONG - Synonyms are NOT in LiITA graphs
GRAPH <http://liita.it/data> {
    ?sense lexinfo:approximateSynonym ?synonymSense .
}

# CORRECT - Use SERVICE for CompL-it
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?sense lexinfo:approximateSynonym ?synonymSense .
}
```

---

## VALIDATION CHECKLIST

For SYNONYM/ANTONYM queries:
- [ ] Using SERVICE block for CompL-it
- [ ] Using `lexinfo:approximateSynonym` or `lexinfo:antonym` (NOT hypernym/hyponym)
- [ ] Using UNION for bidirectional matching
- [ ] Variable is `?word` (not `?lemma`) in SERVICE
- [ ] Linking to LiITA via shared URI OUTSIDE SERVICE
- [ ] NO lila: properties inside SERVICE
"""


def validate_lexical_relation_query(sparql_query: str) -> tuple[bool, list[str]]:
    """
    Validate lexical relation query structure.

    Args:
        sparql_query: The SPARQL query to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    query_upper = sparql_query.upper()

    # Check if it's a lexical relation query
    has_synonym = "LEXINFO:APPROXIMATESYNONYM" in query_upper
    has_antonym = "LEXINFO:ANTONYM" in query_upper

    is_lexical_relation = has_synonym or has_antonym

    if not is_lexical_relation:
        return (True, [])  # Not a lexical relation query, skip validation

    # Check 1: Should use SERVICE
    if "SERVICE" not in query_upper:
        errors.append(
            "LEXICAL RELATION ERROR: Queries with synonyms/antonyms must use "
            "SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/>"
        )

    # Check 2: Should use UNION for bidirectional matching
    if "UNION" not in query_upper:
        errors.append(
            "LEXICAL RELATION WARNING: Synonymy and antonymy are symmetric relations. "
            "Consider using UNION to match in both directions: "
            "{ ?sense lexinfo:approximateSynonym ?other } UNION { ?other approximateSynonym ?sense }"
        )

    # Check 3: lexinfo properties should not be in GRAPH blocks
    if "GRAPH" in query_upper:
        # Check if approximateSynonym or lexinfo:antonym appears before SERVICE
        service_pos = query_upper.find("SERVICE")
        synonym_pos = query_upper.find("LEXINFO:APPROXIMATESYNONYM")
        antonym_pos = query_upper.find("LEXINFO:ANTONYM")

        if synonym_pos > 0 and service_pos > 0 and synonym_pos < service_pos:
            errors.append(
                "LEXICAL RELATION ERROR: approximateSynonym should be inside SERVICE block"
            )
        if antonym_pos > 0 and service_pos > 0 and antonym_pos < service_pos:
            errors.append(
                "LEXICAL RELATION ERROR: lexinfo:antonym should be inside SERVICE block"
            )

    # Check 4: lila: properties should not be in SERVICE
    if "SERVICE" in query_upper:
        service_match = re.search(
            r"SERVICE\s*<[^>]+>\s*\{([^}]+)\}", query_upper, re.DOTALL
        )
        if service_match:
            service_content = service_match.group(1)
            if "LILA:" in service_content:
                errors.append(
                    "LEXICAL RELATION ERROR: lila: properties do not exist in "
                    "CompL-it SERVICE. Use them outside SERVICE in GRAPH blocks."
                )

    return (len(errors) == 0, errors)

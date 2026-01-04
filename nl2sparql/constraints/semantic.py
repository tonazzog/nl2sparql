"""Semantic query constraints for LiIta SPARQL synthesis."""

import re

SEMANTIC_MANDATORY_PATTERNS = """
## CRITICAL SEMANTIC QUERY CONSTRAINTS

Semantic relations and definitions are accessed via EXTERNAL SERVICE, NOT graphs!

### CORE PRINCIPLE: All Semantic Data is in CompL-it SERVICE

**CORRECT PATTERN:**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] ;
          ontolex:sense [ skos:definition ?definition ] .
    FILTER(STR(?wr) = "casa")
}
```

**WRONG PATTERN:**
```sparql
# WRONG - Senses are NOT in LiIta graphs!
GRAPH <http://liita.it/data> {
    ?word ontolex:sense ?sense .  # WRONG!
}
```

---

## CRITICAL: SEMANTIC RELATION DIRECTIONS

The lexinfo vocabulary uses COUNTERINTUITIVE property directions!
The property name indicates what the OBJECT is, not the SUBJECT.

### lexinfo:hypernym - Points FROM general TO specific
- `?sensaA lexinfo:hypernym ?senseB` means "A has B among its hyponyms" = "B is more specific than A"
- To find HYPONYMS (more specific) of "veicolo": `?veicoloSense lexinfo:hypernym ?hyponymSense`
- To find HYPERNYMS (more general) of "cane": `?caneSense lexinfo:hyponym ?hypernymSense`

### lexinfo:hyponym - Points FROM specific TO general
- `?senseA lexinfo:hyponym ?senseB` means "A has B as its hypernym" = "B is more general than A"
- This is how you find BROADER categories!

### lexinfo:partMeronym - Points FROM part TO whole
- `?partSense lexinfo:partMeronym ?wholeSense` means "partSense is a PART OF wholeSense"
- To find PARTS of "corpo": `?partSense lexinfo:partMeronym ?corpoSense`
- To find WHOLES containing "braccio": `?braccioSense lexinfo:partMeronym ?wholeSense`

---

## Pattern 1: Find HYPONYMS (more specific terms)

**Example: Find hyponyms of "veicolo" (vehicle) - like "automobile", "bicicletta"**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the target word
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wrIta ] .
    FILTER(STR(?wrIta) = "veicolo")

    # Get its sense
    ?word ontolex:sense ?sense .

    # Find hyponyms: use lexinfo:hypernym FROM the general term
    ?sense lexinfo:hypernym ?hyponymSense .

    # Get the hyponym words
    ?hyponymWord ontolex:sense ?hyponymSense ;
                 ontolex:canonicalForm [ ontolex:writtenRep ?result ] .
}
```

---

## Pattern 2: Find HYPERNYMS (more general terms)

**Example: Find hypernyms of "cane" (dog) - like "animale", "mammifero"**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the target word
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wrIta ] .
    FILTER(STR(?wrIta) = "cane")

    # Get its sense
    ?word ontolex:sense ?sense .

    # Find hypernyms: use lexinfo:hyponym FROM the specific term
    ?sense lexinfo:hyponym ?hypernymSense .

    # Get the hypernym words
    ?hypernymWord ontolex:sense ?hypernymSense ;
                  ontolex:canonicalForm [ ontolex:writtenRep ?result ] .
}
```

---

## Pattern 3: Find PARTS (meronyms) of something

**Example: Find parts of "corpo" (body) - like "braccio", "mano", "testa"**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the WHOLE (corpo)
    ?wholeWord ontolex:canonicalForm [ ontolex:writtenRep ?wholeWr ] .
    FILTER(STR(?wholeWr) = "corpo")

    # Get the whole's sense
    ?wholeWord ontolex:sense ?wholeSense .

    # Find parts: partMeronym points FROM part TO whole
    ?partSense lexinfo:partMeronym ?wholeSense .

    # Get the part words
    ?partWord ontolex:sense ?partSense ;
              ontolex:canonicalForm [ ontolex:writtenRep ?result ] .
}
```

---

## Pattern 4: Find WHOLES containing something

**Example: What is "braccio" (arm) a part of? - like "corpo"**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the PART (braccio)
    ?partWord ontolex:canonicalForm [ ontolex:writtenRep ?partWr ] .
    FILTER(STR(?partWr) = "braccio")

    # Get the part's sense
    ?partWord ontolex:sense ?partSense .

    # Find wholes: partMeronym points FROM part TO whole
    ?partSense lexinfo:partMeronym ?wholeSense .

    # Get the whole words
    ?wholeWord ontolex:sense ?wholeSense ;
               ontolex:canonicalForm [ ontolex:writtenRep ?result ] .
}
```

---

## Pattern 5: Basic Definition Lookup
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] .
    FILTER(STR(?wr) = "word_to_lookup")

    ?word ontolex:sense [ skos:definition ?definition ] .
}
```

---

## Pattern 6: Linking CompL-it to LiIta (CRITICAL!)

**CompL-it and LiIta share URIs for lexical entries!**

Variables bound in the SERVICE block can be used directly outside to access LiIta data.

**Example: Find meronyms of "giorno" with their Parmigiano translations**
```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find the target word
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?lemma ] ;
          ontolex:sense ?sense .
    FILTER(STR(?lemma) = "giorno")

    # Find meronyms (parts of "giorno" like "mattina", "sera")
    ?senseMeronym lexinfo:partMeronym ?sense .

    # Get the word that has this meronym sense
    ?wordMeronym ontolex:sense ?senseMeronym .
}

# OUTSIDE SERVICE: Use ?wordMeronym directly to access LiIta!
?wordMeronym ontolex:canonicalForm ?liitaLemma .

# Continue with LiIta patterns (e.g., get Parmigiano translation)
?translationEntry ontolex:canonicalForm ?liitaLemma ;
                  ^lime:entry <http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon> .
?translationEntry vartrans:translatableAs ?dialectEntry .
?dialectEntry ontolex:canonicalForm ?dialectLemma .
?dialectLemma ontolex:writtenRep ?dialectWord .
```

**KEY INSIGHT**: The `?wordMeronym` URI from CompL-it is the SAME URI used in LiIta, so you can:
1. Bind it inside SERVICE
2. Use it outside SERVICE to access LiIta properties
3. NO need for string matching with FILTER(STR(?x) = STR(?y))

**CRITICAL**: The linking statement `?word ontolex:canonicalForm ?lemma` must be OUTSIDE the SERVICE block!

```sparql
# WRONG - linking inside SERVICE
SERVICE <...> {
    ?word ontolex:sense ?sense .
    ?word ontolex:canonicalForm ?liitaLemma .  # WRONG: move this outside!
}

# CORRECT - linking outside SERVICE
SERVICE <...> {
    ?word ontolex:sense ?sense .
}
?word ontolex:canonicalForm ?liitaLemma .  # CORRECT: outside SERVICE
```

**CRITICAL**: When using shared URI linking, NO additional FILTER is needed!

```sparql
# WRONG - unnecessary filter referencing outside variable
SERVICE <...> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] .
    FILTER(STR(?wr) = ?italianWord)  # WRONG: ?italianWord is outside, and linking is via URI!
}
?word ontolex:canonicalForm ?lemma .

# CORRECT - linking via URI is sufficient, no filter needed
SERVICE <...> {
    ?word ontolex:sense ?sense .
}
?word ontolex:canonicalForm ?lemma .  # The URI match handles the linking!
```

**CRITICAL**: Always use FILTER(STR()) for string matching, never direct literals!

```sparql
# WRONG - direct literal may fail due to language tags
?word ontolex:canonicalForm [ ontolex:writtenRep "sentimento" ] .

# CORRECT - use variable + FILTER with STR()
?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] .
FILTER(STR(?wr) = "sentimento")
```

---

## Pattern 7: Filtering + Definition (CRITICAL ORDER!)

When you need to filter by text pattern (REGEX) AND get definitions, you MUST:
1. Start with SERVICE and apply the filter INSIDE
2. Use the shared URI to link to LiITA OUTSIDE SERVICE

**WRONG - Variables from outside cannot be used in FILTER inside SERVICE:**
```sparql
# WRONG - This will fail!
GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ;
           lila:hasPOS lila:noun ;
           ontolex:writtenRep ?wr .
    FILTER(REGEX(STR(?wr), "zione$"))
}

SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wrComplit ] ;
          ontolex:sense [ skos:definition ?definition ] .
    FILTER(STR(?wrComplit) = STR(?wr))  # ERROR: ?wr is not accessible here!
}
```

**CORRECT - Filter inside SERVICE, then link to LiITA:**
```sparql
# First: Query CompL-it with filter INSIDE the SERVICE
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:sense ?sense .
    ?sense skos:definition ?definition .
    ?word ontolex:canonicalForm ?lemma .
    ?lemma ontolex:writtenRep ?writtenRep .
    FILTER(REGEX(?writtenRep, "zione$", "i"))
}

# Then: Use shared URI to access LiITA
?word ontolex:canonicalForm ?liitaLemma .
GRAPH <http://liita.it/data> {
    ?liitaLemma ontolex:writtenRep ?italianWord ;
                lila:hasPOS lila:noun .
}
```

**KEY RULES:**
1. FILTER with REGEX must be INSIDE the SERVICE where the data lives
2. Variables bound outside SERVICE CANNOT be used in FILTER inside SERVICE
3. Use the shared URI (?word) to link SERVICE results to LiITA
4. LiITA-specific filters (like lila:hasPOS) go in GRAPH block AFTER SERVICE

---

## Pattern 8: Starting from LiITA Lemma (CRITICAL!)

When you START with a LiITA lemma and need definitions from CompL-it, use the SAME VARIABLE NAME for writtenRep to create a natural join:

**WRONG - Variables from outside cannot be used in FILTER inside SERVICE:**
```sparql
# WRONG - This will cause error: "Variable 'writtenRep' is used but not assigned"
GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ;
           lila:hasPOS lila:noun ;
           ontolex:writtenRep ?writtenRep .
    FILTER(REGEX(STR(?writtenRep), "etto$"))
}

SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] ;
          ontolex:sense [ skos:definition ?definition ] .
    FILTER(STR(?wr) = STR(?writtenRep))  # ERROR: ?writtenRep not accessible here!
}
```

**CORRECT - Use the SAME variable name to create a natural join:**
```sparql
# CORRECT - Same variable ?writtenRep creates automatic join
GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ;
           lila:hasPOS lila:noun ;
           ontolex:writtenRep ?writtenRep .
    FILTER(REGEX(STR(?writtenRep), "etto$"))
}

SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?writtenRep ] ;
          ontolex:sense [ skos:definition ?definition ] .
    # NO FILTER needed! The same variable name creates the join automatically
}
```

**KEY INSIGHT**: SPARQL federation allows the same variable to be used in both the main query and SERVICE block. This creates an implicit join on that variable's value - much more efficient than string comparison!

**When to use this pattern:**
- Starting from LiITA lemmas (GRAPH block) and need CompL-it data (definitions, senses)
- The join is on the written form of the word (ontolex:writtenRep)
- Works for both exact matches and regex filters

---

## Pattern 9: When to Use Which Join Strategy

| Scenario | Strategy |
|----------|----------|
| Start from LiITA lemma, need definition | Use SAME variable for writtenRep (Pattern 8) |
| Need definition + filter by pattern | Start with SERVICE, filter inside, link via URI |
| Need definition for specific word | Start with SERVICE, filter by exact word |
| Need semantic relations + LiITA data | Start with SERVICE, link via URI |
| Need LiITA-only data (no definition) | Just use GRAPH, no SERVICE needed |

**The general rule**:
- If starting from LiITA: Use same variable name for writtenRep to join with SERVICE
- If starting from CompL-it: Filter inside SERVICE, then link via shared URI

---

## QUICK REFERENCE: Semantic Relation Directions

| I want to find... | Starting from X | Use this pattern |
|-------------------|-----------------|------------------|
| Hyponyms (more specific) | X = general term | `?xSense lexinfo:hypernym ?resultSense` |
| Hypernyms (more general) | X = specific term | `?xSense lexinfo:hyponym ?resultSense` |
| Parts (meronyms) | X = the whole | `?resultSense lexinfo:partMeronym ?xSense` |
| Wholes (holonyms) | X = the part | `?xSense lexinfo:partMeronym ?resultSense` |

**Memory trick:**
- `hypernym` property points DOWNWARD (to more specific)
- `hyponym` property points UPWARD (to more general)
- `partMeronym` property points FROM part TO whole

---

## COMMON MISTAKES TO AVOID

### Mistake 1: Wrong direction for hypernym/hyponym
```sparql
# WRONG - This finds hypernyms, not hyponyms!
?xSense lexinfo:hyponym ?resultSense .  # This gives MORE GENERAL terms

# CORRECT - To find hyponyms (more specific):
?xSense lexinfo:hypernym ?resultSense .
```

### Mistake 2: Wrong direction for meronyms
```sparql
# WRONG - This finds wholes, not parts!
?xSense lexinfo:partMeronym ?resultSense .  # This gives things X is part OF

# CORRECT - To find parts of X:
?resultSense lexinfo:partMeronym ?xSense .  # Things that are parts of X
```

### Mistake 3: Using GRAPH for senses
```sparql
# WRONG - Senses are in SERVICE, not GRAPH
GRAPH <http://liita.it/data> {
    ?word ontolex:sense ?sense .
}

# CORRECT
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:sense ?sense .
}
```

---

## VALIDATION CHECKLIST

- [ ] All sense/definition queries use SERVICE block
- [ ] SERVICE URL: `<https://klab.ilc.cnr.it/graphdb-compl-it/>`
- [ ] For hyponyms: using `lexinfo:hypernym` (counterintuitive!)
- [ ] For hypernyms: using `lexinfo:hyponym` (counterintuitive!)
- [ ] For parts: `?partSense lexinfo:partMeronym ?wholeSense`
- [ ] Variable is `?word` (not `?lemma`) in SERVICE
- [ ] Use STR() in FILTER for string matching
- [ ] NO lila: properties inside SERVICE
- [ ] NO GRAPH blocks inside SERVICE
"""


# CompL-it SERVICE endpoint
SEMANTIC_SERVICE_ENDPOINT = "https://klab.ilc.cnr.it/graphdb-compl-it/"


def validate_semantic_query(sparql_query: str) -> tuple[bool, list[str]]:
    """
    Validate semantic query structure.

    Args:
        sparql_query: The SPARQL query to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    query_upper = sparql_query.upper()

    # Check if it's a semantic query
    has_sense = "ONTOLEX:SENSE" in query_upper
    has_definition = "SKOS:DEFINITION" in query_upper
    has_semantic_rel = any(
        rel in query_upper
        for rel in [
            "LEXINFO:HYPERNYM",
            "LEXINFO:HYPONYM",
            "LEXINFO:PARTMERONYM",
            "LEXINFO:HOLONYM",
        ]
    )

    is_semantic = has_sense or has_definition or has_semantic_rel

    if not is_semantic:
        return (True, [])  # Not a semantic query, skip validation

    # Check 1: Should use SERVICE
    if "SERVICE" not in query_upper:
        errors.append(
            "SEMANTIC ERROR: Queries with senses/definitions must use "
            "SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/>"
        )

    # Check 2: Senses should not be in GRAPH blocks
    if "ONTOLEX:SENSE" in query_upper:
        if "GRAPH" in query_upper:
            graph_pos = query_upper.find("GRAPH")
            service_pos = query_upper.find("SERVICE")
            sense_pos = query_upper.find("ONTOLEX:SENSE")

            if service_pos > 0 and sense_pos > 0:
                if sense_pos < service_pos:
                    errors.append(
                        "SEMANTIC ERROR: ontolex:sense should be inside SERVICE block, "
                        "not in GRAPH blocks"
                    )

    # Check 3: lila: properties should not be in SERVICE
    if "SERVICE" in query_upper:
        service_match = re.search(
            r"SERVICE\s*<[^>]+>\s*\{([^}]+)\}", query_upper, re.DOTALL
        )
        if service_match:
            service_content = service_match.group(1)
            if "LILA:" in service_content:
                errors.append(
                    "SEMANTIC ERROR: lila: properties (like lila:Lemma, lila:hasPOS) "
                    "do not exist in CompL-it SERVICE. Use OntoLex vocabulary instead."
                )

    # Check 4: Recommend STR() in FILTER
    if "FILTER" in query_upper and "=" in query_upper:
        if "STR(" not in query_upper:
            errors.append(
                "WARNING: Consider using STR() in FILTER for string comparisons, "
                'e.g., FILTER(STR(?wr) = "word")'
            )

    return (len(errors) == 0, errors)


# Semantic relations reference - CORRECTED DIRECTIONS
SEMANTIC_RELATIONS = {
    "hypernym": {
        "property": "lexinfo:hypernym",
        "use_to_find": "HYPONYMS (more specific terms)",
        "pattern": "?generalSense lexinfo:hypernym ?specificSense",
        "example": "To find hyponyms of 'veicolo': ?veicoloSense lexinfo:hypernym ?resultSense",
        "note": "Counterintuitive! The hypernym property points TO the hyponym.",
    },
    "hyponym": {
        "property": "lexinfo:hyponym",
        "use_to_find": "HYPERNYMS (more general terms)",
        "pattern": "?specificSense lexinfo:hyponym ?generalSense",
        "example": "To find hypernyms of 'cane': ?caneSense lexinfo:hyponym ?resultSense",
        "note": "Counterintuitive! The hyponym property points TO the hypernym.",
    },
    "partMeronym": {
        "property": "lexinfo:partMeronym",
        "use_to_find": "PARTS or WHOLES depending on direction",
        "pattern_find_parts": "?partSense lexinfo:partMeronym ?wholeSense",
        "pattern_find_wholes": "?partSense lexinfo:partMeronym ?wholeSense",
        "example_parts": "To find parts of 'corpo': ?resultSense lexinfo:partMeronym ?corpoSense",
        "example_wholes": "To find wholes containing 'braccio': ?braccioSense lexinfo:partMeronym ?resultSense",
        "note": "The part points TO the whole it belongs to.",
    },
}

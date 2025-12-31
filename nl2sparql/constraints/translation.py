"""Translation query constraints for LiIta SPARQL synthesis."""

import re

TRANSLATION_MANDATORY_PATTERNS = """
## CRITICAL TRANSLATION QUERY CONSTRAINTS

Translation queries have specific structural requirements based on OntoLex-Lemon vocabulary.

### CORE PRINCIPLES:

1. **Translations link LEXICAL ENTRIES, not lemmas directly**
2. **Direction is ALWAYS Italian → Dialect** (never dialect → Italian)
3. **No GRAPH clauses needed** for basic translation queries

---

### TRANSLATION DIRECTION (CRITICAL!):

**CORRECT - Italian entry translates TO dialect entry:**
```sparql
?italianLexEntry vartrans:translatableAs ?dialectLexEntry .
```

**WRONG - This direction does NOT exist in the data:**
```sparql
?dialectLexEntry vartrans:translatableAs ?italianLexEntry .  # WRONG!
```

---

### Pattern 1: Find DIALECT words and their ITALIAN translations

Use this when: "Find Parmigiano/Sicilian words that... and show Italian translation"

```sparql
# Step 1: Dialect lemma (no GRAPH needed!)
?dialectLemma a lila:Lemma ;
              ontolex:writtenRep ?dialectWr .

# Step 2: Dialect lexical entry
?dialectLexEntry ontolex:canonicalForm ?dialectLemma .

# Step 3: Italian entry that translates TO this dialect entry
?italianLexEntry vartrans:translatableAs ?dialectLexEntry ;
                 ontolex:canonicalForm ?italianLemma .

# Step 4: Italian word
?italianLemma ontolex:writtenRep ?italianWr .
```

**Real example - Parmigiano verbs ending with "or" and Italian translation:**
```sparql
SELECT ?lemma ?wr ?liitaLemma ?wrIT
WHERE {
  ?lemma a lila:Lemma ;
         ontolex:writtenRep ?wr ;
         lila:hasPOS lila:verb .
  ?le ontolex:canonicalForm ?lemma .
  ?leITA vartrans:translatableAs ?le ;
         ontolex:canonicalForm ?liitaLemma .
  ?liitaLemma ontolex:writtenRep ?wrIT .
  FILTER regex(str(?wr), "or$") .
}
```

---

### Pattern 2: Find ITALIAN words and their DIALECT translations

Use this when: "Find Italian words that... and show Parmigiano/Sicilian translation"

```sparql
# Step 1: Italian lemma
?italianLemma a lila:Lemma ;
              ontolex:writtenRep ?italianWr .

# Step 2: Italian lexical entry
?italianLexEntry ontolex:canonicalForm ?italianLemma .

# Step 3: Translation link (Italian → Dialect)
?italianLexEntry vartrans:translatableAs ?dialectLexEntry .

# Step 4: Dialect lemma and word
?dialectLexEntry ontolex:canonicalForm ?dialectLemma .
?dialectLemma ontolex:writtenRep ?dialectWr .
```

**Real example - Italian "donna" and Parmigiano translations:**
```sparql
SELECT ?wrsIT ?wrs
WHERE {
  ?lemma a lila:Lemma ;
         ontolex:writtenRep ?wr .
  ?le ontolex:canonicalForm ?lemma .
  ?leITA vartrans:translatableAs ?le ;
         ontolex:canonicalForm ?liitaLemma .
  ?liitaLemma ontolex:writtenRep ?wrIT .
  FILTER regex(str(?wrIT), "^donna$")
}
```

---

### Dialect Identification (CRITICAL - Different per Dialect!):

**Sicilian Dialect:**
```sparql
# Use dcterms:isPartOf with LemmaBank URI
?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> .
```

**Parmigiano Dialect:**
```sparql
# Use INVERSE lime:entry with Lexicon URI
?parmigianoLexEntry ^lime:entry <http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon> .
```

**Italian (Standard):**
```sparql
?italianLemma dcterms:isPartOf <http://liita.it/data/id/lemma/LemmaBank> .
```

**CRITICAL:**
- Sicilian -> `dcterms:isPartOf` on LEMMA + LemmaBank URI
- Parmigiano -> `^lime:entry` on LEXICAL ENTRY + Lexicon URI
- Don't mix these patterns!

---

### DIALECT-SPECIFIC GRAPHS (CRITICAL!):

Each dialect has its OWN NAMED GRAPH with its own lemmas:

**Main Italian:** `GRAPH <http://liita.it/data>`
**Parmigiano:** `GRAPH <http://liita.it/data/id/DialettoParmigiano>`
**Sicilian:** `GRAPH <http://liita.it/data/id/DialettoSiciliano/>`

**IMPORTANT DISTINCTION:**

1. **"Italian words that translate to Parmigiano"** = query main graph + join with Parmigiano lexicon
2. **"Parmigiano words/lemmas"** = query Parmigiano graph DIRECTLY

**Example: POS distribution IN Parmigiano (Parmigiano lemmas):**
```sparql
# CORRECT - Query Parmigiano graph directly
SELECT ?pos (COUNT(DISTINCT ?lemma) AS ?count)
WHERE {
  GRAPH <http://liita.it/data/id/DialettoParmigiano> {
    ?lemma a lila:Lemma ;
           lila:hasPOS ?pos .
  }
}
GROUP BY ?pos
```

**WRONG approach (gives Italian lemmas with Parmigiano translations):**
```sparql
# WRONG - This queries Italian lemmas from main graph!
SELECT ?pos (COUNT(DISTINCT ?lemma) AS ?count)
WHERE {
  GRAPH <http://liita.it/data> {        # <-- WRONG GRAPH for Parmigiano lemmas!
    ?lemma a lila:Lemma ;
           lila:hasPOS ?pos .
  }
  ?lexEntry ontolex:canonicalForm ?lemma ;
            ^lime:entry <...Parmigiano/Lexicon> .
}
GROUP BY ?pos
```

**Alternative - Get POS from Parmigiano forms:**
```sparql
SELECT ?pos (COUNT(?parmigianoLexEntry) AS ?count)
WHERE {
  ?parmigianoLexEntry ^lime:entry <http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon> .
  ?parmigianoLexEntry ontolex:canonicalForm ?parmigianoForm .
  ?parmigianoForm lila:hasPOS ?pos .
}
GROUP BY ?pos
```

---

### Translation + Sentiment/Emotion:
```sparql
# Italian lemma as join point
?italianLemma a lila:Lemma ;
              ontolex:writtenRep ?italianWord .

# Sentiment on one lexical entry
?sentimentLexEntry ontolex:canonicalForm ?italianLemma ;
                   marl:hasPolarity ?polarity ;
                   marl:hasPolarityValue ?polarityValue .

# Translation on possibly different lexical entry!
?translationLexEntry ontolex:canonicalForm ?italianLemma ;
                     vartrans:translatableAs ?dialectLexEntry .

# Dialect lemma
?dialectLexEntry ontolex:canonicalForm ?dialectLemma .
?dialectLemma dcterms:isPartOf ?dialectLemmaBank ;
              ontolex:writtenRep ?dialectWord .
```

**CRITICAL INSIGHT:**
- Multiple lexical entries can share the same canonicalForm (lemma)
- One lexEntry might have sentiment, another might have translation
- Use the LEMMA as the join point between them

---

### VALIDATION CHECKLIST:

- [ ] **Translation direction**: Italian → Dialect (NEVER dialect → Italian)
  - CORRECT: `?italianEntry vartrans:translatableAs ?dialectEntry`
  - WRONG: `?dialectEntry vartrans:translatableAs ?italianEntry`
- [ ] Translation uses `vartrans:translatableAs` on lexical entries, NOT lemmas
- [ ] `ontolex:canonicalForm` connects lexical entries to lemmas
- [ ] `ontolex:writtenRep` accessed on lemmas
- [ ] Morphological properties (POS, gender) queried on lemmas
- [ ] **No GRAPH clauses needed** for basic translation queries
- [ ] For dialect-specific aggregations (e.g., POS distribution), use dialect-specific GRAPH
"""


def validate_translation_query(sparql_query: str) -> tuple[bool, list[str]]:
    """
    Validate translation query structure.

    Args:
        sparql_query: The SPARQL query to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    query_upper = sparql_query.upper()
    query_normalized = query_upper.replace(" ", "").replace("\n", "")

    # Only validate if it's a translation query
    if "VARTRANS:TRANSLATABLEAS" not in query_upper:
        return (True, [])  # Not a translation query, skip validation

    # Check 1: translatableAs should be on variables, not lemmas
    if (
        "?LEMMAVARTRANS:TRANSLATABLEAS" in query_normalized
        or "?LEMMA.VARTRANS:TRANSLATABLEAS" in query_normalized
    ):
        errors.append(
            "CRITICAL: vartrans:translatableAs links lexical entries, NOT lemmas directly"
        )

    # Check 2: Translation direction - dialect entry should NOT be subject of translatableAs
    # Look for patterns like ?parmigianoEntry vartrans:translatableAs or ?sicilianEntry vartrans:translatableAs
    wrong_direction_patterns = [
        r"\?parmigiano\w*\s+vartrans:translatableAs",
        r"\?sicilian\w*\s+vartrans:translatableAs",
        r"\?dialect\w*\s+vartrans:translatableAs",
    ]
    for pattern in wrong_direction_patterns:
        if re.search(pattern, sparql_query, re.IGNORECASE):
            errors.append(
                "CRITICAL: Translation direction is WRONG! "
                "Direction must be Italian → Dialect, not Dialect → Italian. "
                "Use: ?italianEntry vartrans:translatableAs ?dialectEntry"
            )
            break

    # Check 3: Should use canonicalForm to connect entries to lemmas
    if "VARTRANS:TRANSLATABLEAS" in query_upper:
        if "ONTOLEX:CANONICALFORM" not in query_upper:
            errors.append(
                "WARNING: Translation queries typically need ontolex:canonicalForm to access lemmas"
            )

    # Check 3: WrittenRep should be on lemmas, not directly on lexEntries
    if (
        "?LE.ONTOLEX:WRITTENREP" in query_normalized
        or "?LEXENTRYONTOLEX:WRITTENREP" in query_normalized
        or "?LEXICALENTRYONTOLEX:WRITTENREP" in query_normalized
    ):
        errors.append(
            "WARNING: ontolex:writtenRep is on lemmas; "
            "use property path: ?lexEntry ontolex:canonicalForm/ontolex:writtenRep ?word"
        )

    # Check 4: Dialect resource usage
    if "DIALETTOSICILIANO" in query_upper:
        if "^LIME:ENTRY" in query_upper and "SICILIAN" in query_upper:
            errors.append(
                "CRITICAL: Sicilian should use dcterms:isPartOf with LemmaBank, not ^lime:entry"
            )
        if "LEXICON" in query_upper and "SICILIAN" in query_upper:
            errors.append(
                "WARNING: Sicilian typically uses LemmaBank URI, not Lexicon URI"
            )

    if "PARMIGIANO" in query_upper:
        if "DCTERMS:ISPARTOF" in query_upper and "PARMIGIANO" in query_upper:
            errors.append(
                "WARNING: Parmigiano typically uses ^lime:entry with Lexicon, not dcterms:isPartOf"
            )

    return (len(errors) == 0, errors)


# Dialect resource constants
DIALECT_RESOURCES = {
    "sicilian": {
        "graph": "<http://liita.it/data/id/DialettoSiciliano/>",
        "lemma_bank": "<http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank>",
        "property_pattern": "dcterms:isPartOf",
        "applies_to": "lemma",
        "example": "?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank>",
        "direct_query": "GRAPH <http://liita.it/data/id/DialettoSiciliano/> { ?lemma a lila:Lemma }",
    },
    "parmigiano": {
        "graph": "<http://liita.it/data/id/DialettoParmigiano>",
        "lexicon": "<http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon>",
        "property_pattern": "^lime:entry",
        "applies_to": "lexical entry",
        "example": "?parmigianoLexEntry ^lime:entry <http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon>",
        "direct_query": "GRAPH <http://liita.it/data/id/DialettoParmigiano> { ?lemma a lila:Lemma }",
    },
    "italian": {
        "graph": "<http://liita.it/data>",
        "lemma_bank": "<http://liita.it/data/id/lemma/LemmaBank>",
        "property_pattern": "dcterms:isPartOf",
        "applies_to": "lemma",
        "example": "?italianLemma dcterms:isPartOf <http://liita.it/data/id/lemma/LemmaBank>",
        "direct_query": "GRAPH <http://liita.it/data> { ?lemma a lila:Lemma }",
    },
}

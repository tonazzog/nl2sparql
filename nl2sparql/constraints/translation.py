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

### Dialect Identification (Use dcterms:isPartOf for ALL dialects!):

**Sicilian Dialect:**
```sparql
?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> .
```

**Parmigiano Dialect:**
```sparql
?parmigianoLemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank> .
```

**Italian (Standard):**
```sparql
?italianLemma dcterms:isPartOf <http://liita.it/data/id/lemma/LemmaBank> .
```

**Alternative for Parmigiano (via Lexicon - use for lexical entries):**
```sparql
?parmigianoLexEntry ^lime:entry <http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon> .
```

**CRITICAL:**
- Use `dcterms:isPartOf` with LemmaBank URI for identifying LEMMAS
- Use `^lime:entry` with Lexicon URI for identifying LEXICAL ENTRIES (Parmigiano only)
- **Do NOT use GRAPH clauses** for dialect queries!

---

### QUERYING DIALECT LEMMAS (NO GRAPH NEEDED!):

**CRITICAL: Do NOT use GRAPH clauses for dialect queries!**
Use the identification patterns below instead.

**Sicilian lemmas - use dcterms:isPartOf:**
```sparql
SELECT ?lemma ?wr
WHERE {
  ?lemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> .
  ?lemma ontolex:writtenRep ?wr .
}
```

**Parmigiano lemmas - use dcterms:isPartOf:**
```sparql
SELECT ?lemma ?wr
WHERE {
  ?lemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank> .
  ?lemma ontolex:writtenRep ?wr .
}
```

**POS distribution in Sicilian:**
```sparql
SELECT ?pos (COUNT(DISTINCT ?lemma) AS ?count)
WHERE {
  ?lemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> .
  ?lemma lila:hasPOS ?pos .
}
GROUP BY ?pos
ORDER BY DESC(?count)
```

**POS distribution in Parmigiano:**
```sparql
SELECT ?pos (COUNT(DISTINCT ?lemma) AS ?count)
WHERE {
  ?lemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank> .
  ?lemma lila:hasPOS ?pos .
}
GROUP BY ?pos
ORDER BY DESC(?count)
```

**Sicilian lemmas starting with 'd' or 'r':**
```sparql
SELECT ?lemma ?wr1 ?wr2
WHERE {
  ?lemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> .
  ?lemma ontolex:writtenRep ?wr1, ?wr2 .
  FILTER(?wr1 != ?wr2)
  FILTER(regex(str(?wr1), "^d"))
  FILTER(regex(str(?wr2), "^r"))
}
```

---

### MULTI-DIALECT TRANSLATIONS (CRITICAL!):

When querying translations to MULTIPLE dialects, use DIFFERENT Italian lexical entry variables!

**WRONG - Same variable for both dialects (returns empty results):**
```sparql
?italianLexEntry ontolex:canonicalForm ?liitaLemma .
?italianLexEntry vartrans:translatableAs ?sicilianLexEntry .   # Sicilian
?italianLexEntry vartrans:translatableAs ?parmigianoLexEntry . # Parmigiano - WRONG!
```

**CORRECT - Different variables for each dialect:**
```sparql
# Sicilian translation (via one Italian lexical entry)
?italianSicilianLexEntry ontolex:canonicalForm ?liitaLemma .
?italianSicilianLexEntry vartrans:translatableAs ?sicilianLexEntry .
?sicilianLexEntry ontolex:canonicalForm ?sicilianLemma .
?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> ;
               ontolex:writtenRep ?sicilianWord .

# Parmigiano translation (via DIFFERENT Italian lexical entry!)
?italianParmigianoLexEntry ontolex:canonicalForm ?liitaLemma .
?italianParmigianoLexEntry vartrans:translatableAs ?parmigianoLexEntry .
?parmigianoLexEntry ontolex:canonicalForm ?parmigianoLemma .
?parmigianoLemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank> ;
                 ontolex:writtenRep ?parmigianoWord .
```

**Complete example - Italian adjectives with Sicilian + Parmigiano + definition:**
```sparql
SELECT DISTINCT ?italianWord ?definition ?sicilianWord ?parmigianoWord
WHERE {
  # 1. Get Italian adjectives from CompL-it with definition
  SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?italianWord ] ;
          ontolex:sense [ skos:definition ?definition ] .
    FILTER(REGEX(STR(?italianWord), "oso$", "i"))
  }

  # 2. Link to LiITA lemma
  ?word ontolex:canonicalForm ?liitaLemma .
  GRAPH <http://liita.it/data> {
    ?liitaLemma a lila:Lemma ;
                lila:hasPOS lila:adjective .
  }

  # 3. Sicilian translation (via Italian lexical entry for Sicilian)
  ?italianSicilianLexEntry ontolex:canonicalForm ?liitaLemma .
  ?italianSicilianLexEntry vartrans:translatableAs ?sicilianLexEntry .
  ?sicilianLexEntry ontolex:canonicalForm ?sicilianLemma .
  ?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> ;
                 ontolex:writtenRep ?sicilianWord .

  # 4. Parmigiano translation (via DIFFERENT Italian lexical entry!)
  ?italianParmigianoLexEntry ontolex:canonicalForm ?liitaLemma .
  ?italianParmigianoLexEntry vartrans:translatableAs ?parmigianoLexEntry .
  ?parmigianoLexEntry ontolex:canonicalForm ?parmigianoLemma .
  ?parmigianoLemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank> ;
                   ontolex:writtenRep ?parmigianoWord .
}
```

**KEY INSIGHT:**
- The SAME Italian lemma can have MULTIPLE lexical entries
- Each lexical entry may link to a DIFFERENT dialect
- Use descriptive variable names: `?italianSicilianLexEntry`, `?italianParmigianoLexEntry`

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
- [ ] **Multi-dialect queries**: Use DIFFERENT Italian entry variables for each dialect!
  - WRONG: Same `?italianLexEntry` for both Sicilian and Parmigiano
  - CORRECT: `?italianSicilianLexEntry` and `?italianParmigianoLexEntry`
- [ ] Translation uses `vartrans:translatableAs` on lexical entries, NOT lemmas
- [ ] `ontolex:canonicalForm` connects lexical entries to lemmas
- [ ] `ontolex:writtenRep` accessed on lemmas
- [ ] Morphological properties (POS, gender) queried on lemmas
- [ ] **No GRAPH clauses needed** for dialect lemma queries (use dcterms:isPartOf)
- [ ] Identify dialect lemmas with `dcterms:isPartOf <...LemmaBank>`
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
# NOTE: Do NOT use GRAPH clauses - use dcterms:isPartOf with LemmaBank instead
DIALECT_RESOURCES = {
    "sicilian": {
        "lemma_bank": "<http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank>",
        "property_pattern": "dcterms:isPartOf",
        "applies_to": "lemma",
        "example": "?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank>",
    },
    "parmigiano": {
        "lemma_bank": "<http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank>",
        "lexicon": "<http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon>",
        "property_pattern": "dcterms:isPartOf",
        "applies_to": "lemma",
        "example": "?parmigianoLemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank>",
        "alt_pattern": "^lime:entry",
        "alt_example": "?parmigianoLexEntry ^lime:entry <http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon>",
    },
    "italian": {
        "lemma_bank": "<http://liita.it/data/id/lemma/LemmaBank>",
        "property_pattern": "dcterms:isPartOf",
        "applies_to": "lemma",
        "example": "?italianLemma dcterms:isPartOf <http://liita.it/data/id/lemma/LemmaBank>",
    },
}

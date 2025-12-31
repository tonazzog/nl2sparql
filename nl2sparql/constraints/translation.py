"""Translation query constraints for LiIta SPARQL synthesis."""

TRANSLATION_MANDATORY_PATTERNS = """
## CRITICAL TRANSLATION QUERY CONSTRAINTS

Translation queries have specific structural requirements based on OntoLex-Lemon vocabulary.

### CORE PRINCIPLE: Translations Link Lexical Entries, NOT Lemmas

**CORRECT PATTERN:**
```sparql
# Get lexical entry with Italian lemma
?italianLexEntry ontolex:canonicalForm ?italianLemma .

# Translation at LEXICAL ENTRY level
?italianLexEntry vartrans:translatableAs ?dialectLexEntry .

# Get dialect lemma from dialect lexical entry
?dialectLexEntry ontolex:canonicalForm ?dialectLemma .
```

**WRONG PATTERN:**
```sparql
# WRONG - Translation is NOT directly on lemmas
?italianLemma vartrans:translatableAs ?dialectLemma .
```

---

### Basic Translation Structure:
```sparql
# Step 1: Italian lemma
?italianLemma a lila:Lemma ;
              ontolex:writtenRep ?italianWord .

# Step 2: Get lexical entry with this lemma as canonical form
?italianLexEntry ontolex:canonicalForm ?italianLemma .

# Step 3: Translation link (at lexical entry level!)
?italianLexEntry vartrans:translatableAs ?dialectLexEntry .

# Step 4: Get dialect lemma from dialect lexical entry
?dialectLexEntry ontolex:canonicalForm ?dialectLemma .

# Step 5: Get dialect word
?dialectLemma ontolex:writtenRep ?dialectWord .
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

- [ ] Translation uses `vartrans:translatableAs` on lexical entries
- [ ] NOT using `vartrans:translatableAs` directly on lemmas
- [ ] `ontolex:canonicalForm` connects lexical entries to lemmas
- [ ] `ontolex:writtenRep` accessed on lemmas (or via property path)
- [ ] Sicilian uses `dcterms:isPartOf <...DialettoSiciliano/lemma/LemmaBank>`
- [ ] Parmigiano uses `^lime:entry <...DialettoParmigiano/Lexicon>`
- [ ] Morphological properties (POS, gender) queried on lemmas, not entries
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

    # Check 2: Should use canonicalForm to connect entries to lemmas
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

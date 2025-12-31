"""Multi-entry pattern constraints for LiIta SPARQL synthesis.

This module addresses the #1 source of errors: understanding that one lemma
can have multiple lexical entries from different datasets.
"""

import re

MULTI_ENTRY_CRITICAL_PATTERN = """
## CRITICAL: ONE LEMMA = MULTIPLE LEXICAL ENTRIES

This is the #1 source of errors in synthetic queries.

### THE FUNDAMENTAL ARCHITECTURE:

In LiIta, a SINGLE LEMMA can have MULTIPLE LEXICAL ENTRIES from different datasets:

```
         ?lemma (lila:Lemma) "triste"
                    |
                    | ontolex:canonicalForm
        +-----------+-----------+--------------+
        |           |           |              |
   ?elitaEntry  ?sentixEntry  ?transEntry  ?senseEntry
   [ELITA graph] [no graph]   [no graph]   [SERVICE]
   - HasEmotion  - Polarity   - translation - definition
```

### WHY THIS MATTERS:

Different properties come from DIFFERENT lexical entries:
- **Emotions** (elita:HasEmotion) -> on ELITA entries in ELITA graph
- **Polarity** (marl:hasPolarity) -> on SENTIX entries OUTSIDE graphs
- **Translations** (vartrans:translatableAs) -> on translation entries OUTSIDE graphs
- **Definitions** (ontolex:sense) -> on sense entries in external SERVICE

### WRONG PATTERN (DO NOT DO THIS):
```sparql
# WRONG: Using same variable for emotion AND polarity
?lexEntry ontolex:canonicalForm ?lemma .
GRAPH <http://w3id.org/elita> {
    ?lexEntry elita:HasEmotion ?emotion ;      # WRONG!
              marl:hasPolarityValue ?value .    # WRONG! MARL not in ELITA graph!
}
```

**Problems:**
1. Assumes emotion and polarity are on same entry (they're not!)
2. Puts MARL inside ELITA graph (it's not there!)
3. Will return EMPTY RESULTS

### CORRECT PATTERN (MANDATORY):
```sparql
# Step 1: Get the lemma (join point)
?lemma a lila:Lemma ;
       ontolex:writtenRep ?word .

# Step 2: Emotion entry (in ELITA graph)
?emotionEntry ontolex:canonicalForm ?lemma .
GRAPH <http://w3id.org/elita> {
    ?emotionEntry elita:HasEmotion ?emotion .
    ?emotion rdfs:label ?emotionLabel .
}

# Step 3: Polarity entry (DIFFERENT entry, OUTSIDE graphs)
?polarityEntry ontolex:canonicalForm ?lemma ;
               marl:hasPolarity ?polarity ;
               marl:hasPolarityValue ?polarityValue .
?polarity rdfs:label ?polarityLabel .

# Step 4: Translation entry (DIFFERENT entry, OUTSIDE graphs)
?translationEntry ontolex:canonicalForm ?lemma ;
                  vartrans:translatableAs ?dialectEntry .

# The LEMMA is the JOIN POINT connecting all entries!
```

### VARIABLE NAMING CONVENTION:

Use descriptive variable names to avoid confusion:

**For URIs (resources):**
- `?lemma` - for lila:Lemma resources
- `?word`, `?entry` - for ontolex:LexicalEntry resources
- `?elitaEntry` or `?emotionEntry` - for entries with emotions
- `?sentixEntry` or `?polarityEntry` - for entries with polarity
- `?translationEntry` or `?italianLexEntry` - for entries with translations
- `?dialectEntry` or `?sicilianLexEntry` - for dialect entries

**For literals (strings):**
- `?wr`, `?wordWr`, `?writtenRep` - for ontolex:writtenRep values
- `?definition` - for skos:definition values
- `?label` - for rdfs:label values

**CRITICAL: NEVER reuse a variable for both a URI and a literal!**
```sparql
# WRONG - ?word used for both resource and string:
?word ontolex:canonicalForm [ ontolex:writtenRep ?word ]

# CORRECT - different variables:
?word ontolex:canonicalForm [ ontolex:writtenRep ?wordWr ]
```

DO NOT use generic `?lexEntry` for multiple different entry types!

### VALIDATION CHECKLIST:

When combining emotion + polarity, verify:
- [ ] Emotion uses one variable (e.g., ?emotionEntry)
- [ ] Polarity uses DIFFERENT variable (e.g., ?polarityEntry)
- [ ] Both share same ?lemma via canonicalForm
- [ ] Emotion properties INSIDE GRAPH <http://w3id.org/elita>
- [ ] Polarity properties OUTSIDE any graph
- [ ] Translation uses yet another variable (e.g., ?translationEntry)
"""


PROPERTY_GRAPH_LOCATIONS = """
## PROPERTY LOCATIONS IN GRAPHS

Some properties are in named graphs, others are not. This is NOT optional!

### INSIDE GRAPH <http://w3id.org/elita>:
- elita:HasEmotion
- elita:Emotion (the emotion resource itself)
- rdfs:label (on emotion resources)

### INSIDE GRAPH <http://liita.it/data>:
- lila:Lemma
- lila:hasPOS
- lila:hasGender
- lila:hasNumber
- ontolex:writtenRep (on lemmas)
- dcterms:isPartOf

### OUTSIDE ANY NAMED GRAPH:
- ontolex:canonicalForm (links entries to lemmas)
- vartrans:translatableAs (translation links)
- marl:hasPolarity (from Sentix dataset)
- marl:hasPolarityValue (from Sentix dataset)
- marl:Polarity (polarity resource)
- rdfs:label (on polarity resources)
- lexinfo:hypernym, lexinfo:hyponym (semantic relations)
- ontolex:sense (lexical senses)

### CRITICAL MISTAKES TO AVOID:
- NEVER put MARL properties inside ELITA graph
- NEVER put emotion properties outside ELITA graph
- NEVER put ontolex:canonicalForm inside a graph
- NEVER assume emotion and polarity are on same lexical entry
"""


def validate_multi_entry_pattern(sparql_query: str) -> tuple[bool, list[str]]:
    """
    Validate that emotion and polarity use different entry variables.

    Args:
        sparql_query: The SPARQL query to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    query_upper = sparql_query.upper()

    # Check if both emotion and polarity are present
    has_emotion = "ELITA:HASEMOTION" in query_upper
    has_polarity = (
        "MARL:HASPOLARITY" in query_upper or "MARL:HASPOLARITYVALUE" in query_upper
    )

    if has_emotion and has_polarity:
        # Find the variable used for emotion
        emotion_var_match = re.search(r"\?(\w+)\s+ELITA:HASEMOTION", query_upper)
        if emotion_var_match:
            emotion_var = emotion_var_match.group(1)

            # Check if same variable is used for MARL
            polarity_var_match = re.search(r"\?(\w+)\s+MARL:", query_upper)
            if polarity_var_match:
                polarity_var = polarity_var_match.group(1)

                if emotion_var == polarity_var:
                    errors.append(
                        f"CRITICAL: Variable ?{emotion_var} used for BOTH emotion and polarity. "
                        f"These must be on DIFFERENT lexical entries. "
                        f"Use ?emotionEntry for emotions and ?polarityEntry for polarity, "
                        f"both linking to the same ?lemma via canonicalForm."
                    )

        # Check if MARL is inside ELITA graph
        graph_pattern = r"GRAPH\s*<[^>]*ELITA[^>]*>.*?MARL:"
        if re.search(graph_pattern, query_upper, re.DOTALL):
            errors.append(
                "CRITICAL: MARL properties (polarity) found inside ELITA graph. "
                "MARL properties are from Sentix dataset and should be OUTSIDE any graph."
            )

    return (len(errors) == 0, errors)

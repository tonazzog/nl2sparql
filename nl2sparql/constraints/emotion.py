"""Emotion query constraints for LiIta SPARQL synthesis."""

EMOTION_MANDATORY_PATTERNS = """
## CRITICAL EMOTION QUERY PATTERNS

### 1. NAMED GRAPH STRUCTURE (MANDATORY):
```sparql
# Emotion annotations are in the ELITA graph
GRAPH <http://w3id.org/elita> {
    ?expression elita:HasEmotion ?emotion .
    ?expression marl:Polarity ?polarity .
    ?expression marl:hasPolarityValue ?polarityValue .
    ?emotion rdfs:label ?emotionLabel .
}

# Lemmas are in the main data graph
GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma .
    ?lemma ontolex:writtenRep ?wr .
}

# Link them OUTSIDE the graphs:
?expression ontolex:canonicalForm ?lemma .
```

### 2. CORRECT LINKING PATTERN (CRITICAL):
**WRONG**: `?lemma elita:HasEmotion ?emotion` (lemmas don't have emotions directly)
**RIGHT**: `?lexicalEntry elita:HasEmotion ?emotion` (lexical entries have emotions)

**Correct Flow**:
- Step 1: Get lemma from data graph
- Step 2: Get lexical entry that has this lemma as canonicalForm
- Step 3: Get emotions from lexical entry in ELITA graph

### 3. EMOTION PROPERTY PATTERNS:
```sparql
# Emotion attachment (in ELITA graph)
?expression elita:HasEmotion ?emotion .
?emotion rdfs:label ?emotionLabel .
?expression rdfs:label ?expressionLabel .

# Polarity (in ELITA graph)
?expression marl:Polarity ?polarity .
?expression marl:hasPolarityValue ?polarityValue .

# Common filters
FILTER(?polarityValue > 0.7)  # Highly positive
FILTER(?polarityValue < -0.5) # Negative
FILTER regex(str(?emotionLabel), "gioia", "i")  # Specific emotion
```

### 4. VALID EMOTION LABELS (use these in filters):
- "gioia" (joy)
- "tristezza" (sadness)
- "paura" (fear)
- "rabbia" (anger)
- "amore" (love)
- "sorpresa" (surprise)

### 5. COMMON ANTI-PATTERNS TO AVOID:
- `?lemma elita:HasEmotion ?emotion` - Emotions are NOT on lemmas
- `?lemma marl:Polarity ?polarity` - Polarity is NOT on lemmas
- Querying emotions without GRAPH <http://w3id.org/elita>
- Forgetting to link via ontolex:canonicalForm
- Using emotions outside the ELITA graph

### 6. WORKING QUERY TEMPLATES:

**Template A: Basic Emotion Query**
```sparql
SELECT DISTINCT ?wr ?emotionLabel ?polarityValue
WHERE {
  GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ;
           ontolex:writtenRep ?wr .
  }

  ?lexEntry ontolex:canonicalForm ?lemma .

  GRAPH <http://w3id.org/elita> {
    ?lexEntry elita:HasEmotion ?emotion ;
              marl:hasPolarityValue ?polarityValue .
    ?emotion rdfs:label ?emotionLabel .
  }
}
LIMIT 50
```

**Template B: Emotion + POS Filter**
```sparql
SELECT DISTINCT ?wr ?emotionLabel ?pos
WHERE {
  GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ;
           lila:hasPOS ?pos ;
           ontolex:writtenRep ?wr .
  }

  ?lexEntry ontolex:canonicalForm ?lemma .

  GRAPH <http://w3id.org/elita> {
    ?lexEntry elita:HasEmotion ?emotion .
    ?emotion rdfs:label ?emotionLabel .
  }

  FILTER(?pos = lila:verb)
}
LIMIT 50
```

**Template C: Multiple Emotions**
```sparql
SELECT DISTINCT ?wr
WHERE {
  GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ;
           ontolex:writtenRep ?wr .
  }

  ?lexEntry ontolex:canonicalForm ?lemma .

  GRAPH <http://w3id.org/elita> {
    ?lexEntry elita:HasEmotion ?emotion1 ;
              elita:HasEmotion ?emotion2 .
    ?emotion1 rdfs:label ?label1 .
    ?emotion2 rdfs:label ?label2 .
  }

  FILTER regex(str(?label1), "gioia", "i")
  FILTER regex(str(?label2), "tristezza", "i")
}
LIMIT 50
```
"""


EMOTION_COMBINATION_PATTERNS = {
    ("basic", "emotion"): {
        "mandatory_structure": """
        GRAPH <http://liita.it/data> {
            ?lemma a lila:Lemma ;
                   lila:hasPOS ?pos ;
                   ontolex:writtenRep ?wr .
        }

        ?lexEntry ontolex:canonicalForm ?lemma .

        GRAPH <http://w3id.org/elita> {
            ?lexEntry elita:HasEmotion ?emotion .
            ?emotion rdfs:label ?emotionLabel .
        }
        """,
        "example": "Find all verbs with emotion 'gioia'",
        "key_points": [
            "Use GRAPH for both data and elita",
            "Link via ontolex:canonicalForm",
            "Apply POS filter in data graph",
            "Get emotions from lexEntry in elita graph",
        ],
    },
    ("translation", "emotion"): {
        "mandatory_structure": """
        GRAPH <http://liita.it/data> {
            ?lemma a lila:Lemma ;
                   ontolex:writtenRep ?italian .
        }

        ?lexEntry ontolex:canonicalForm ?lemma ;
                  vartrans:translatableAs ?dialectLemma .

        ?dialectLemma ontolex:writtenRep ?dialect .

        GRAPH <http://w3id.org/elita> {
            ?lexEntry elita:HasEmotion ?emotion ;
                      marl:hasPolarityValue ?polarityValue .
            ?emotion rdfs:label ?emotionLabel .
        }
        """,
        "example": "Find Italian words with Sicilian translations that express sadness",
        "key_points": [
            "Translations link to lexical entries, not directly to lemmas",
            "Emotions attached to Italian lexEntry",
            "Dialect lemma gets writtenRep outside graphs",
        ],
    },
    ("sense_definition", "emotion"): {
        "mandatory_structure": """
        GRAPH <http://liita.it/data> {
            ?lemma a lila:Lemma ;
                   ontolex:writtenRep ?wr .
        }

        ?lexEntry ontolex:canonicalForm ?lemma .

        GRAPH <http://w3id.org/elita> {
            ?lexEntry elita:HasEmotion ?emotion ;
                      marl:hasPolarityValue ?polarity .
            ?emotion rdfs:label ?emotionLabel .
        }

        # Federated query for definitions
        SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
            ?word ontolex:canonicalForm [ ontolex:writtenRep ?wr ] ;
                  ontolex:sense ?sense .
            ?sense skos:definition ?definition .
        }
        """,
        "example": "Get definitions and emotions for highly positive words",
        "key_points": [
            "Definitions come from external SERVICE",
            "Match lemmas via writtenRep",
            "Emotions from ELITA graph",
            "Filter polarity before SERVICE call",
        ],
    },
}


def validate_emotion_query(sparql: str) -> tuple[bool, list[str]]:
    """
    Validate that an emotion query follows correct structural patterns.

    Args:
        sparql: The SPARQL query to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    sparql_upper = sparql.upper()

    # Check 1: Must use ELITA graph for emotions
    if "ELITA:HASEMOTION" in sparql_upper or "ELITA:EMOTION" in sparql_upper:
        if "GRAPH <HTTP://W3ID.ORG/ELITA>" not in sparql_upper:
            errors.append(
                "CRITICAL: Emotion properties must be inside GRAPH <http://w3id.org/elita>"
            )

    # Check 2: Must use data graph for lemmas
    if "LILA:LEMMA" in sparql_upper:
        if "GRAPH <HTTP://LIITA.IT/DATA>" not in sparql_upper:
            errors.append(
                "CRITICAL: Lemmas must be queried inside GRAPH <http://liita.it/data>"
            )

    # Check 3: Must link via canonicalForm
    if ("ELITA:HASEMOTION" in sparql_upper) and ("LILA:LEMMA" in sparql_upper):
        if "ONTOLEX:CANONICALFORM" not in sparql_upper:
            errors.append(
                "CRITICAL: Must link lemmas to emotions via ?lexEntry ontolex:canonicalForm ?lemma"
            )

    # Check 4: Emotions should not be directly on lemmas
    query_normalized = sparql_upper.replace(" ", "").replace("\n", "")
    if "?LEMMAELITA:HASEMOTION" in query_normalized:
        errors.append(
            "SEMANTIC ERROR: Emotions are attached to lexical entries, not lemmas directly"
        )

    if "?LEMMAMARL:" in query_normalized:
        errors.append("SEMANTIC ERROR: Polarity is on lexical entries, not lemmas")

    # Check 5: Should have proper variable naming
    if "ELITA:HASEMOTION" in sparql_upper:
        if not any(
            var in sparql
            for var in ["?lexEntry", "?lexicalEntry", "?expression", "?emotionEntry"]
        ):
            errors.append(
                "WARNING: Consider using explicit variable names like ?lexEntry for clarity"
            )

    return (len(errors) == 0, errors)


# Valid emotion labels in the ELITA dataset
VALID_EMOTIONS = [
    "gioia",  # joy
    "tristezza",  # sadness
    "paura",  # fear
    "rabbia",  # anger
    "amore",  # love
    "sorpresa",  # surprise
]

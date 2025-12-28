"""Compositional query patterns for complex semantic reasoning."""

COMPOSITIONAL_PATTERNS = """
## COMPOSITIONAL QUERY REASONING

When a user query requires combining multiple concepts, decompose it step by step.

### Pattern: [Category] + [Property Filter]

**User asks for**: "animali velenosi", "piante commestibili", "strumenti musicali a corda"

**Decomposition Strategy**:
1. Identify the CATEGORY word (animale, pianta, strumento)
2. Identify the FILTER criterion (velenoso, commestibile, a corda)
3. Find hyponyms of the category
4. Filter by definition or other properties

**Example: "Trova tutti gli animali velenosi"**

Step 1: Category = "animale"
Step 2: Filter = "veleno" (root of "velenoso")
Step 3: Build query

```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Step 1: Find the category word
    ?categoryWord ontolex:canonicalForm [ ontolex:writtenRep ?catWr ] .
    FILTER(STR(?catWr) = "animale")
    ?categoryWord ontolex:sense ?categorySense .

    # Step 2: Find all hyponyms (more specific terms)
    # Remember: lexinfo:hypernym points FROM general TO specific
    ?categorySense lexinfo:hypernym ?hyponymSense .

    # Step 3: Get the words and definitions
    ?resultWord ontolex:sense ?hyponymSense ;
                ontolex:canonicalForm [ ontolex:writtenRep ?result ] .
    ?hyponymSense skos:definition ?definition .

    # Step 4: Apply the filter criterion
    FILTER(REGEX(STR(?definition), "veleno", "i"))
}
```

---

### Pattern: [Part] of [Category]

**User asks for**: "parti del corpo degli uccelli", "componenti di un motore"

**Decomposition Strategy**:
1. Identify the WHOLE category (uccello, motore)
2. Find hyponyms of the category (if asking about a general category)
3. Find parts (meronyms) of each

**Example: "Quali sono le parti del corpo di un uccello?"**

```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find "uccello"
    ?bird ontolex:canonicalForm [ ontolex:writtenRep ?birdWr ] .
    FILTER(STR(?birdWr) = "uccello")
    ?bird ontolex:sense ?birdSense .

    # Find parts: partMeronym points FROM part TO whole
    ?partSense lexinfo:partMeronym ?birdSense .

    # Get the part words
    ?partWord ontolex:sense ?partSense ;
              ontolex:canonicalForm [ ontolex:writtenRep ?result ] .
}
```

---

### Pattern: [Relationship] between [A] and [B]

**User asks for**: "relazione tra cane e lupo", "cosa hanno in comune casa e palazzo"

**Decomposition Strategy**:
1. Find both words' senses
2. Check for shared hypernyms (common ancestor)
3. Or check direct relationships

**Example: "Qual è la relazione tra cane e lupo?"**

```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find both words
    ?word1 ontolex:canonicalForm [ ontolex:writtenRep ?wr1 ] .
    FILTER(STR(?wr1) = "cane")
    ?word1 ontolex:sense ?sense1 .

    ?word2 ontolex:canonicalForm [ ontolex:writtenRep ?wr2 ] .
    FILTER(STR(?wr2) = "lupo")
    ?word2 ontolex:sense ?sense2 .

    # Find common hypernym (shared parent category)
    ?sense1 lexinfo:hyponym ?commonSense .
    ?sense2 lexinfo:hyponym ?commonSense .

    ?commonWord ontolex:sense ?commonSense ;
                ontolex:canonicalForm [ ontolex:writtenRep ?commonTerm ] .
}
```

---

### Pattern: Negation or Exclusion

**User asks for**: "animali che non sono mammiferi", "verbi che non esprimono movimento"

**Decomposition Strategy**:
1. Find the category
2. Find the exclusion criterion
3. Use FILTER NOT EXISTS or MINUS

**Example: "Trova animali che non sono mammiferi"**

```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    # Find "animale" hyponyms
    ?animale ontolex:canonicalForm [ ontolex:writtenRep ?aWr ] .
    FILTER(STR(?aWr) = "animale")
    ?animale ontolex:sense ?animaleSense .
    ?animaleSense lexinfo:hypernym ?resultSense .

    # Exclude "mammifero" hyponyms
    FILTER NOT EXISTS {
        ?mammifero ontolex:canonicalForm [ ontolex:writtenRep ?mWr ] .
        FILTER(STR(?mWr) = "mammifero")
        ?mammifero ontolex:sense ?mammiferoSense .
        ?mammiferoSense lexinfo:hypernym ?resultSense .
    }

    ?resultWord ontolex:sense ?resultSense ;
                ontolex:canonicalForm [ ontolex:writtenRep ?result ] .
}
```

---

## REASONING CHECKLIST

When you receive a complex query:

1. **Identify the core concept(s)**: What entities are we looking for?
2. **Identify relationships needed**: Hyponymy? Meronymy? Definition search?
3. **Identify filters**: Text patterns? Property constraints?
4. **Determine direction**:
   - More specific → use `lexinfo:hypernym` on the general term
   - More general → use `lexinfo:hyponym` on the specific term
   - Parts → use `lexinfo:partMeronym` pointing TO the whole
5. **Combine patterns**: Build the query step by step

## COMMON COMPOSITIONAL TRIGGERS

| User phrase | Decomposition needed |
|-------------|---------------------|
| "tutti i/gli/le [X] che sono [Y]" | Hyponyms of X + filter Y |
| "[X] [adjective]" | Hyponyms of X + definition filter |
| "tipi di [X]" | Direct hyponyms of X |
| "parti di [X]" | Meronyms of X |
| "[X] e [Y] hanno in comune" | Shared hypernym |
| "[X] che non sono [Y]" | Hyponyms of X minus hyponyms of Y |
"""


def detect_compositional_pattern(question: str) -> list[str]:
    """
    Detect if a question requires compositional reasoning.

    Supports both Italian and English questions.

    Returns list of detected compositional patterns.
    """
    import re

    patterns_detected = []
    question_lower = question.lower()

    # Category + adjective pattern (Italian)
    # "animali velenosi", "piante commestibili"
    if re.search(r'\b(tutti|tutte|gli|le|i)\s+\w+\s+\w+(i|e|o|a)\b', question_lower):
        patterns_detected.append("category_filter")

    # Category + adjective pattern (English)
    # "venomous animals", "edible plants", "all the animals that are"
    if re.search(r'\b(all|every|all the)\s+\w+\s+(that|which)\s+(are|have)\b', question_lower):
        patterns_detected.append("category_filter")
    if re.search(r'\b\w+(ous|ive|ble|ing|ed)\s+(animals?|plants?|words?|nouns?|verbs?)\b', question_lower):
        patterns_detected.append("category_filter")

    # "tipi di X" / "types of X" pattern
    if re.search(r'\b(tipi|generi|categorie)\s+di\b', question_lower):
        patterns_detected.append("hyponym_search")
    if re.search(r'\b(types?|kinds?|categories)\s+of\b', question_lower):
        patterns_detected.append("hyponym_search")

    # "parti di X" / "parts of X" pattern
    if re.search(r'\bparti?\s+(di|del|della|degli|delle)\b', question_lower):
        patterns_detected.append("meronym_search")
    if re.search(r'\b(parts?|components?)\s+of\b', question_lower):
        patterns_detected.append("meronym_search")

    # Negation pattern (Italian)
    # "X che non sono Y"
    if re.search(r'\bche\s+non\s+(sono|è|hanno)\b', question_lower):
        patterns_detected.append("negation_filter")
    # Negation pattern (English)
    # "X that are not Y", "X which don't have"
    if re.search(r'\b(that|which)\s+(are\s+not|aren\'t|don\'t|do\s+not)\b', question_lower):
        patterns_detected.append("negation_filter")

    # Relationship/common ancestor pattern (Italian)
    # "relazione tra X e Y" or "in comune"
    if re.search(r'\b(relazione\s+tra|in\s+comune|hanno\s+in\s+comune)\b', question_lower):
        patterns_detected.append("relationship_search")
    # Relationship pattern (English)
    # "relationship between X and Y", "in common"
    if re.search(r'\b(relation(ship)?\s+between|in\s+common|have\s+in\s+common)\b', question_lower):
        patterns_detected.append("relationship_search")

    # Filter with relative clause (Italian)
    # "X che sono Y" or "X che hanno Y"
    if re.search(r'\bche\s+(sono|hanno|esprimono|indicano)\b', question_lower):
        patterns_detected.append("category_filter")
    # Filter with relative clause (English)
    # "X that are Y", "X that have Y", "X which express"
    if re.search(r'\b(that|which)\s+(are|have|express|indicate|contain)\b', question_lower):
        patterns_detected.append("category_filter")

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p in patterns_detected:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique

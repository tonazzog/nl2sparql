# Failure Analysis — claude-sonnet-4-6 on the 100-Query Benchmark

**Model**: `claude-sonnet-4-6`
**Dataset**: `nl2sparql/data/test_dataset.json` (100 test cases, English)
**Report**: `f1_report_anthropic_claude-sonnet-4-6.json`
**Date**: 2026-02-24

---

## Final Scores (after evaluator fixes)

| Metric | Value |
|---|---|
| Avg F1 | **0.7224** |
| Macro F1 | **0.7375** |
| Avg Precision | 0.7159 |
| Avg Recall | 0.7342 |

### By Category

| Category | Avg F1 | n |
|---|---|---|
| translation | 0.8333 | 6 |
| semantic_combined | 0.7393 | 29 |
| complex | 0.7108 | 56 |
| emotion | 0.6667 | 9 |

### By Pattern

| Pattern | Avg F1 | n |
|---|---|---|
| LEXICAL_FORM | 1.0000 | 7 |
| MORPHO_REGEX | 0.9138 | 13 |
| SENSE_DEFINITION | 0.9013 | 18 |
| SENSE_COUNT | 0.8964 | 14 |
| META_GRAPH | 0.8933 | 23 |
| COUNT_ENTITIES | 0.8000 | 15 |
| MULTI_TRANSLATION | 0.7473 | 28 |
| TRANSLATION | 0.7330 | 29 |
| EMOTION_LEXICON | 0.7016 | 36 |
| SERVICE_INTEGRATION | 0.6815 | 48 |
| POS_FILTER | 0.5181 | 21 |
| SEMANTIC_RELATION | **0.2318** | 17 |

---


## Model Failures

---

### Failure Class 1 — SEMANTIC_RELATION: Hypernym/Hyponym Property Confusion

**Affected cases**: 2030, 2032, 2104, and partially 2106
**Pattern**: `SEMANTIC_RELATION` + `SERVICE_INTEGRATION` (+ `POS_FILTER`)
**Avg F1 on these cases**: ~0.0

**Description**: The model confuses `lexinfo:hypernym` and `lexinfo:hyponym` when querying
the external Italian WordNet endpoint. In the LiITA/graphdb ontology, the property
`lexinfo:hypernym` is used to navigate **from a sense to its more specific subtypes** (i.e.
it behaves as a hyponym link from the data modelling perspective). The model applies standard
OWL/WordNet semantics and inverts the property, producing queries that return no results or
completely wrong ones.

**Example — Case 2030**: "What are the hypernyms of the Italian word 'veicolo'?"

```sparql
-- GOLD (returns 133 results)
?sense(veicolo) lexinfo:hypernym ?hypernymSense .
?hypernymWordEntry ontolex:sense ?hypernymSense ;
                   ontolex:canonicalForm [ ontolex:writtenRep ?hypernymWord ] .

-- PREDICTED (returns 0 results)
?sense(veicolo) lexinfo:hyponym ?hypernymSense .   ← wrong property
```

**Example — Case 2032**: "Can you find the hypernyms of 'strumento'?"

```sparql
-- GOLD (returns 1364 results)
?sense lexinfo:hypernym ?hypernymSense .

-- PREDICTED (returns 2 results)
?sense lexinfo:hyponym ?hyperSense .               ← wrong property
```

**Contrast with passing case 2024**: "What are the specific subtypes of 'vehicle'?" also uses
`lexinfo:hypernym` in the gold — and the model generates exactly that, getting F1=1.0. The
model learns the correct property from this wording but fails to generalise when the NL
question uses "hypernym" explicitly.

**Root cause**: The few-shot examples for SEMANTIC_RELATION queries do not make sufficiently
clear that `lexinfo:hypernym` in this endpoint traverses towards **hyponyms** (specific types),
not towards hypernyms (general categories). This is a non-standard usage that contradicts the
property's name.

---

### Failure Class 2 — SEMANTIC_RELATION: Reversed Part-Meronym Direction

**Affected cases**: 2025, 2026, 2028, 2031, 2100, 2102, 2105
**Pattern**: `SEMANTIC_RELATION` + `SERVICE_INTEGRATION`
**Avg F1 on these cases**: ~0.0

**Description**: For questions asking "what are the parts of X?", the gold queries use
`?sense(X) lexinfo:partMeronym ?partSense` (whole → part direction). The model inverts the
triple, generating `?partSense lexinfo:partMeronym ?sense(X)` (part → whole direction),
which asks "what is X a part of?" instead.

**Example — Case 2025**: "What are the component parts that make up a 'building'?"

```sparql
-- GOLD: whole → part (correct, returns 9 results)
?word(edificio) ontolex:sense ?sense .
?sense lexinfo:partMeronym ?partSense .           ← edificio's sense has partMeronym
?partWord ontolex:sense ?partSense ;
          ontolex:canonicalForm [ ontolex:writtenRep ?hyponymWord ] .

-- PREDICTED: part → whole (reversed, returns 184 wrong results)
?partSense lexinfo:partMeronym ?wholeSense .      ← reversed subject/object
```

The reversed query does execute (returning results from other words that are parts of
buildings), which is why pred=184 but gold=9 — the answer is completely wrong.

**Example — Case 2026**: "Parts of 'strumento'" → pred=0 (no results with reversed direction
for this particular word).

**Root cause**: Same issue as Class 1. The convention in this ontology for
`lexinfo:partMeronym` is `whole lexinfo:partMeronym part` (i.e. the property goes from the
whole entity to its parts). This is not immediately obvious, and the model applies the inverse
direction. Better few-shot examples that demonstrate the correct subject/object orientation
of this property are needed.

---

### Failure Class 3 — POS_FILTER: Wrong POS URI Namespace

**Affected cases**: 2055
**Pattern**: `POS_FILTER` + `COUNT_ENTITIES` + `META_GRAPH`

**Description**: The model uses `lexinfo:adjective` (from the `lexinfo:` namespace) instead
of `lila:adjective` (from the `lila:` namespace) as the object of `lila:hasPOS`. Since no
resource in the graph matches `lexinfo:adjective`, the query returns 0 instead of 38,639.

```sparql
-- GOLD (returns 38,639)
?italianLemma lila:hasPOS lila:adjective .

-- PREDICTED (returns 0)
?italianLemma lila:hasPOS lexinfo:adjective .    ← wrong namespace
```

All other POS values (noun, verb, adverb, pronoun) were generated correctly using the `lila:`
namespace. Adjective appears to be the single exception.

**Root cause**: In many lexical ontologies POS values live in the `lexinfo:` namespace. The
model generalises from this convention and misapplies it to the one POS category that happens
to be named identically in both namespaces. A single correct example in the few-shot set
would be sufficient to fix this.

---

### Failure Class 4 — POS_FILTER: Spurious `a lila:Lemma` Type Constraint

**Affected cases**: 2056
**Pattern**: `POS_FILTER` + `COUNT_ENTITIES` + `META_GRAPH`

**Description**: For adverb counting, the model adds an explicit type assertion
`?lemma a lila:Lemma` before the `lila:hasPOS lila:adverb` constraint. The gold query omits
the type check. In the LiITA graph, the `lila:hasPOS` property is carried by resources that
are not declared as `lila:Lemma` instances (or whose type triple is stored in a different
named graph), so the type assertion reduces the count from 5,533 to 811.

```sparql
-- GOLD (returns 5,533)
?italianLemma lila:hasPOS lila:adverb .

-- PREDICTED (returns 811 — over-constrained)
?italianLemma a lila:Lemma ;
              lila:hasPOS lila:adverb .           ← extra type constraint
```

**Root cause**: The model tries to be semantically precise by asserting the expected type of
`?italianLemma`. This is generally good practice in SPARQL generation, but in this graph the
type declaration and the POS triple are not co-located for all resources, making the type
check inadvertently restrictive.

---

### Failure Class 5 — Complex Queries: Variable Mapping Confusion

**Affected cases**: 2043, 2132
**Pattern**: Complex multi-pattern queries (e.g. `EMOTION_LEXICON` + `SENSE_DEFINITION` + `POS_FILTER` + `SERVICE_INTEGRATION`)

**Description**: For queries involving many different output variables, the positional
variable-mapping heuristic in the evaluator incorrectly aligns gold and predicted variables,
causing F1=0 even when the predicted query is structurally reasonable.

**Example — Case 2043**: "What verbs carry strong positive sentiment, and what emotions and
definitions are linked to them?"

Gold primary answer variables: `['emotionLabel', 'definition']`
Predicted SELECT: `?italianWord ?emotionLabel ?polarityValue ?definition`

The positional mapper assigns the first predicted `primary`-category variable (`italianWord`)
to the first gold primary variable (`emotionLabel`), producing the mapping:

```
emotionLabel  →  italianWord   ← wrong
definition    →  emotionLabel  ← wrong
```

Both `emotionLabel` and `italianWord` are classified as `primary` variables by
`ANSWER_VARIABLE_CATEGORIES`, so the category-based Phase 1 matching fires before the
direct-name Phase 2 matching that would correctly map `emotionLabel → emotionLabel`.

**Root cause**: The variable mapping strategy prioritises positional category-matching over
direct name matching. For complex queries where predicted and gold SELECT clauses have
different orderings and additional variables, this causes wrong alignments. Swapping the
priority so that exact name matching (Phase 2) runs before positional category matching
(Phase 1) would fix this class of errors.

---

## Summary Table

| # | Failure Class | Affected Cases | Root Cause | Fixable By |
|---|---|---|---|---|
| 1 | Hypernym/hyponym property confusion | 2030, 2032, 2104, 2106 | Non-standard use of `lexinfo:hypernym` in endpoint | Better few-shot examples |
| 2 | Reversed part-meronym direction | 2025, 2026, 2028, 2031, 2100, 2102, 2105 | Non-standard direction of `lexinfo:partMeronym` | Better few-shot examples |
| 3 | Wrong POS namespace (`lexinfo:` vs `lila:`) | 2055 | Adjective POS uses `lila:` not `lexinfo:` | One corrected example |
| 4 | Spurious `a lila:Lemma` type constraint | 2056 | Type triple not universally co-located in graph | Explicit note in system prompt |
| 5 | Variable mapping confusion | 2043, 2132 | Positional mapper overrides name matching | Swap Phase 1/2 priority in evaluator |

---

## Possible fixes

### 1. Fix few-shot examples for SEMANTIC_RELATION queries (highest impact)

Add at least one example per relation type (hypernym, meronym, synonym) that makes the
correct property direction explicit. The examples should cover:

- `lexinfo:hypernym` used as "sense → more-specific-sense" (i.e. towards hyponyms)
- `sense lexinfo:partMeronym partSense` (whole → part, not part → whole)

Fixing these would recover approximately **11–13 cases** currently scoring F1=0.

### 2. Add a canonical POS namespace example

Include one explicit counting-by-POS example in the few-shot set that uses
`lila:hasPOS lila:adjective` (with the `lila:` namespace) to prevent the
`lexinfo:adjective` substitution.

### 3. Fix variable mapping priority in the evaluator

In `build_variable_mapping` (`nl2sparql/evaluation/f1_evaluator.py`), move Phase 2 (direct
name match) **before** Phase 1 (positional category match). This prevents the evaluator from
penalising correct predictions that use different SELECT orderings.

### 4. Note `a lila:Lemma` restriction in system prompt

Add a note in the system prompt or in the ontology constraints section clarifying that
`lila:hasPOS` can be queried without an explicit `a lila:Lemma` type check, since not all
POS-annotated resources in the graph have an explicit type triple in the default graph.

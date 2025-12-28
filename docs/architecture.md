# NL2SPARQL Architecture

## The Problem

LiITA is a complex linguistic knowledge base with data distributed across multiple sources: the main LiITA graph for lemmas and morphology, ELITA for emotion annotations, dialect translations, and CompL-it for semantic relations. Writing SPARQL queries for this system requires understanding which properties live where, how to join data across graphs and federated services, and the peculiarities of the OntoLex-Lemon vocabulary.

The goal is to let users ask questions in natural language and get valid SPARQL queries back.

## Approach: Retrieval-Augmented Generation

We use a RAG (Retrieval-Augmented Generation) approach rather than relying solely on an LLM's internal knowledge. The reasons:

1. **Domain specificity**: LiITA has a unique architecture that no general-purpose LLM was trained on. The LLM doesn't know that emotions are in a separate graph, or that CompL-it uses a federated SERVICE endpoint.

2. **Precision matters**: SPARQL is unforgiving. A query with the wrong graph URI or property direction will silently return zero results. We need to guide the LLM with exact patterns.

3. **Few-shot learning works**: LLMs are good at pattern matching. If we show them a few similar queries, they can adapt them to new questions. This is more reliable than asking them to generate queries from scratch.

The system works in three stages: detect what kind of query the user needs, retrieve similar examples from a curated dataset, and synthesize a new query using those examples as templates.

---

## Component 1: Pattern Detection

Before retrieving examples, we analyze the user's question to understand what kind of query they need. This serves two purposes:

1. **Retrieval boosting**: We can prioritize examples that match the detected patterns.
2. **Constraint selection**: We can include only the relevant domain constraints in the LLM prompt, keeping it focused.

Pattern detection uses keyword matching in both Italian and English. For example:

- "emozione", "tristezza", "gioia" trigger the EMOTION pattern
- "traduzione", "siciliano", "dialetto" trigger the TRANSLATION pattern
- "definizione", "significato" trigger the SENSE_DEFINITION pattern
- "iperonimo", "iponimo", "parte di" trigger the SEMANTIC_RELATION pattern

We also detect compositional patterns that require multi-step reasoning, like "tutti gli animali velenosi" (all venomous animals), which needs to find hyponyms of "animale" and filter by definition.

This is not meant to be perfect classification. It's a heuristic that improves retrieval and prompt construction. The LLM ultimately decides how to handle the query.

---

## Component 2: Hybrid Retrieval

Given a user question, we need to find the most relevant examples from our dataset of 131 curated query pairs. We use three retrieval methods and combine their scores.

### Why Hybrid?

Each retrieval method has different strengths:

- **Semantic search** finds conceptually similar questions even with different wording ("find sad words" matches "lemmi che esprimono tristezza")
- **BM25** finds lexical matches that semantic models might miss (exact entity names, technical terms)
- **Pattern matching** ensures we retrieve examples with the same structural requirements

Using all three gives us robustness. If one method fails to find good matches, the others can compensate.

### Semantic Search

We use sentence-transformers with the `paraphrase-multilingual-MiniLM-L12-v2` model. This model was chosen because:

1. **Multilingual**: Users ask questions in Italian, but we want the system to also work with English queries. This model handles both.
2. **Lightweight**: At 118M parameters, it's fast enough for real-time use.
3. **Paraphrase-trained**: It's specifically trained to recognize that different phrasings can have the same meaning.

We encode all example questions into vectors and store them in a FAISS index for fast similarity search. At query time, we encode the user's question and find the nearest neighbors.

### BM25

BM25 is a classical text retrieval algorithm based on term frequency. It complements semantic search by catching exact matches that embedding models might overlook.

For example, if the user asks about "braccio" (arm), BM25 will strongly prefer examples that contain exactly that word, while semantic search might return examples about other body parts.

We tokenize questions into words and use the rank-bm25 library for scoring.

### Pattern Boosting

Examples in our dataset are tagged with patterns (EMOTION, TRANSLATION, SEMANTIC_RELATION, etc.). When we detect patterns in the user's question, we boost examples that share those patterns.

This is important because structural similarity matters more than semantic similarity for query generation. A question about "dog emotions" is structurally more similar to "cat emotions" than to "dog translations", even though a semantic model might rank them differently.

### Score Combination

We combine the three scores with configurable weights:

```
final_score = w1 * semantic + w2 * bm25 + w3 * pattern
```

Default weights are (0.4, 0.3, 0.3). We normalize each score to [0, 1] before combining so they're on the same scale.

---

## Component 3: Constraint Prompts

The LLM needs to know the rules of LiITA's architecture. We encode this knowledge as structured prompts that are included based on the detected patterns.

### Why Not Just Use Examples?

Examples alone aren't enough because:

1. **Negative knowledge**: The LLM needs to know what NOT to do. Examples show correct patterns but don't explain why alternatives are wrong.

2. **Edge cases**: Our dataset can't cover every combination. The constraints help the LLM generalize.

3. **Counterintuitive patterns**: Some LiITA patterns are surprising. For example, `lexinfo:hypernym` points FROM the general term TO the specific term (the opposite of what the name suggests). Without explicit documentation, even a smart LLM will get this wrong.

### Constraint Categories

**Base constraints** are always included:
- SPARQL prefixes
- Overview of LiITA's three data sources (main graph, ELITA, CompL-it)
- Property location table (which properties go in which graph)

**Pattern-specific constraints** are added based on detection:

- **Emotion constraints**: ELITA graph patterns, how to link emotion entries to lemmas, polarity handling
- **Translation constraints**: Sicilian vs Parmigiano differences, the vartrans:translatableAs pattern
- **Semantic constraints**: CompL-it SERVICE patterns, the counterintuitive direction of hypernym/hyponym/meronym properties
- **Compositional constraints**: Decomposition strategies for complex queries

### The Multi-Entry Problem

One subtle but critical pattern: in LiITA, a single lemma can have multiple lexical entries from different sources. The ELITA entry for emotions is different from the Sentix entry for polarity, which is different from the translation entry.

This means queries that combine emotion and polarity need to use different variables:

```sparql
?emotionEntry ontolex:canonicalForm ?lemma .
GRAPH <http://w3id.org/elita> {
    ?emotionEntry elita:HasEmotion ?emotion .
}

?polarityEntry ontolex:canonicalForm ?lemma .
?polarityEntry marl:hasPolarityValue ?polarity .
```

Using the same variable would silently fail. This pattern is documented in the multi-entry constraint, which is triggered when we detect queries combining multiple data sources.

---

## Component 4: Query Synthesis

With retrieved examples and relevant constraints, we construct a prompt for the LLM:

1. **System prompt**: Domain knowledge and constraints
2. **Few-shot examples**: The top-k retrieved query pairs
3. **User question**: The natural language question to translate

The LLM generates a SPARQL query. We then validate it and optionally attempt to fix errors.

### Validation

We validate queries at three levels:

1. **Syntax**: Parse with rdflib to catch malformed SPARQL
2. **Semantic**: Check that the query follows the documented constraints (correct graphs, SERVICE usage, etc.)
3. **Endpoint**: Execute against the LiITA SPARQL endpoint to verify it returns results

If validation fails and auto-fix is enabled, we send the query back to the LLM with the error message and ask it to correct the issue. This retry loop runs up to a configurable number of times.

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| RAG over pure LLM | LiITA-specific knowledge not in training data |
| Hybrid retrieval | Combines semantic understanding with lexical precision |
| Multilingual embeddings | Support Italian and English queries |
| Lightweight embedding model | Fast enough for interactive use |
| Pattern detection | Focus retrieval and prompt construction |
| Structured constraints | Encode negative knowledge and edge cases |
| Multi-level validation | Catch errors before users see them |
| Retry loop | Many errors are fixable with guidance |

---

## Cross-Endpoint Linking

A discovery during development: CompL-it and LiITA share URIs for lexical entries. This means variables bound inside a SERVICE block can be used directly outside:

```sparql
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?wordMeronym ontolex:sense ?senseMeronym .
}
# Same variable works in LiITA - no string matching needed
?wordMeronym ontolex:canonicalForm ?liitaLemma .
```

This is more efficient than joining on string values and was incorporated into the semantic constraints after we identified it from working query examples.

---

## Limitations and Future Work

**Dataset size**: 131 examples is enough for common patterns but may miss rare query types. The system can be improved by adding more examples as users discover gaps.

**Pattern detection**: Keyword-based detection is simple but brittle. A classifier trained on the examples could be more robust.

**Compositional queries**: Complex multi-step queries like "find all venomous animals" require reasoning that few-shot learning doesn't always capture. More sophisticated decomposition strategies could help.

**Evaluation**: We haven't systematically measured accuracy across query types. A benchmark with held-out examples would help identify weaknesses.

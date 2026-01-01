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

## Component 5: Agentic Architecture (LangGraph)

While the basic RAG approach works well for straightforward queries, complex questions benefit from a more intelligent workflow. The agentic architecture uses LangGraph to create a self-correcting pipeline that can reason about failures and adapt its approach.

### Why Agentic?

The basic pipeline has limitations:

1. **Blind retries**: When a query fails, the fix loop just sends the error back to the LLM. It doesn't analyze why the failure happened or consider alternative approaches.

2. **No execution feedback**: The basic system validates syntax but doesn't always verify that results make semantic sense.

3. **Static retrieval**: Examples are retrieved once at the start. If the first attempt fails, we don't reconsider which examples might be more helpful.

4. **No schema exploration**: When queries return empty results, there's no mechanism to discover what properties or patterns actually exist in the data.

### The Agent Workflow

The agent implements a state machine with the following nodes:

```
analyze → plan → retrieve → generate → execute → verify
                    ↑                        ↓
                    └── refine ←── (if invalid)
                    └── explore ←── (if empty results)
                                         ↓
                                      output
```

**Nodes:**

| Node | Purpose | LLM Tier |
|------|---------|----------|
| `analyze` | Detect patterns, complexity, dialects needed | Fast |
| `plan` | Decompose complex queries into sub-tasks | Fast |
| `retrieve` | Get relevant examples and constraints | - |
| `generate` | Generate SPARQL query | Default (capable) |
| `execute` | Run query against endpoint | - |
| `verify` | Check results semantically | Fast |
| `refine` | Record failure, prepare for retry | - |
| `explore` | Discover schema when stuck | - |
| `output` | Prepare final result | - |

### Model Tiers

The agent uses different model tiers for different tasks to balance cost and capability:

- **Fast tier**: Used for analysis, planning, and verification. These tasks require understanding but not complex generation. Uses cheaper models like `gpt-4.1-mini` or `claude-3-5-haiku`.

- **Default tier**: Used for SPARQL generation, which requires precise syntax and domain knowledge. Uses more capable models like `gpt-4.1` or `claude-sonnet-4`.

If a specific model is provided, it's used for all operations.

### Self-Correction Loop

When verification fails, the agent enters a refinement loop:

1. The failed query and error are recorded in `refinement_history`
2. On the next generation attempt, this history is included in the prompt
3. The LLM can learn from previous mistakes and try a different approach
4. After 3 attempts (configurable), the agent outputs the best effort

This is more effective than blind retries because the LLM sees the full context of what went wrong.

### Schema Exploration

When a query returns zero results and the agent hasn't explored the schema yet, it can query the endpoint to discover available properties:

```sparql
SELECT DISTINCT ?property WHERE {
    ?s ?property ?o .
    FILTER(STRSTARTS(STR(?property), "http://lila-erc.eu/ontologies/lila/"))
} LIMIT 30
```

This helps when the LLM generates queries with incorrect property names. The discovered properties are included in subsequent generation attempts.

### State Management

The agent uses a TypedDict state that flows through all nodes:

```python
class NL2SPARQLState(TypedDict):
    # Input
    question: str
    language: str
    provider: str
    model: str | None
    api_key: str | None

    # Analysis
    detected_patterns: list[str]
    complexity: Literal["simple", "moderate", "complex"]
    requires_service: bool
    dialects_needed: list[str]

    # Generation
    generated_sparql: str
    generation_attempts: int

    # Execution
    result_count: int
    execution_error: str | None

    # Refinement (accumulates across attempts)
    refinement_history: Annotated[list[dict], add]

    # Output
    final_sparql: str
    confidence: float
```

The `refinement_history` uses LangGraph's `Annotated[..., add]` to accumulate entries across iterations rather than replacing them.

### Integration with Existing Components

The agent reuses all existing components:

- **Retrieval**: Uses `HybridRetriever` for example retrieval
- **Constraints**: Uses `get_constraints_for_patterns` for domain rules
- **Validation**: Uses `validate_syntax` and `validate_endpoint`

This means improvements to retrieval or constraints automatically benefit the agent.

### When to Use the Agent

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple, single-pattern queries | Basic `NL2SPARQL` (faster, cheaper) |
| Complex multi-pattern queries | Agent (better accuracy) |
| Queries that often fail validation | Agent (self-correction) |
| Interactive/exploratory use | Agent with streaming |
| Batch evaluation | Basic `NL2SPARQL` (more predictable) |

### Usage

```python
from nl2sparql.agent import NL2SPARQLAgent

agent = NL2SPARQLAgent(
    provider="openai",
    model="gpt-4.1",        # optional
    api_key="sk-...",      # optional
)

# Basic translation
result = agent.translate("Trova aggettivi con traduzioni siciliane")

# Streaming to see progress
for node, state in agent.stream("Complex query here"):
    print(f"[{node}] completed")
```

### CLI Commands

```bash
# Agentic translation
nl2sparql agent "Find nouns expressing sadness"
nl2sparql agent -p anthropic "Traduzioni siciliane"
nl2sparql agent --stream "Step-by-step output"

# Visualize workflow
nl2sparql agent-viz
```

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

**Pattern detection**: Keyword-based detection is simple but brittle. A classifier trained on the examples could be more robust. The agent's LLM-based analysis is more flexible but adds latency.

**Compositional queries**: Complex multi-step queries like "find all venomous animals" require reasoning that few-shot learning doesn't always capture. The agent's planning node helps with decomposition, but more sophisticated strategies could help.

**Execution accuracy**: The current evaluation measures syntax validity and component matching, but not execution accuracy (comparing result sets). Adding ground truth result comparison would provide stronger validation.

**Agent cost**: The agentic workflow makes multiple LLM calls (analyze, plan, generate, verify), which increases cost and latency compared to the basic pipeline. The tiered model approach mitigates this by using cheaper models for simpler tasks.

## Evaluation Framework

The system includes an evaluation framework with a structured test dataset covering all patterns and combinations. See [evaluation.md](evaluation.md) for details.

Key metrics:
- Syntax validity rate
- Endpoint execution success
- Component matching score (expected SPARQL patterns present)
- Pattern detection accuracy

Run evaluation via CLI:
```bash
nl2sparql evaluate -p openai -o report.json
```

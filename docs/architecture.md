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

2. **Static retrieval**: Examples are retrieved once at the start. If the first attempt fails, we don't reconsider which examples might be more helpful.

3. **No schema exploration**: When queries return empty results, there's no mechanism to discover what properties or patterns actually exist in the data.

4. **No auto-correction**: Common issues like case-sensitive filters or variable reuse bugs require regeneration rather than targeted fixes.

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
| `generate` | Generate SPARQL with mandatory prefixes and domain rules | Default (capable) |
| `execute` | Run query, auto-fix case-sensitive filters, detect variable reuse | - |
| `verify` | Check for technical errors only (syntax, execution) | - |
| `refine` | Record failure, prepare for retry | - |
| `explore` | Discover schema when stuck | - |
| `output` | Prepare final result | - |

**Important**: The `verify` node intentionally does NOT perform semantic verification of results. This is a deliberate design choice because:
1. LLMs lack domain knowledge (e.g., dialect translations can be identical to Italian)
2. Emotion lexicons have complex associations that appear "wrong" to naive analysis
3. Over-aggressive verification caused regeneration which introduced new bugs

If a query executes successfully and returns results, we trust the retrieval-based generation.

### Model Tiers

The agent uses different model tiers for different tasks to balance cost and capability:

- **Fast tier**: Used for analysis, planning, and verification. These tasks require understanding but not complex generation. Uses cheaper models like `gpt-4.1-mini` or `claude-3-5-haiku`.

- **Default tier**: Used for SPARQL generation, which requires precise syntax and domain knowledge. Uses more capable models like `gpt-4.1` or `claude-sonnet-4`.

If a specific model is provided, it's used for all operations.

### Generation Rules

The `generate` node uses a detailed system prompt with mandatory rules to prevent common LLM mistakes:

1. **Mandatory Prefixes**: Exact URIs are provided to prevent mistakes like using `<http://w3id.org/elita/ontology#>` instead of `<http://w3id.org/elita/>`

2. **Translation Direction**: Always Italian → Dialect (never reverse)

3. **Multi-Dialect Variables**: Use DIFFERENT Italian lexical entry variables for each dialect

4. **Variable Typing**: Never reuse a variable for both URI and literal values

5. **SERVICE Block Rules**:
   - Only for CompL-it (definitions, semantic relations)
   - Never for ELITA emotions or dialect translations
   - Only valid endpoint: `https://klab.ilc.cnr.it/graphdb-compl-it/`

6. **LiITA-to-CompL-it Linking**: Use same variable name for natural joins (see Cross-Endpoint Linking section)

### Self-Correction Loop

When verification fails, the agent enters a refinement loop:

1. The failed query and error are recorded in `refinement_history`
2. On the next generation attempt, this history is included in the prompt
3. The LLM can learn from previous mistakes and try a different approach
4. After 3 attempts (configurable), the agent outputs the best effort

This is more effective than blind retries because the LLM sees the full context of what went wrong.

### Auto-Correction Features

The `execute` node includes targeted fixes that don't require full query regeneration:

**Variable Reuse Detection**

A common LLM mistake is reusing a variable for both a URI and a literal value:

```sparql
# WRONG - ?hypernymWord is both a URI (subject) and a literal (writtenRep)
?hypernymWord ontolex:sense ?sense ;
              ontolex:canonicalForm [ ontolex:writtenRep ?hypernymWord ] .
```

The agent detects this pattern before execution and returns an error that triggers regeneration with a clear explanation.

**Case-Sensitive Filter Auto-Fix**

When a query returns zero results, the agent checks for case-sensitive string filters:

```sparql
# Original (fails for "Rabbia")
FILTER(STR(?emotionLabel) = "rabbia")

# Auto-fixed (matches "Rabbia", "rabbia", "RABBIA")
FILTER(REGEX(STR(?emotionLabel), "^rabbia$", "i"))
```

If the fixed query returns results, it's used automatically without regeneration. This preserves the query structure while fixing the filter.

### Schema Exploration (Ontology Retrieval)

When a query returns zero results and the agent hasn't explored the schema yet, it uses **semantic search over an ontology catalog** to discover relevant classes and properties.

Unlike simply querying the endpoint for property URIs (which tells the LLM nothing about meaning), the ontology retriever provides:

1. **Descriptions**: What each property/class means (e.g., "hypernym: A term with a broader meaning")
2. **Domain/Range**: What types of subjects and objects the property connects
3. **Inverse properties**: Related properties (e.g., hypernym ↔ hyponym)
4. **SPARQL patterns**: Example usage

**Example**: For "What is a more general term for 'dog'?", the retriever finds:

```
### Property: lexinfo:hypernym
- URI: <http://www.lexinfo.net/ontology/3.0/lexinfo#hypernym>
- Description: A term with a broader meaning
- Domain (subject type): LexicalSense
- Range (object type): LexicalSense
- Inverse property: hyponym
- SPARQL pattern: ?subject lexinfo:hypernym ?object
```

This is far more useful than just returning `http://www.lexinfo.net/ontology/3.0/lexinfo#hypernym`.

**Implementation**:

```python
from nl2sparql.retrieval import OntologyRetriever

retriever = OntologyRetriever()
results = retriever.retrieve_properties("broader meaning", top_k=5)
prompt_text = retriever.format_for_prompt(results)
```

The ontology catalog (`data/ontology.json`) includes classes and properties from:
- OntoLex-Lemon (ontolex, vartrans, lime, synsem)
- LexInfo (lexinfo)
- SKOS (skos)
- LiLA (lila)
- ELITA (elita)
- MARL (marl)
- Dublin Core (dcterms)

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

    # Retrieval
    retrieved_examples: list[dict]
    relevant_constraints: str

    # Generation
    generated_sparql: str
    generation_attempts: int

    # Execution
    result_count: int
    execution_error: str | None

    # Refinement (accumulates across attempts)
    refinement_history: Annotated[list[dict], add]

    # Schema exploration
    discovered_properties: list[str]
    schema_context: str  # Detailed ontology descriptions for prompt
    schema_explored: bool

    # Output
    final_sparql: str
    confidence: float

    # Fallback
    first_valid_sparql: str  # First syntactically valid query (returned if refinement doesn't improve)
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

### LiITA-to-CompL-it Linking Pattern

When starting from a LiITA lemma and needing CompL-it data (definitions, semantic relations), use the **same variable name** for `writtenRep` in both blocks to create a natural join:

```sparql
# CORRECT - Same variable ?writtenRep creates automatic join
GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ;
           ontolex:writtenRep ?writtenRep .
}
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?writtenRep ] ;
          ontolex:sense [ skos:definition ?definition ] .
}

# WRONG - Causes "variable not assigned" error
GRAPH <http://liita.it/data> {
    ?lemma a lila:Lemma ; ontolex:writtenRep ?lilaRep .
}
SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
    ?word ontolex:canonicalForm [ ontolex:writtenRep ?complRep ] .
    FILTER(STR(?complRep) = STR(?lilaRep))  # ERROR: ?lilaRep not visible inside SERVICE!
}
```

Variables bound outside a SERVICE block are NOT visible inside for FILTER comparisons. The shared variable name approach avoids this limitation by letting the SPARQL engine handle the join.

---

## Component 6: MCP Server

The MCP (Model Context Protocol) server provides a third way to access NL2SPARQL capabilities, alongside the basic `NL2SPARQL` class and the agentic `NL2SPARQLAgent`.

### Why MCP?

MCP is a protocol that allows LLM clients (like Claude Desktop) to call external tools. The MCP server exposes NL2SPARQL's capabilities as tools that any MCP-compatible client can use.

**Key differences from the basic and agentic approaches:**

| Aspect | Basic NL2SPARQL | Agentic (LangGraph) | MCP Server |
|--------|-----------------|---------------------|------------|
| Control flow | User code | LangGraph workflow | External LLM client |
| LLM calls | Single (generate + optional fix) | Multiple (analyze, plan, generate, verify) | Configured on server |
| Iteration | Fixed retry loop | Self-correcting with refinement | Client-controlled |
| Integration | Python API | Python API | Claude Desktop, any MCP client |

### MCP Tools

The server exposes 9 tools:

**Translation:**
- `translate` - Full NL-to-SPARQL translation using the configured LLM

**Retrieval:**
- `infer_patterns` - Detect query patterns from natural language
- `retrieve_examples` - Get similar examples for few-shot learning
- `search_ontology` - Semantic search over ontology catalog
- `get_constraints` - Get domain constraints for patterns

**Validation:**
- `validate_sparql` - Comprehensive validation (syntax, semantic, endpoint)
- `execute_sparql` - Execute query against LiITA endpoint

**Utilities:**
- `fix_case_sensitivity` - Auto-fix case-sensitive filters
- `check_variable_reuse` - Detect variable reuse bugs

### MCP Resources

The server provides read-only resources:

- `liita://ontology/catalog` - Full ontology catalog (JSON)
- `liita://constraints/base` - Base system prompt and prefixes
- `liita://config` - Current server configuration

### Server Architecture

```
NL2SPARQLMCPServer
├── config: MCPConfig          # Provider, model, endpoint settings
├── translator: NL2SPARQL      # Lazy-loaded, for translate tool
├── hybrid_retriever           # Lazy-loaded, for retrieve_examples
├── ontology_retriever         # Lazy-loaded, for search_ontology
└── server: mcp.Server         # MCP protocol handler
    ├── list_tools()           # Tool definitions
    ├── call_tool()            # Tool execution
    ├── list_resources()       # Resource definitions
    └── read_resource()        # Resource content
```

Components are lazy-loaded to minimize startup time. The embedding models and retrievers are only initialized when first accessed.

### Configuration

The server is configured via CLI options or environment variables:

```bash
# CLI options
nl2sparql mcp serve --provider anthropic --model claude-sonnet-4-20250514

# Environment variables
NL2SPARQL_PROVIDER=anthropic
NL2SPARQL_MODEL=claude-sonnet-4-20250514
NL2SPARQL_ENDPOINT=https://liita.it/sparql
NL2SPARQL_TIMEOUT=30
```

### When to Use MCP

| Scenario | Recommended Approach |
|----------|---------------------|
| Python scripting | Basic `NL2SPARQL` |
| Complex queries needing iteration | Agentic `NL2SPARQLAgent` |
| Claude Desktop integration | MCP Server |
| External LLM client integration | MCP Server |
| Batch processing | Basic `NL2SPARQL` |
| Interactive exploration | MCP Server with Claude |

### Claude Desktop Example

When configured in Claude Desktop:

1. User asks: "Find Italian nouns expressing sadness"
2. Claude calls `translate(question="Find Italian nouns expressing sadness")`
3. Server uses configured LLM to generate SPARQL
4. Returns validated query with result count
5. Claude presents results to user

The MCP approach gives the external LLM full control over the workflow while the server provides domain expertise.

---

## Component 7: Gradio Web UI

The Gradio web interface provides a user-friendly way to interact with NL2SPARQL without writing code or using the command line.

### Why Gradio?

While the MCP server integrates with Claude Desktop, some users prefer:

1. **Direct browser access**: No desktop application needed
2. **Visual feedback**: See patterns, validation, and results in a structured layout
3. **Multiple tools in one interface**: Translate, analyze, search, and execute from one page
4. **Shareable links**: Create public URLs for demos or collaboration

### Architecture

The Gradio app uses the same components as the MCP server but calls them directly:

```
Gradio Web UI
├── Translate tab     → NL2SPARQL.translate()
├── Analyze tab       → handle_infer_patterns()
├── Search Ontology   → handle_search_ontology() + OntologyRetriever
├── Execute SPARQL    → handle_execute_sparql()
└── Fix Query         → handle_fix_case_sensitivity()
```

Unlike MCP, there's no external LLM orchestrating tool calls. The user directly selects which functionality to use via tabs.

### Tabs

| Tab | Description |
|-----|-------------|
| **Translate** | Full NL-to-SPARQL with patterns, validation, and sample results |
| **Analyze Patterns** | See detected patterns and complexity without generating SPARQL |
| **Search Ontology** | Find relevant properties and classes by semantic search |
| **Execute SPARQL** | Run queries directly against the LiITA endpoint |
| **Fix Query** | Auto-fix case-sensitive filters in existing queries |

### Usage

```bash
# Start with default provider (Mistral)
python scripts/gradio_app.py

# Specify provider and model
python scripts/gradio_app.py --provider ollama --model llama3

# Create shareable link
python scripts/gradio_app.py --provider mistral --share
```

Opens at `http://localhost:7860` by default.

### When to Use the Web UI

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick interactive testing | Web UI |
| Demos and presentations | Web UI with `--share` |
| Python integration | Basic `NL2SPARQL` |
| Self-correcting queries | Agentic `NL2SPARQLAgent` |
| Claude Desktop users | MCP Server |

---

## Limitations and Future Work

**Dataset size**: 140 examples is enough for common patterns but may miss rare query types. The system can be improved by adding more examples as users discover gaps.

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

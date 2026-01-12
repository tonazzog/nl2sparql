# NL2SPARQL

Natural Language to SPARQL translation for the [LiITA](https://www.liita.it/) (Linking Italian) linguistic knowledge base.

## Overview

NL2SPARQL translates natural language questions (in Italian or English) into SPARQL queries for querying the LiITA knowledge base. It uses a hybrid retrieval system combined with LLM-based query synthesis to generate SPARQL queries.

### Features

- **Multi-LLM Support**: Works with OpenAI, Anthropic, Mistral, Google Gemini, and local Ollama models
- **Hybrid Retrieval**: Combines semantic search (sentence transformers + FAISS), BM25, and pattern matching
- **Ontology-Aware**: Semantic search over ontology definitions to discover relevant properties and classes
- **Domain-Specific Constraints**: Built-in knowledge of LiITA's architecture (emotions, translations, semantic relations)
- **Query Validation**: Syntax checking, endpoint validation, and constraint verification
- **Auto-Fix**: Automatically fixes case-sensitive filters and detects variable reuse bugs
- **Bilingual**: Supports questions in both Italian and English
- **Agentic Mode**: LangGraph-powered agent with self-correction and ontology exploration
- **MCP Server**: Model Context Protocol server for integration with Claude Desktop and other MCP clients
- **Web UI**: Gradio-based web interface for interactive query generation

## Installation

```bash
# Basic installation
pip install liita-nl2sparql

# With specific LLM provider
pip install liita-nl2sparql[openai]      # For OpenAI
pip install liita-nl2sparql[anthropic]   # For Anthropic (Claude)
pip install liita-nl2sparql[mistral]     # For Mistral AI
pip install liita-nl2sparql[gemini]      # For Google Gemini
pip install liita-nl2sparql[ollama]      # For local Ollama models

# All providers
pip install liita-nl2sparql[all]

# Agentic mode (LangGraph-based)
pip install liita-nl2sparql[agent-openai]     # Agent with OpenAI
pip install liita-nl2sparql[agent-anthropic]  # Agent with Anthropic
pip install liita-nl2sparql[agent-all]        # Agent with all providers

# MCP server (for Claude Desktop integration)
pip install liita-nl2sparql[mcp-openai]       # MCP with OpenAI
pip install liita-nl2sparql[mcp-anthropic]    # MCP with Anthropic
pip install liita-nl2sparql[mcp-all]          # MCP with all providers

# Web UI (Gradio)
pip install liita-nl2sparql[ui]               # Gradio web interface
```

### Development Installation

```bash
git clone https://github.com/tonazzog/nl2sparql.git
cd nl2sparql
pip install -e ".[dev,all]"
```

## Configuration

Set your API key as an environment variable:

**Linux / macOS:**
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export MISTRAL_API_KEY="your-api-key"
export GEMINI_API_KEY="your-api-key"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key
set ANTHROPIC_API_KEY=your-api-key
set MISTRAL_API_KEY=your-api-key
set GEMINI_API_KEY=your-api-key
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:ANTHROPIC_API_KEY="your-api-key"
$env:MISTRAL_API_KEY="your-api-key"
$env:GEMINI_API_KEY="your-api-key"
```

Ollama runs locally and does not require an API key.

## Quick Start

For an interactive tutorial, see the [Quick Start Notebook](notebooks/quickstart.ipynb).

## Usage

### Command Line Interface

```bash
# Basic translation (Italian)
nl2sparql translate "Quali lemmi esprimono tristezza?"

# Basic translation (English)
nl2sparql translate "Find all words that express sadness"

# Specify provider and model
nl2sparql translate -p anthropic "What are the hyponyms of vehicle?"

# Save output to file
nl2sparql translate "Definition of love" -o query.sparql

# Verbose output with validation details
nl2sparql translate -V "Find the Sicilian translations of 'house'"

# Validate an existing query
nl2sparql validate query.sparql

# List available models
nl2sparql list-models

# Debug retrieval (see which examples are retrieved)
nl2sparql retrieve "What are the parts of the human body?"

# Agentic mode (self-correcting with LangGraph)
nl2sparql agent "Find all nouns expressing sadness"
nl2sparql agent -p anthropic "Trova aggettivi con traduzioni siciliane"
nl2sparql agent --stream "Complex query with step-by-step output"
nl2sparql agent-viz  # Show workflow diagram
```

### Python API

#### Simple Usage

```python
from nl2sparql import translate

# Italian
result = translate("Quali lemmi esprimono tristezza?")
print(result.sparql)

# English
result = translate("Find all nouns that express joy")
print(result.sparql)
```

#### Advanced Usage

```python
from nl2sparql import NL2SPARQL

# Initialize with specific provider
translator = NL2SPARQL(
    provider="openai",
    model="gpt-4.1",
    validate=True,
    fix_errors=True,
    max_retries=3
)

# Translate a question (Italian or English)
result = translator.translate("Find the Sicilian translations of 'casa'")

# Access results
print(result.sparql)                    # The generated SPARQL query
print(result.detected_patterns)         # Detected query patterns
print(result.confidence)                # Confidence score
print(result.validation.is_valid)       # Validation status
print(result.validation.result_count)   # Number of results from endpoint

# Check if query was auto-fixed
if result.was_fixed:
    print(f"Query was fixed after {result.fix_attempts} attempts")
```

#### Working with Retrieved Examples

```python
from nl2sparql import NL2SPARQL

translator = NL2SPARQL(provider="openai")
result = translator.translate("Parti del corpo umano")

# See which examples were retrieved for few-shot learning
for ex in result.retrieved_examples:
    print(f"Score: {ex.score:.3f}")
    print(f"Question: {ex.example.nl}")
    print(f"SPARQL: {ex.example.sparql[:100]}...")
```

#### Agentic Mode (Recommended for Complex Queries)

The agent uses a LangGraph workflow that can analyze, execute, verify, and self-correct queries:

```python
from nl2sparql.agent import NL2SPARQLAgent

# Initialize with provider and optional API key
agent = NL2SPARQLAgent(
    provider="openai",       # or "anthropic", "mistral", "gemini", "ollama"
    model="gpt-4.1",          # optional, uses provider default
    api_key="sk-...",        # optional, uses environment variable
)

# Translate a question
result = agent.translate(
    question="Trova tutti i sostantivi che esprimono tristezza",
    language="it",
    verbose=True
)

# Access results
print(result["sparql"])              # The generated SPARQL query
print(result["confidence"])          # Confidence score (0-1)
print(result["attempts"])            # Number of generation attempts
print(result["result_count"])        # Results from endpoint execution
print(result["is_valid"])            # Whether validation passed
print(result["detected_patterns"])   # Patterns identified
print(result["refinement_history"])  # Previous failed attempts (if any)
```

**Streaming mode** to see each step as it executes:

```python
agent = NL2SPARQLAgent(provider="anthropic")

for node_name, state in agent.stream("Find adjectives with Sicilian translations"):
    print(f"[{node_name}] completed")
    if node_name == "execute":
        print(f"  Results: {state.get('result_count', 0)}")
```

**Async support:**

```python
import asyncio

async def main():
    agent = NL2SPARQLAgent(provider="openai")
    result = await agent.atranslate("Trova verbi con emozioni positive")
    print(result["sparql"])

asyncio.run(main())
```

#### Ontology Retrieval

The agent uses semantic search over an ontology catalog to discover relevant properties and classes. This helps when the LLM needs to find the right vocabulary for a query:

```python
from nl2sparql.retrieval import OntologyRetriever

retriever = OntologyRetriever()

# Find properties related to "broader meaning" (e.g., for hypernyms)
results = retriever.retrieve_properties("broader meaning", top_k=5)

for item in results:
    print(f"{item.entry.prefix_local}: {item.entry.short_text}")
    # lexinfo:hypernym: A term with a broader meaning

# Format for LLM prompt
prompt_text = retriever.format_for_prompt(results)
```

The ontology catalog includes classes and properties from OntoLex-Lemon, LexInfo, SKOS, LiLA, ELITA, and other vocabularies used by LiITA.

### MCP Server (Claude Desktop Integration)

The MCP (Model Context Protocol) server exposes NL2SPARQL tools to MCP-compatible clients like Claude Desktop. This allows Claude to translate natural language questions to SPARQL queries for the LiITA knowledge base.

#### Starting the Server

```bash
# Start MCP server with default provider (OpenAI)
nl2sparql mcp serve

# Start with specific provider
nl2sparql mcp serve --provider anthropic
nl2sparql mcp serve --provider ollama --model llama3

# Show configuration
nl2sparql mcp config
```

#### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nl2sparql": {
      "command": "full path to python.exe in virtual environment",
      "args": [ "-m", "nl2sparql.mcp", "serve", "--provider", "mistral", "--api-key", "YOUR_KEY" ]
    }
  }
}
```

#### Available MCP Tools

| Tool | Description |
|------|-------------|
| `translate` | Full NL-to-SPARQL translation using configured LLM |
| `infer_patterns` | Detect query patterns from natural language |
| `retrieve_examples` | Get similar query examples for few-shot learning |
| `search_ontology` | Search ontology catalog for relevant properties/classes |
| `get_constraints` | Get domain constraints for detected patterns |
| `validate_sparql` | Validate SPARQL (syntax, semantic, endpoint) |
| `execute_sparql` | Execute query against LiITA endpoint |
| `fix_case_sensitivity` | Auto-fix case-sensitive filters |
| `check_variable_reuse` | Detect variable reuse bugs |

#### Python API

```python
import asyncio
from nl2sparql.mcp import NL2SPARQLMCPServer
from nl2sparql.mcp.server import MCPConfig

# Configure and run the server
config = MCPConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
)
server = NL2SPARQLMCPServer(config)
asyncio.run(server.run())
```
### Web UI (Gradio)

A Gradio-based web interface for interactive SPARQL query generation. This provides a user-friendly way to translate natural language questions without using the command line or writing Python code.

#### Starting the Web UI

```bash
# Start with default provider (Mistral)
python scripts/gradio_app.py

# Start with specific provider
python scripts/gradio_app.py --provider ollama --model llama3
python scripts/gradio_app.py --provider anthropic --api-key YOUR_KEY

# Create a public shareable link
python scripts/gradio_app.py --provider mistral --share

# Custom port
python scripts/gradio_app.py --port 8080
```

The UI opens at `http://localhost:7860` by default.

#### Features

| Tab | Description |
|-----|-------------|
| **Translate** | Convert natural language questions to SPARQL with validation and sample results |
| **Analyze Patterns** | Detect query patterns without generating SPARQL |
| **Retreive Examples** | Retrieve similar query examples from the dataset |
| **Search Ontology** | Browse and search classes/properties in the LiITA ontology |
| **Execute SPARQL** | Run SPARQL queries directly against the LiITA endpoint |
| **Fix Query** | Auto-fix case-sensitive string filters |

#### Screenshot

The Translate tab shows:
- Input field for natural language questions
- Generated SPARQL query with syntax highlighting
- Detected patterns and confidence score
- Validation status (syntax, endpoint)
- Sample results from the query

### Web UI (Gradio Agent)

An agent-based Gradio web interface with dual-LLM architecture. Watch in real-time as the orchestrator decides which tools to call and the translator generates SPARQL queries.

#### Architecture

The app uses two LLMs:
- **Orchestrator**: Decides which tools to call (can be a smaller/cheaper model)
- **Translator**: Expert SPARQL generator (used by the `generate_sparql` tool)

#### Starting the Web UI

```bash
# Start with default provider (Mistral for both LLMs)
python scripts/gradio_app_agent.py

# Use different providers for orchestrator and translator
python scripts/gradio_app_agent.py \
    --orchestrator-provider anthropic --orchestrator-model claude-3-haiku-20240307 \
    --translator-provider openai --translator-model gpt-4.1

# With explicit API keys
python scripts/gradio_app_agent.py \
    -op mistral -ok "YOUR_MISTRAL_KEY" \
    -tp openai -tk "YOUR_OPENAI_KEY"

# Create a public shareable link
python scripts/gradio_app_agent.py --share

# Custom port
python scripts/gradio_app_agent.py --port 8080
```

The UI opens at `http://localhost:7861` by default.

#### Command-line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--orchestrator-provider` | `-op` | Orchestrator LLM provider (default: mistral) |
| `--orchestrator-model` | `-om` | Orchestrator model (uses provider default) |
| `--orchestrator-api-key` | `-ok` | Orchestrator API key (uses env var if not set) |
| `--translator-provider` | `-tp` | Translator LLM provider (default: mistral) |
| `--translator-model` | `-tm` | Translator model (uses provider default) |
| `--translator-api-key` | `-tk` | Translator API key (uses env var if not set) |
| `--share` | | Create a public shareable link |
| `--port` | | Port to run on (default: 7861) |

#### Available Tools

The orchestrator can call these tools:

| Tool | Description |
|------|-------------|
| `infer_patterns` | Detect query patterns (EMOTION_LEXICON, TRANSLATION, etc.) |
| `retrieve_examples` | Find similar SPARQL examples for few-shot learning |
| `search_ontology` | Search for RDF classes and properties |
| `get_constraints` | Get domain-specific rules for detected patterns |
| `generate_sparql` | Generate SPARQL using the translator LLM |
| `validate_sparql` | Validate query (syntax, semantic, endpoint) |
| `execute_sparql` | Execute query and return results |
| `fix_query` | Auto-fix common issues (case sensitivity, SERVICE clauses) |
| `final_answer` | Return the final SPARQL query to the user |

#### Real-time Tool Calls

The UI streams tool calls as they happen:
- **Running** indicators show which tool is currently executing
- **Completed** indicators show results from each tool
- The SPARQL output updates as soon as a query is generated
- Validation results and fixes are shown in real-time

## Supported Query Types

| Query Type | Example Question | Description |
|------------|------------------|-------------|
| Emotion | "Quali lemmi esprimono tristezza?" | Queries ELITA emotion annotations |
| Translation | "Traduzioni siciliane di casa" | Queries dialect translations (Sicilian, Parmigiano) |
| Definition | "Definizione di amore" | Queries CompL-it sense definitions |
| Semantic Relations | "Iperonimi di cane" | Queries hypernyms, hyponyms, meronyms |
| POS Filter | "Trova tutti i verbi" | Filters by part of speech |
| Morphological | "Lemmi che iniziano con 'pre'" | Pattern matching on word forms |
| Compositional | "Tutti gli animali velenosi" | Complex multi-step reasoning |

## Project Structure

```
nl2sparql/
├── scripts/
│   ├── gradio_app.py          # Simple Gradio web UI
│   ├── gradio_app_agent.py    # Agent-based Gradio UI (dual LLM)
│   └── test_mcp_tools.py      # Direct tool testing
├── notebooks/
│   └── quickstart.ipynb       # Interactive tutorial
├── __init__.py                # Public API
├── __main__.py                # Entry point for python -m nl2sparql
├── cli.py                     # Command-line interface
├── config.py                  # Configuration management
├── agent/                     # Agentic LangGraph workflow
│   ├── __init__.py            # Public API (NL2SPARQLAgent)
│   ├── state.py               # State definition for workflow
│   ├── nodes.py               # Node implementations (analyze, generate, verify, etc.)
│   └── graph.py               # LangGraph workflow definition
├── mcp/                       # MCP (Model Context Protocol) server
│   ├── __init__.py            # Public API (NL2SPARQLMCPServer)
│   ├── __main__.py            # Entry point for python -m nl2sparql.mcp
│   ├── server.py              # MCP server implementation
│   ├── tools.py               # Tool handler implementations
│   └── resources.py           # Resource providers
├── constraints/               # Domain-specific prompts and validation
│   ├── __init__.py            # Public API for constraints
│   ├── base.py                # Core SPARQL patterns and system prompt
│   ├── emotion.py             # ELITA emotion constraints
│   ├── translation.py         # Dialect translation constraints
│   ├── semantic.py            # CompL-it semantic constraints
│   ├── lexical_relation.py    # Synonym/antonym constraints
│   ├── multi_entry.py         # Multi-entry pattern validation
│   ├── compositional.py       # Complex query reasoning
│   └── prompt_builder.py      # Dynamic prompt construction
├── retrieval/                 # Hybrid retrieval system
│   ├── __init__.py            # Public API for retrieval
│   ├── hybrid_retriever.py    # Main retriever combining all methods
│   ├── ontology_retriever.py  # Semantic search over ontology definitions
│   ├── embeddings.py          # Sentence transformers + FAISS
│   ├── bm25.py                # BM25 with pattern boosting
│   └── patterns.py            # Query pattern inference (keyword + semantic)
├── generation/                # Query synthesis
│   ├── __init__.py            # Public API for generation
│   ├── synthesizer.py         # Main NL2SPARQL class
│   └── adapters.py            # Query adaptation utilities
├── llm/                       # LLM provider abstraction
│   ├── __init__.py            # Public API for LLM clients
│   ├── base.py                # Abstract client interface
│   ├── openai_client.py       # OpenAI implementation
│   ├── anthropic_client.py    # Anthropic implementation
│   ├── mistral_client.py      # Mistral implementation
│   ├── gemini_client.py       # Google Gemini implementation
│   └── ollama_client.py       # Ollama implementation
├── validation/                # Query validation
│   ├── __init__.py            # Public API for validation
│   ├── syntax.py              # rdflib syntax validation
│   ├── endpoint.py            # SPARQL endpoint validation
│   └── semantic.py            # Constraint-based validation
├── evaluation/                # Evaluation framework
│   ├── __init__.py            # Public API for evaluation
│   ├── evaluate.py            # Test runner and metrics
│   └── batch_evaluate.py      # Multi-model comparison
├── synthetic/                 # Synthetic data generation
│   ├── __init__.py            # Public API for synthetic generation
│   └── generator.py           # Training data generator
└── data/
    ├── sparql_queries_examples.json  # Example queries dataset
    ├── test_dataset.json             # Evaluation test cases
    └── ontology.json                 # Ontology catalog (classes & properties)
```

## LiITA Knowledge Base Architecture

The system understands LiITA's multi-source architecture:

- **Main LiITA**: Lemmas, POS, morphology (`GRAPH <http://liita.it/data>`)
- **ELITA**: Emotion annotations (`GRAPH <http://w3id.org/elita>`)
- **Dialect Translations**: Sicilian, Parmigiano (via `vartrans:translatableAs`)
- **CompL-it**: Senses, definitions, semantic relations (`SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/>`)

## Available Models

| Provider | Default Model | Other Models |
|----------|--------------|--------------|
| OpenAI | gpt-4.1-mini | gpt-5.2, gpt-4.1, gpt-4.1-nano, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | claude-sonnet-4-20250514 | claude-opus-4-20250514, claude-3-5-haiku-20241022 |
| Mistral | mistral-large-latest | mistral-medium-latest, mistral-small-latest |
| Gemini | gemini-pro | gemini-pro-vision |
| Ollama | llama3 | mistral, codellama, phi3 |

## Evaluation

The package includes a test framework for systematic evaluation of single models and batch comparison of multiple models.

### Single Model Evaluation

```bash
# Full evaluation with default settings
nl2sparql evaluate

# Evaluate with specific provider
nl2sparql evaluate -p anthropic

# Test only single-pattern queries
nl2sparql evaluate -c single_pattern

# Test specific patterns
nl2sparql evaluate --pattern EMOTION_LEXICON --pattern TRANSLATION

# Save results to file (includes generated SPARQL queries)
nl2sparql evaluate -o report.json
```

### Batch Model Comparison

Compare multiple LLM providers and models:

```bash
# Quick comparison (GPT-4o-mini vs Claude 3.5 Haiku)
nl2sparql batch-evaluate -p quick

# Compare all OpenAI models
nl2sparql batch-evaluate -p openai -o ./reports

# Compare default models from all providers
nl2sparql batch-evaluate -p all_defaults -c comparison.json

# Custom model selection
nl2sparql batch-evaluate --provider openai --provider anthropic
```

Available presets:
- `quick` - Fast comparison with smaller models
- `openai` - All OpenAI models
- `anthropic` - All Anthropic models
- `mistral` - All Mistral models
- `all_defaults` - Default model from each provider

### Python API

```python
from nl2sparql import NL2SPARQL
from nl2sparql.evaluation import (
    evaluate_dataset,
    print_report,
    save_report,
    # Batch evaluation
    ModelConfig,
    run_batch_evaluation,
    create_comparison_report,
    print_comparison,
    PRESETS,
)

# Single model evaluation
translator = NL2SPARQL(provider="openai")
report = evaluate_dataset(translator, language="it")
print_report(report)
save_report(report, "report.json")  # Includes generated SPARQL queries

# Batch model comparison
configs = [
    ModelConfig("openai", "gpt-4.1", "GPT-4.1"),
    ModelConfig("anthropic", "claude-sonnet-4-20250514", "Claude Sonnet"),
]
results = run_batch_evaluation(configs, output_dir="./reports")
comparison = create_comparison_report(results, "comparison.json")
print_comparison(comparison)
```

### Metrics

- **Syntax validity**: Percentage of queries that parse correctly
- **Endpoint success**: Percentage of queries that execute without errors
- **Component score**: Percentage of expected SPARQL components present
- **Pattern detection accuracy**: How well the system identifies query types

See [docs/evaluation.md](docs/evaluation.md) for detailed documentation.

## Synthetic Data Generation

Generate training data for fine-tuning custom LLMs on NL2SPARQL:

```bash
# Generate synthetic (NL, SPARQL) pairs
nl2sparql generate-synthetic -o training_data.jsonl

# With options
nl2sparql generate-synthetic -o data.jsonl -n 10 -m 500 -f alpaca
```

The generator creates validated training pairs by:
1. Generating NL variations of seed examples
2. Creating pattern combination questions
3. Validating all SPARQL against the endpoint

Output formats: `jsonl`, `json`, `alpaca`, `sharegpt`, `hf` (HuggingFace)

See [docs/synthetic_data.md](docs/synthetic_data.md) for detailed documentation.

## License

MIT License - see [LICENSE](LICENSE) for details.


# NL2SPARQL

Natural Language to SPARQL translation for the [LiITA](https://www.liita.it/) (Linking Italian) linguistic knowledge base.

## Overview

NL2SPARQL translates natural language questions (in Italian or English) into SPARQL queries for querying the LiITA knowledge base. It uses a hybrid retrieval system combined with LLM-based query synthesis to generate SPARQL queries.

### Features

- **Multi-LLM Support**: Works with OpenAI, Anthropic, Mistral, Google Gemini, and local Ollama models
- **Hybrid Retrieval**: Combines semantic search (sentence transformers + FAISS), BM25, and pattern matching
- **Domain-Specific Constraints**: Built-in knowledge of LiITA's architecture (emotions, translations, semantic relations)
- **Query Validation**: Syntax checking, endpoint validation, and semantic constraint verification
- **Auto-Fix**: Automatically attempts to fix invalid queries
- **Bilingual**: Supports questions in both Italian and English

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
├── notebooks/
│   └── quickstart.ipynb     # Interactive tutorial
├── __init__.py              # Public API
├── cli.py                   # Command-line interface
├── config.py                # Configuration management
├── constraints/             # Domain-specific prompts and validation
│   ├── base.py              # Core SPARQL patterns and system prompt
│   ├── emotion.py           # ELITA emotion constraints
│   ├── translation.py       # Dialect translation constraints
│   ├── semantic.py          # CompL-it semantic constraints
│   ├── compositional.py     # Complex query reasoning
│   └── prompt_builder.py    # Dynamic prompt construction
├── retrieval/               # Hybrid retrieval system
│   ├── hybrid_retriever.py  # Main retriever combining all methods
│   ├── embeddings.py        # Sentence transformers + FAISS
│   ├── bm25.py              # BM25 with pattern boosting
│   └── patterns.py          # Query pattern inference
├── generation/              # Query synthesis
│   ├── synthesizer.py       # Main NL2SPARQL class
│   └── adapters.py          # Query adaptation utilities
├── llm/                     # LLM provider abstraction
│   ├── base.py              # Abstract client interface
│   ├── openai_client.py     # OpenAI implementation
│   ├── anthropic_client.py  # Anthropic implementation
│   ├── mistral_client.py    # Mistral implementation
│   ├── gemini_client.py     # Google Gemini implementation
│   └── ollama_client.py     # Ollama implementation
├── validation/              # Query validation
│   ├── syntax.py            # rdflib syntax validation
│   ├── endpoint.py          # SPARQL endpoint validation
│   └── semantic.py          # Constraint-based validation
├── evaluation/              # Evaluation framework
│   ├── evaluate.py          # Test runner and metrics
│   └── batch_evaluate.py    # Multi-model comparison
└── data/
    ├── sparql_queries_final.json  # Training dataset
    └── test_dataset.json          # Evaluation test cases
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
| OpenAI | gpt-4.1-mini | gpt-5.2, gpt-4.1, gpt-4.1-nano, gpt-4o,  gpt-4o-mini,  gpt-4-turbo,  gpt-3.5-turbo |
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
    ModelConfig("openai", "gpt-4o", "GPT-4o"),
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

## License

MIT License - see [LICENSE](LICENSE) for details.


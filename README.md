# NL2SPARQL

Natural Language to SPARQL translation for the [LiITA](https://lila-erc.eu/) (Linked Italian) linguistic knowledge base.

## Overview

NL2SPARQL translates natural language questions (in Italian or English) into SPARQL queries for querying the LiITA knowledge base. It uses a hybrid retrieval system combined with LLM-based query synthesis to generate accurate SPARQL queries.

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
pip install nl2sparql

# With specific LLM provider
pip install nl2sparql[openai]      # For OpenAI (GPT-4, GPT-5.1)
pip install nl2sparql[anthropic]   # For Anthropic (Claude)
pip install nl2sparql[mistral]     # For Mistral AI
pip install nl2sparql[gemini]      # For Google Gemini
pip install nl2sparql[ollama]      # For local Ollama models

# All providers
pip install nl2sparql[all]
```

### Development Installation

```bash
git clone https://github.com/tonazzog/nl2sparql.git
cd nl2sparql
pip install -e ".[dev,all]"
```

## Configuration

Set your API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# Mistral
export MISTRAL_API_KEY="your-api-key"

# Google Gemini
export GEMINI_API_KEY="your-api-key"

# Ollama (no API key needed - runs locally)
```

## Usage

### Command Line Interface

```bash
# Basic translation
nl2sparql translate "Quali lemmi esprimono tristezza?"

# Specify provider and model
nl2sparql translate -p anthropic -m claude-sonnet-4-20250514 "Trova traduzioni siciliane di casa"

# Save output to file
nl2sparql translate "Definizione di amore" -o query.sparql

# Verbose output with validation details
nl2sparql translate -V "Quali sono gli iperonimi di cane?"

# Validate an existing query
nl2sparql validate query.sparql

# List available models
nl2sparql list-models

# Debug retrieval (see which examples are retrieved)
nl2sparql retrieve "Trova verbi che esprimono gioia"
```

### Python API

#### Simple Usage

```python
from nl2sparql import translate

result = translate("Quali lemmi esprimono tristezza?")
print(result.sparql)
```

#### Advanced Usage

```python
from nl2sparql import NL2SPARQL

# Initialize with specific provider
translator = NL2SPARQL(
    provider="openai",
    model="gpt-4o",
    validate=True,
    fix_errors=True,
    max_retries=3
)

# Translate a question
result = translator.translate("Trova le traduzioni siciliane di 'casa'")

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
└── data/
    └── sparql_queries_final.json  # Training dataset
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
| OpenAI | gpt-5.1 | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| Anthropic | claude-sonnet-4-20250514 | claude-opus-4-20250514, claude-3-5-haiku-20241022 |
| Mistral | mistral-large-latest | mistral-medium-latest, mistral-small-latest |
| Gemini | gemini-pro | gemini-pro-vision |
| Ollama | llama3 | mistral, codellama, phi3 |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project was developed as part of research on the [LiLA - Linking Latin](https://lila-erc.eu/) project, funded by the European Research Council (ERC).

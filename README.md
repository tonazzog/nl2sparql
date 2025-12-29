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


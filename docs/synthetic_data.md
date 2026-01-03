# Synthetic Data Generation

This document describes the synthetic data generation module for creating training data to fine-tune LLMs on NL2SPARQL translation.

## Overview

The synthetic data generator creates validated (Natural Language question, SPARQL query) pairs by:

1. **Variation Generation**: Creating paraphrases and entity variations of seed examples
2. **Pattern Combination**: Combining multiple query patterns into complex questions
3. **Validation**: Ensuring all generated SPARQL executes correctly on the endpoint

This enables fine-tuning smaller, faster, or domain-specific LLMs for the NL2SPARQL task.

## Installation

```bash
# With HuggingFace datasets support
pip install liita-nl2sparql[synthetic]

# Or minimal (uses agent dependencies)
pip install liita-nl2sparql[agent-openai]
```

## Quick Start

### Command Line

```bash
# Basic generation
nl2sparql generate-synthetic -o training_data.jsonl

# Generate 500 pairs with 10 variations per seed
nl2sparql generate-synthetic -o data.jsonl -n 10 -m 500

# Use Anthropic for generation
nl2sparql generate-synthetic -o data.jsonl -p anthropic -m claude-3-5-haiku-20241022

# Export in Alpaca format for fine-tuning
nl2sparql generate-synthetic -o alpaca_train.json -f alpaca
```

### Python API

```python
from nl2sparql.synthetic import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(
    provider="openai",
    model="gpt-4.1-mini",
    use_agent=True,  # Use agentic workflow for better quality
)

# Generate dataset
pairs, stats = generator.generate_dataset(
    num_variations_per_seed=5,   # NL variations per seed example
    num_combinations=10,          # Pattern combination questions
    min_results=1,                # Minimum query results required
    max_pairs=500,                # Maximum total pairs
    include_seeds=True,           # Include original examples
)

# Save in desired format
generator.save_dataset(pairs, "training_data.jsonl", format="jsonl")

print(f"Generated {len(pairs)} training pairs")
print(f"Success rate: {stats.successful_pairs}/{stats.total_attempts}")
```

## Generation Methods

### 1. Variation Generation

For each seed example, the generator creates NL variations that preserve the query semantics:

**Original**: "Quali lemmi esprimono tristezza?"
**Variations**:
- "Trova parole che indicano tristezza"
- "Quali sono i termini associati alla tristezza?"
- "Lemmi che esprimono sentimenti di tristezza"
- "Parole italiane che denotano tristezza"

The LLM generates variations by:
- Paraphrasing (different wording, same meaning)
- Entity substitution (tristezza → gioia, paura, rabbia)
- Style variation (formal/informal, question/imperative)
- Structure changes (active/passive, different verbs)

### 2. Pattern Combination

Creates complex questions by combining patterns from multiple seeds:

**Input patterns**: EMOTION_LEXICON + TRANSLATION
**Generated question**: "Quali sono le traduzioni siciliane delle parole che esprimono tristezza?"

**Input patterns**: SEMANTIC_RELATION + POS_FILTER
**Generated question**: "Trova tutti i verbi che sono iponimi di 'muovere'"

### 3. Validation

Every generated SPARQL query is validated:

1. **Syntax check**: Parses correctly with rdflib
2. **Endpoint execution**: Runs without errors on LiITA
3. **Result check**: Returns at least `min_results` results

Only validated pairs are included in the output.

## CLI Reference

```bash
nl2sparql generate-synthetic [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output file path |

### Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --num-variations` | 5 | NL variations per seed example |
| `-c, --num-combinations` | 10 | Pattern combination questions |
| `-m, --max-pairs` | None | Maximum total pairs to generate |
| `--min-results` | 1 | Minimum query results required |
| `--include-seeds/--no-seeds` | True | Include original seed examples |

### Provider Options

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --provider` | openai | LLM provider |
| `--model` | (default) | Model name |
| `--no-agent` | False | Use standard translator instead of agent |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --format` | jsonl | Output format |
| `-l, --language` | it | Language for questions |
| `-q, --quiet` | False | Suppress progress output |

### Examples

```bash
# Quick test with 50 pairs
nl2sparql generate-synthetic -o test.jsonl -m 50 -n 2

# Full generation with Anthropic
nl2sparql generate-synthetic -o train.jsonl -n 10 -c 20 -p anthropic

# English questions
nl2sparql generate-synthetic -o train_en.jsonl -l en

# Alpaca format for LLaMA fine-tuning
nl2sparql generate-synthetic -o alpaca.json -f alpaca --no-seeds

# HuggingFace format
nl2sparql generate-synthetic -o hf_dataset -f hf
```

## Output Formats

### JSONL (Default)

One JSON object per line, with metadata:

```json
{"instruction": "Traduci questa domanda...", "input": "Quali lemmi esprimono tristezza?", "output": "SELECT ?lemma...", "patterns": ["EMOTION_LEXICON"], "result_count": 42, "source": "Q001", "method": "variation"}
{"instruction": "Traduci questa domanda...", "input": "Trova parole che indicano gioia", "output": "SELECT ?lemma...", "patterns": ["EMOTION_LEXICON"], "result_count": 38, "source": "Q001", "method": "variation"}
```

### JSON

JSON array with metadata:

```json
[
  {
    "instruction": "Traduci questa domanda in linguaggio naturale in una query SPARQL per il knowledge base LiITA.",
    "input": "Quali lemmi esprimono tristezza?",
    "output": "SELECT ?lemma WHERE { ... }",
    "patterns": ["EMOTION_LEXICON"],
    "result_count": 42,
    "source": "Q001",
    "method": "variation"
  }
]
```

### Alpaca

Standard Alpaca format for instruction fine-tuning:

```json
[
  {
    "instruction": "Translate this natural language question to SPARQL for the LiITA linguistic knowledge base.",
    "input": "Quali lemmi esprimono tristezza?",
    "output": "SELECT ?lemma WHERE { ... }"
  }
]
```

### ShareGPT

Conversation format for chat model fine-tuning:

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "Translate this question to SPARQL for LiITA:\n\nQuali lemmi esprimono tristezza?"
      },
      {
        "from": "gpt",
        "value": "SELECT ?lemma WHERE { ... }"
      }
    ]
  }
]
```

### HuggingFace (hf)

HuggingFace datasets format (saves as directory):

```python
from datasets import load_from_disk

dataset = load_from_disk("hf_dataset")
print(dataset[0])
# {'instruction': '...', 'input': '...', 'output': '...', 'patterns': [...]}
```

## Python API Reference

### SyntheticPair

```python
@dataclass
class SyntheticPair:
    nl_question: str           # Natural language question
    sparql_query: str          # Generated SPARQL
    patterns: list[str]        # Query patterns (e.g., ["EMOTION_LEXICON"])
    source_example_id: str     # ID of seed example (if variation)
    result_count: int          # Number of results from endpoint
    is_valid: bool             # Validation passed
    language: str              # "it" or "en"
    generation_method: str     # "seed", "variation", or "combination"
```

### SyntheticDataGenerator

```python
class SyntheticDataGenerator:
    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        use_agent: bool = True,
        seed_dataset_path: str = None,
        api_key: str = None,
    ):
        """Initialize the generator."""

    def generate_dataset(
        self,
        num_variations_per_seed: int = 5,
        num_combinations: int = 10,
        min_results: int = 1,
        max_pairs: int = None,
        language: str = "it",
        include_seeds: bool = True,
        verbose: bool = True,
    ) -> tuple[list[SyntheticPair], GenerationStats]:
        """Generate a complete synthetic dataset."""

    def save_dataset(
        self,
        pairs: list[SyntheticPair],
        output_path: str,
        format: str = "jsonl",
        include_metadata: bool = True,
    ) -> None:
        """Save dataset to file."""

    @staticmethod
    def load_dataset(path: str) -> list[SyntheticPair]:
        """Load a previously saved dataset."""
```

### GenerationStats

```python
@dataclass
class GenerationStats:
    total_attempts: int        # Total generation attempts
    successful_pairs: int      # Successfully validated pairs
    failed_validation: int     # Failed validation
    failed_generation: int     # Failed SPARQL generation
    total_time: float          # Total generation time (seconds)
    pairs_by_pattern: dict     # Count of pairs per pattern
```

## Cost Estimation

Approximate costs for generating synthetic data:

| Seeds | Variations | Combinations | Total Pairs | GPT-4.1-mini | Claude Haiku |
|-------|------------|--------------|-------------|--------------|--------------|
| 50 | 5 | 10 | ~260 | $2-4 | $1-2 |
| 50 | 10 | 20 | ~520 | $5-10 | $2-5 |
| 50 | 20 | 50 | ~1050 | $10-20 | $5-10 |

Costs include:
- NL variation generation (~500 tokens per seed)
- SPARQL generation (~2000 tokens per question with agent)
- Pattern combination generation (~800 tokens each)

## Fine-Tuning Workflow

### 1. Generate Training Data

```bash
# Generate 500 high-quality pairs
nl2sparql generate-synthetic -o train.jsonl -n 10 -m 500 -p anthropic
```

### 2. Convert to Fine-Tuning Format

```bash
# For OpenAI fine-tuning
nl2sparql generate-synthetic -o openai_train.jsonl -f jsonl

# For LLaMA/Alpaca
nl2sparql generate-synthetic -o alpaca_train.json -f alpaca

# For HuggingFace
nl2sparql generate-synthetic -o hf_dataset -f hf
```

### 3. Fine-Tune

**OpenAI**:
```bash
openai api fine_tunes.create -t openai_train.jsonl -m gpt-4.1-mini
```

**Local with HuggingFace**:
```python
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, Trainer

dataset = load_from_disk("hf_dataset")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# ... training code
```

### 4. Evaluate Fine-Tuned Model

```python
from nl2sparql.evaluation import evaluate_dataset

# Test your fine-tuned model using a custom translator
report = evaluate_dataset(your_finetuned_translator)
print_report(report)
```

## Best Practices

1. **Quality over Quantity**: Fewer high-quality pairs (validated, diverse) are better than many low-quality ones

2. **Use the Agent**: Set `use_agent=True` for better SPARQL generation with self-correction

3. **Include Seeds**: Keep `include_seeds=True` to ensure the model sees the original high-quality examples

4. **Balance Patterns**: Check `stats.pairs_by_pattern` to ensure coverage of all query types

5. **Validate Manually**: Spot-check a sample of generated pairs before fine-tuning

6. **Test on Held-Out Data**: Use the evaluation framework to test on the test dataset after fine-tuning

## Troubleshooting

### Low Success Rate

If many pairs fail validation:
- Increase `min_results` tolerance (set to 0 to accept empty results)
- Check if the LiITA endpoint is responsive
- Use a more capable model (gpt-4.1 instead of gpt-4.1-mini)

### Slow Generation

- Use `--no-agent` for faster (but lower quality) generation
- Use a faster model (gpt-4.1-mini, claude-3-5-haiku)
- Reduce `num_variations_per_seed`

### Memory Issues

- Process in batches with `max_pairs`
- Use JSONL format for streaming writes

### Rate Limits

- Add delays between requests (not yet implemented)
- Use multiple API keys
- Switch to local models (Ollama)

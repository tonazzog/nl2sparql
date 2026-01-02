# Evaluation Framework

This document describes the evaluation framework for NL2SPARQL, including the test dataset structure, evaluation metrics, and how to run evaluations.

## Test Dataset

The test dataset (`nl2sparql/data/test_dataset.json`) contains structured test cases designed to cover all query patterns and their combinations.

### Dataset Structure

```json
{
  "metadata": {
    "description": "...",
    "version": "1.0",
    "patterns_covered": ["EMOTION_LEXICON", "TRANSLATION", ...]
  },
  "test_cases": [
    {
      "id": "T001",
      "category": "single_pattern",
      "patterns": ["EMOTION_LEXICON"],
      "nl_it": "Quali lemmi esprimono tristezza?",
      "nl_en": "Which lemmas express sadness?",
      "description": "Basic emotion query",
      "expected_components": ["GRAPH <http://w3id.org/elita>", "elita:HasEmotion"]
    }
  ]
}
```

### Test Categories

| Category | Description | Count |
|----------|-------------|-------|
| `single_pattern` | Tests one pattern in isolation | 16 |
| `combination_2` | Tests two patterns together | 12 |
| `combination_3` | Tests three patterns together | 4 |
| `complex` | Complex multi-pattern queries | 3 |

### Patterns Covered

| Pattern | Description | Example Query |
|---------|-------------|---------------|
| EMOTION_LEXICON | Emotion annotations from ELITA | "Words expressing sadness" |
| TRANSLATION | Sicilian/Parmigiano translations | "Sicilian translation of 'house'" |
| MULTI_TRANSLATION | Both dialects | "Translations in both dialects" |
| SENSE_DEFINITION | Word definitions from CompL-it | "Definition of 'love'" |
| SENSE_COUNT | Counting word senses | "How many senses does 'bank' have?" |
| SEMANTIC_RELATION | Hypernyms, hyponyms, meronyms | "Hyponyms of 'vehicle'" |
| POS_FILTER | Part of speech filtering | "Find all verbs" |
| MORPHO_REGEX | Morphological patterns | "Words ending with 'tion'" |
| COUNT_ENTITIES | Counting queries | "How many lemmas?" |
| META_GRAPH | Graph exploration | "List all graphs" |
| SERVICE_INTEGRATION | CompL-it federated queries | (implicit in SENSE_DEFINITION) |
| COMPOSITIONAL | Multi-step reasoning | "All venomous animals" |

## Evaluation Metrics

### Primary Metrics

1. **Syntax Validity Rate**
   - Percentage of generated queries that parse correctly
   - Measured using rdflib SPARQL parser
   - Formula: `syntax_valid / total_tests`

2. **Endpoint Execution Success Rate**
   - Percentage of queries that execute without errors on the LiITA endpoint
   - Does not require results, just successful execution
   - Formula: `endpoint_valid / total_tests`

3. **Component Matching Score**
   - Percentage of expected SPARQL components found in generated query
   - Checks for presence of key patterns (graph URIs, properties, etc.)
   - Formula: `matched_components / expected_components`

4. **Pattern Detection Accuracy**
   - How well the system identifies the required query patterns
   - Checks if expected patterns are subset of detected patterns
   - Formula: `correct_detections / total_tests`

### Aggregate Metrics

- **Average Generation Time**: Mean time to generate a query
- **Average Component Score**: Mean component matching across all tests
- **Results by Category**: Breakdown of success rates by test category
- **Results by Pattern**: Breakdown of success rates by query pattern

## Running Evaluations

### Command Line Interface

```bash
# Full evaluation (standard translator)
nl2sparql evaluate

# Evaluate with the agentic workflow
nl2sparql evaluate --agent

# With specific LLM provider
nl2sparql evaluate -p anthropic -m claude-sonnet-4-20250514

# Agent evaluation with specific provider
nl2sparql evaluate --agent -p openai -m gpt-4.1

# Test in English instead of Italian
nl2sparql evaluate -l en

# Filter by category
nl2sparql evaluate -c single_pattern
nl2sparql evaluate -c single_pattern -c combination_2

# Filter by pattern
nl2sparql evaluate --pattern EMOTION_LEXICON
nl2sparql evaluate --pattern EMOTION_LEXICON --pattern TRANSLATION

# Skip endpoint validation (faster)
nl2sparql evaluate --no-endpoint

# Save report to JSON
nl2sparql evaluate -o evaluation_report.json
```

### Python API

```python
from nl2sparql import NL2SPARQL
from nl2sparql.evaluation import (
    evaluate_dataset,
    evaluate_single,
    load_test_dataset,
    print_report,
    save_report,
)

# Initialize translator
translator = NL2SPARQL(
    provider="openai",
    model="gpt-4.1",
    validate=True,
    fix_errors=True,
)

# Full evaluation
report = evaluate_dataset(translator, language="it")
print_report(report)

# Filter by category
report = evaluate_dataset(
    translator,
    categories=["single_pattern", "combination_2"],
)

# Filter by pattern
report = evaluate_dataset(
    translator,
    patterns=["EMOTION_LEXICON", "SEMANTIC_RELATION"],
)

# Save report
save_report(report, "report.json")
```

### Evaluating the Agent

To evaluate the agentic workflow instead of the standard translator, use the `AgentAdapter`:

```python
from nl2sparql.agent import NL2SPARQLAgent
from nl2sparql.evaluation import (
    AgentAdapter,
    evaluate_dataset,
    print_report,
)

# Initialize agent
agent = NL2SPARQLAgent(
    provider="openai",
    model="gpt-4.1-mini",
)

# Wrap with adapter for evaluation
adapter = AgentAdapter(agent)

# Run evaluation (same API as standard translator)
report = evaluate_dataset(adapter, language="it")
print_report(report)
```

### Single Test Case

```python
from nl2sparql.evaluation import evaluate_single, load_test_dataset

test_data = load_test_dataset()
test_case = test_data["test_cases"][0]

result = evaluate_single(test_case, translator, language="it")

print(f"Test: {result.test_id}")
print(f"Syntax valid: {result.syntax_valid}")
print(f"Endpoint valid: {result.endpoint_valid}")
print(f"Component score: {result.component_score:.2%}")
print(f"Missing components: {result.missing_components}")
```

---

## Batch Model Comparison

The batch evaluation feature allows comparing multiple LLM providers and models systematically.

### Command Line Interface

```bash
# Quick comparison (GPT-4.1-mini vs Claude 3.5 Haiku)
nl2sparql batch-evaluate -p quick

# Compare using the agentic workflow
nl2sparql batch-evaluate --agent -p quick

# Compare all OpenAI models
nl2sparql batch-evaluate -p openai

# Compare all Anthropic models
nl2sparql batch-evaluate -p anthropic

# Compare default models from all providers
nl2sparql batch-evaluate -p all_defaults

# Save individual reports and comparison
nl2sparql batch-evaluate -p openai -o ./reports -c comparison.json

# Custom model selection
nl2sparql batch-evaluate --provider openai --provider anthropic --model gpt-4.1 --model claude-sonnet-4-20250514

# Skip endpoint validation for faster results
nl2sparql batch-evaluate -p quick --no-endpoint
```

### Available Presets

| Preset | Models Included | Use Case |
|--------|-----------------|----------|
| `quick` | GPT-4.1-mini, Claude 3.5 Haiku | Fast initial comparison |
| `openai` | GPT-4.1, GPT-4.1-mini, GPT-4.1-nano | Compare OpenAI tiers |
| `anthropic` | Claude Sonnet 4, Claude 3.5 Haiku | Compare Anthropic tiers |
| `mistral` | Mistral Large, Mistral Small | Compare Mistral tiers |
| `all_defaults` | Default from each provider | Cross-provider comparison |

### Python API

```python
from nl2sparql.evaluation import (
    ModelConfig,
    BatchResult,
    run_batch_evaluation,
    create_comparison_report,
    print_comparison,
    PRESETS,
)

# Use a preset configuration
results = run_batch_evaluation(
    configs=PRESETS["quick"],
    language="it",
    output_dir="./reports",  # Save individual reports
)

# Or define custom configurations
configs = [
    ModelConfig("openai", "gpt-4.1", "GPT-4.1"),
    ModelConfig("openai", "gpt-4.1-mini", "GPT-4.1-mini"),
    ModelConfig("anthropic", "claude-sonnet-4-20250514", "Claude Sonnet 4"),
    ModelConfig("anthropic", "claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
]

results = run_batch_evaluation(
    configs=configs,
    language="it",
    validate_endpoint=True,
    output_dir="./reports",
    verbose=True,
)

# Use the agentic workflow instead of standard translator
results = run_batch_evaluation(
    configs=PRESETS["quick"],
    language="it",
    use_agent=True,  # Enable agent mode
)

# Generate and display comparison
comparison = create_comparison_report(results, "comparison.json")
print_comparison(comparison)
```

### Comparison Report Output

```
======================================================================
MODEL COMPARISON REPORT
======================================================================

Models evaluated: 4
Timestamp: 2025-01-15T10:30:00

----------------------------------------------------------------------
Model                          Syntax       Endpoint     Component    Time
----------------------------------------------------------------------
GPT-4.1                        91.4%        80.0%        85.7%        2.34s
GPT-4.1-mini                   88.6%        74.3%        82.1%        1.12s
Claude Sonnet 4                94.3%        85.7%        89.2%        2.87s
Claude 3.5 Haiku               85.7%        71.4%        78.5%        0.95s
----------------------------------------------------------------------

Rankings:

  By Syntax Validity:
    1. Claude Sonnet 4: 94.3%
    2. GPT-4.1: 91.4%
    3. GPT-4.1-mini: 88.6%
    4. Claude 3.5 Haiku: 85.7%

  By Endpoint Success:
    1. Claude Sonnet 4: 85.7%
    2. GPT-4.1: 80.0%
    ...

  By Generation Speed (fastest):
    1. Claude 3.5 Haiku: 0.95s
    2. GPT-4o-mini: 1.12s
    ...
```

### Output Files

When using `--output-dir` (`-o`), individual reports are saved for each model:

```
reports/
├── report_GPT-4.1.json
├── report_GPT-4.1-mini.json
├── report_Claude_Sonnet_4.json
└── report_Claude_3.5_Haiku.json
```

Each report includes the full evaluation results with generated SPARQL queries, which can be manually tested on the LiITA endpoint.

When using `--comparison` (`-c`), a summary JSON is saved with:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "models_evaluated": 4,
  "models": [
    {
      "name": "GPT-4.1",
      "provider": "openai",
      "syntax_valid_rate": 0.914,
      "endpoint_valid_rate": 0.800,
      "avg_component_score": 0.857,
      "avg_generation_time": 2.34,
      "by_category": { ... }
    }
  ],
  "comparison": {
    "by_syntax_valid": [...],
    "by_endpoint_valid": [...],
    "by_component_score": [...],
    "by_generation_time": [...]
  }
}
```

---

## Interpreting Results

### Example Report Output

```
============================================================
NL2SPARQL EVALUATION REPORT
============================================================

Overall Results:
  Total tests:           35
  Successful generations: 35 (100.0%)
  Syntax valid:          32 (91.4%)
  Endpoint valid:        28 (80.0%)

Aggregate Metrics:
  Avg generation time:   2.34s
  Avg component score:   85.7%
  Pattern detection acc: 94.3%

Results by Category:
  single_pattern: 15/16 (93.8%)
  combination_2: 11/12 (91.7%)
  combination_3: 3/4 (75.0%)
  complex: 3/3 (100.0%)

Results by Pattern:
  EMOTION_LEXICON: 8/8 (100.0%), component score: 92.5%
  TRANSLATION: 6/7 (85.7%), component score: 81.3%
  SEMANTIC_RELATION: 7/8 (87.5%), component score: 78.9%
  ...
```

### What the Metrics Mean

- **High syntax validity (>90%)**: The LLM generates structurally correct SPARQL
- **Lower endpoint success**: Queries may be syntactically correct but semantically wrong for LiITA
- **Component score gaps**: Indicates which SPARQL patterns the system struggles with
- **Pattern-specific issues**: Highlights which query types need more examples or better constraints

### Common Failure Patterns

1. **SERVICE block issues**: Filters referencing external variables, linking inside SERVICE
2. **Wrong graph locations**: Properties queried in wrong GRAPH or outside GRAPH
3. **Semantic relation direction**: Confusing hypernym/hyponym direction
4. **Missing FILTER(STR())**: Direct literal matching instead of string filter

## Adding Test Cases

To add new test cases, edit `nl2sparql/data/test_dataset.json`:

```json
{
  "id": "T036",
  "category": "combination_2",
  "patterns": ["EMOTION_LEXICON", "MORPHO_REGEX"],
  "nl_it": "Trova parole che esprimono paura e iniziano con 't'",
  "nl_en": "Find words expressing fear that start with 't'",
  "description": "Emotion + prefix filter",
  "expected_components": ["elita:HasEmotion", "REGEX", "^t", "paura"]
}
```

Guidelines:
- Use unique IDs (T001, T002, ...)
- Include both Italian and English versions
- List all relevant patterns
- Include key SPARQL components to check
- Add description for clarity

## Comparison with Benchmarks

For context, here's how NL2SPARQL metrics relate to standard benchmarks:

| Benchmark | Metric | Typical Scores |
|-----------|--------|----------------|
| Spider4SPARQL | Execution accuracy | ~45% (state-of-the-art) |
| SPARQL-LLM | F1 Score | Variable by domain |
| LargeRDFBench | Precision/Recall/F1 | Endpoint-specific |

Our component matching score is not directly comparable to execution accuracy (which requires result set comparison), but provides insight into structural correctness.

## Future Improvements

Potential enhancements to the evaluation framework:

1. **Execution accuracy**: Compare result sets with ground truth queries
2. **Result F1**: Precision/recall on returned bindings
3. **Cost tracking**: Token usage per query for LLM cost analysis
4. ~~**Cross-model comparison**: Automated comparison across providers~~ (Implemented in v0.2.0)

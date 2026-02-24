# Evaluation Framework

This document describes the evaluation framework for NL2SPARQL, including the test dataset structure, evaluation metrics, and how to run evaluations.

## Test Dataset

The test dataset (`nl2sparql/data/test_dataset.json`) contains structured test cases designed to cover all query patterns and their combinations.

### Dataset Structure

```json
{
  "metadata": {
    "description": "...",
    "version": "2.0",
    "total_examples": 100,
    "patterns_covered": ["EMOTION_LEXICON", "TRANSLATION", ...]
  },
  "test_cases": [
    {
      "id": 2001,
      "category": "complex",
      "patterns": ["EMOTION_LEXICON", "TRANSLATION"],
      "nl_it": "Quali lemmi esprimono tristezza con traduzioni siciliane?",
      "nl_en": "Which lemmas express sadness with Sicilian translations?",
      "sparql": "PREFIX ...\nSELECT ...",
      "answer_variables": {
        "primary": ["italianWord"],
        "secondary": [],
        "aggregates": [],
        "numeric": []
      },
      "expected_components": ["GRAPH <http://w3id.org/elita>", "elita:HasEmotion"]
    }
  ]
}
```

### Test Categories

The 100 test cases (all in English) are grouped by semantic complexity:

| Category | Description | Count |
|----------|-------------|-------|
| `complex` | Multi-pattern queries combining 2–4 patterns | 56 |
| `semantic_combined` | Semantic relation + SERVICE federated queries | 29 |
| `emotion` | Emotion annotation queries (ELITA) | 9 |
| `translation` | Dialect translation queries (Sicilian/Parmigiano) | 6 |

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

1. **F1 Score on Answers** *(primary metric)*
   - Executes both gold and predicted SPARQL against the LiITA endpoint and compares result sets
   - Precision = fraction of predicted answers that are correct; Recall = fraction of gold answers retrieved
   - F1 = harmonic mean of precision and recall
   - This is the most meaningful metric: two structurally different queries can be semantically equivalent
   - See [docs/f1_evaluator.md](f1_evaluator.md) for full details

2. **Syntax Validity Rate**
   - Percentage of generated queries that parse correctly
   - Measured using rdflib SPARQL parser
   - Formula: `syntax_valid / total_tests`

3. **Endpoint Execution Success Rate**
   - Percentage of queries that execute without errors on the LiITA endpoint
   - Does not require results, just successful execution
   - Formula: `endpoint_valid / total_tests`

4. **Component Matching Score**
   - Percentage of expected SPARQL components found in generated query
   - Checks for presence of key patterns (graph URIs, properties, etc.)
   - Formula: `matched_components / expected_components`

5. **Pattern Detection Accuracy**
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
nl2sparql evaluate -p anthropic -m claude-sonnet-4-6

# Agent evaluation with specific provider
nl2sparql evaluate --agent -p openai -m gpt-5.2

# Test in English instead of Italian
nl2sparql evaluate -l en

# Filter by category
nl2sparql evaluate -c complex
nl2sparql evaluate -c complex -c semantic_combined

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
    model="gpt-5.2",
    validate=True,
    fix_errors=True,
)

# Full evaluation
report = evaluate_dataset(translator, language="it")
print_report(report)

# Filter by category
report = evaluate_dataset(
    translator,
    categories=["complex", "semantic_combined"],
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
    model="gpt-5-mini",
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
# Quick comparison (GPT-5-mini vs Claude Haiku 4.5)
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
nl2sparql batch-evaluate --provider openai --provider anthropic --model gpt-5.2 --model claude-sonnet-4-6

# Skip endpoint validation for faster results
nl2sparql batch-evaluate -p quick --no-endpoint
```

### Available Presets

| Preset | Models Included | Use Case |
|--------|-----------------|----------|
| `quick` | GPT-5-mini, Claude Haiku 4.5 | Fast initial comparison |
| `openai` | GPT-5.2, GPT-5, GPT-5-mini | Compare OpenAI tiers |
| `anthropic` | Claude Sonnet 4.6, Claude Haiku 4.5 | Compare Anthropic tiers |
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
    ModelConfig("openai", "gpt-5.2", "GPT-5.2"),
    ModelConfig("openai", "gpt-5-mini", "GPT-5-mini"),
    ModelConfig("anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ModelConfig("anthropic", "claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]

results = run_batch_evaluation(
    configs=configs,
    language="it",
    validate_endpoint=True,
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

--------------------------------------------------------------------------------
Model                          Avg F1     Syntax     Endpoint   Component  Time
--------------------------------------------------------------------------------
GPT-5.2                        0.7012     91.4%      80.0%      85.7%      2.34s
GPT-5-mini                     0.6543     88.6%      74.3%      82.1%      1.12s
Claude Sonnet 4.6              0.7224     94.3%      85.7%      89.2%      2.87s
Claude Haiku 4.5               0.6201     85.7%      71.4%      78.5%      0.95s
--------------------------------------------------------------------------------

Rankings:

  By Avg F1 Score (primary metric):
    1. Claude Sonnet 4.6: 0.7224
    2. GPT-5.2: 0.7012
    3. GPT-5-mini: 0.6543
    4. Claude Haiku 4.5: 0.6201

  By Syntax Validity:
    1. Claude Sonnet 4.6: 94.3%
    2. GPT-5.2: 91.4%
    3. GPT-5-mini: 88.6%
    4. Claude Haiku 4.5: 85.7%

  By Endpoint Success:
    1. Claude Sonnet 4.6: 85.7%
    2. GPT-5.2: 80.0%
    ...

  By Generation Speed (fastest):
    1. Claude Haiku 4.5: 0.95s
    2. GPT-5-mini: 1.12s
    ...
```

### Output Files

Individual reports are always saved to `reports/` by default (one per model):

```
reports/
├── report_GPT-5-2.json
├── report_GPT-5-mini.json
├── report_Claude_Sonnet_4-6.json
└── report_Claude_Haiku_4-5.json
```

Each report includes the full evaluation results with generated SPARQL queries, which can be manually tested on the LiITA endpoint.

When using `--comparison` (`-c`), a summary JSON is also saved with:

```json
{
  "timestamp": "2026-01-15T10:30:00",
  "models_evaluated": 4,
  "models": [
    {
      "name": "Claude Sonnet 4.6",
      "provider": "anthropic",
      "syntax_valid_rate": 0.943,
      "endpoint_valid_rate": 0.857,
      "avg_component_score": 0.892,
      "avg_generation_time": 2.87,
      "avg_f1_score": 0.7224,
      "f1_evaluated_count": 100,
      "by_category": { ... }
    }
  ],
  "comparison": {
    "by_f1_score": [...],
    "by_syntax_valid": [...],
    "by_endpoint_valid": [...],
    "by_component_score": [...],
    "by_generation_time": [...]
  }
}
```

---

## F1 Score Evaluation

For end-to-end answer accuracy, use `scripts/run_f1_evaluation.py`. This executes both the gold and predicted SPARQL queries against the LiITA endpoint and computes precision, recall, and F1 on the result sets.

```bash
# Run F1 evaluation with Anthropic, English questions, LIMIT stripping
python scripts/run_f1_evaluation.py \
    --provider anthropic \
    --model claude-sonnet-4-6 \
    --language en \
    --strip-limit

```

The F1 report is saved to `reports/f1_report_<provider>_<model>.json` and includes per-test-case results, breakdowns by category and pattern, and score distributions.

See [docs/f1_evaluator.md](f1_evaluator.md) for the full API reference and explanation of the metric.

---

## Interpreting Results

### Example Report Output

```
============================================================
NL2SPARQL EVALUATION REPORT
============================================================

Overall Results:
  Total tests:           100
  Successful generations: 100 (100.0%)
  Syntax valid:          96 (96.0%)
  Endpoint valid:        89 (89.0%)

Aggregate Metrics:
  Avg generation time:   2.34s
  Avg component score:   85.7%
  Pattern detection acc: 94.0%
  Avg F1 score:          0.7224 (n=100)

Results by Category:
  complex:           50/56 (89.3%)
  semantic_combined: 26/29 (89.7%)
  emotion:           9/9  (100.0%)
  translation:       6/6  (100.0%)

Results by Pattern:
  EMOTION_LEXICON: 100.0%, component score: 92.5%
  TRANSLATION: 100.0%, component score: 88.3%
  SEMANTIC_RELATION: 85.7%, component score: 78.9%
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
  "id": 2200,
  "category": "complex",
  "patterns": ["EMOTION_LEXICON", "MORPHO_REGEX"],
  "nl_it": "Trova parole che esprimono paura e iniziano con 't'",
  "nl_en": "Find words expressing fear that start with 't'",
  "sparql": "PREFIX ...\nSELECT ?italianWord WHERE { ... }",
  "answer_variables": {
    "primary": ["italianWord"],
    "secondary": [],
    "aggregates": [],
    "numeric": []
  },
  "description": "Emotion + prefix filter",
  "expected_components": ["elita:HasEmotion", "REGEX", "^t", "paura"]
}
```

Guidelines:
- Use unique numeric IDs (current max is ~2137; use 2200+ for new cases)
- Include both Italian and English NL questions
- Provide the gold SPARQL query and classify the answer variables
- List all relevant patterns
- Include key SPARQL components to check
- Add description for clarity
- Validate the gold query against the LiITA endpoint before adding

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

1. **Cost tracking**: Token usage per query for LLM cost analysis
2. **Finer failure categorisation**: Automated tagging of failure types (wrong graph, wrong property, direction error, etc.)
3. **Incremental evaluation**: Skip re-running test cases that haven't changed when adding new ones

# F1 Score on Answers Evaluator

This document describes the F1 Score on Answers evaluation metric used to measure how well a predicted SPARQL query retrieves the same results as the gold (reference) query.

## Overview

Unlike structural metrics (syntax validity, component matching) that compare SPARQL query *text*, the F1 evaluator compares query *results*. Both the gold and predicted SPARQL queries are executed against the LiITA endpoint, and their result sets are compared using precision, recall, and F1 score.

This is the primary evaluation metric for the NL2SPARQL system because two structurally different SPARQL queries can return identical results, and the user ultimately cares about getting the right answers.

## How F1 is Computed

### Step-by-step Pipeline

For each test case, the evaluator performs these steps:

```
1. Execute gold SPARQL against LiITA endpoint
       → gold result set (list of variable bindings)

2. Execute predicted SPARQL against LiITA endpoint
       → predicted result set (list of variable bindings)

3. Build variable mapping (gold var names → predicted var names)

4. Classify variables into answer categories
       → primary, secondary, aggregates, numeric, uris

5. Extract answer tuples from both result sets
       → using only primary variables (the "answers")

6. Compare tuples as multisets (Counter)
       → TP = sum of min(gold_count, pred_count) for each matching tuple

7. Compute Precision, Recall, F1
```

### The Metric Formally

Given:
- **G** = multiset of answer tuples from the gold result set
- **P** = multiset of answer tuples from the predicted result set

For each distinct tuple **t**:

```
TP = Σ min(G(t), P(t))     for all distinct tuples t
```

Where `G(t)` and `P(t)` are the multiplicities (counts) of tuple `t` in each multiset.

```
Precision = TP / |P|       (what fraction of predicted answers are correct)
Recall    = TP / |G|       (what fraction of gold answers were found)
F1        = 2 * P * R / (P + R)
```

### What is an "Answer Tuple"?

An answer tuple is the combination of values from the **primary answer variables** in a single result row. Secondary and URI variables are excluded: secondary variables are internal identifiers (e.g., `?italianLemma`, `?sense`) that differ between query structures even when the answers are the same.

**Example**: For a query with `SELECT ?italianWord ?emotionLabel ?sicilianWord`:

| ?italianWord | ?emotionLabel | ?sicilianWord |
|---|---|---|
| casa | Joy | casa |
| acqua | Fear | acqua |
| casa | Joy | casa |

The answer tuples would be:
```
("casa", "Joy", "casa")  — count 2
("acqua", "Fear", "acqua")  — count 1
```

If the predicted query returns the same rows, TP=3, Precision=1.0, Recall=1.0, F1=1.0. If the predicted query misses "acqua", TP=2, Recall=2/3, etc.

## Variable Classification

Not all variables in a SELECT clause are treated equally. The evaluator classifies them into categories based on `ANSWER_VARIABLE_CATEGORIES` (defined in `synthetic/test_generator.py`):

| Category | Role in F1 | Examples |
|---|---|---|
| **primary** | **Included** in answer tuples | `italianWord`, `sicilianWord`, `definition`, `emotionLabel`, `hypernymWord` |
| **secondary** | Excluded from answer tuples | `pos`, `sense`, `lexEntryLabel`, `gender` |
| **aggregates** | Exact value match (separate score) | `count`, `avgPolarityValue`, `wordCount` |
| **numeric** | Normalized to 4 decimal places | `polarityValue`, `value`, `length` |
| **uris** | Excluded from comparison | `emotionLexEntry`, `graph`, `form`, `lexicalEntry` |

### Why Exclude Secondary and URI Variables?

**Secondary variables** like `?pos`, `?sense`, or `?italianLemma` are often present in the gold query to enable joins but are absent from (or named differently in) a correct predicted query. Including them in the tuple comparison would make F1=0 for any query that omits an internal join variable — even when the final answers are identical.

**URI variables** like `?emotionLexEntry` or `?lexicalEntry` are internal identifiers (e.g., `http://liita.it/data/id/...`). Two queries can return different intermediate URIs but reach the same final answer. Comparing URIs would penalize structurally different but semantically correct queries.

### Aggregate Handling

For queries that only return aggregate values (e.g., `SELECT (COUNT(?lemma) AS ?count)`):

1. There are no primary/secondary variables → no tuple comparison
2. The first row of each result set is compared
3. Each aggregate variable is matched by exact value (after numeric normalization)
4. The aggregate score (matches / total aggregates) is used as the F1 score

This means a COUNT query that returns `42` from both gold and predicted gets F1=1.0, while one that returns `42` vs `41` gets F1=0.0.

## Variable Mapping

A key challenge is that the gold and predicted queries may use **different variable names** for the same concept. The evaluator handles this with a three-phase mapping strategy.

### Phase 1: Category + Position Matching

Variables are classified by category, then matched by position within each category.

```
Gold:       SELECT ?italianWord ?emotionLabel ?sicilianWord
Predicted:  SELECT ?word        ?emotion      ?translation
```

Both `?italianWord` and `?word` are classified as **primary**. They appear at position 0 in their category list, so they are mapped: `italianWord → word`.

### Phase 2: Direct Name Match

For any unmapped gold variables, check if the predicted query uses the exact same name.

```
Gold:       ?definition
Predicted:  ?definition    → direct match
```

### Phase 3: Substring Similarity

For remaining unmapped variables, find the best match based on shared substrings (minimum length 3).

```
Gold:       ?senseDefinition
Predicted:  ?def              → shares "def" (length 3) → mapped
```

### Mapping Example

```
Gold query:      SELECT ?italianWord ?emotionLabel ?sicilianWord ?pos
Predicted query: SELECT ?word ?emotion ?sicWord ?pos

Mapping result:
  italianWord  → word          (Phase 1: both primary, position 0)
  emotionLabel → emotion       (Phase 1: both primary, position 1)
  sicilianWord → sicWord       (Phase 3: shared substring "sic")
  pos          → pos           (Phase 2: direct name match)
```

## Special Cases

| Condition | Result |
|---|---|
| Gold query fails to execute | Test case **skipped** (not counted) |
| Predicted query fails to execute | F1 = 0.0 |
| Both return empty results | F1 = 1.0 (perfect match on empty) |
| No primary/secondary variables | Row count comparison (TP = min of both counts) |
| Aggregate-only query | Aggregate exact match score used as F1 |

## Quick Start

### Python API

```python
from nl2sparql.evaluation import F1Evaluator

evaluator = F1Evaluator()

# Evaluate a single test case
result = evaluator.evaluate_single(
    gold_sparql="PREFIX ... SELECT ?italianWord WHERE { ... }",
    predicted_sparql="PREFIX ... SELECT ?word WHERE { ... }",
    answer_variables={
        "primary": ["italianWord"],
        "secondary": [],
        "aggregates": [],
        "numeric": [],
    },
    test_id=1,
)

print(f"Precision: {result.precision:.3f}")
print(f"Recall:    {result.recall:.3f}")
print(f"F1:        {result.f1:.3f}")
print(f"Gold rows: {result.gold_count}")
print(f"Pred rows: {result.predicted_count}")
print(f"TP:        {result.true_positives}")
```

### Evaluate a Full Dataset

```python
import json
from nl2sparql.evaluation import F1Evaluator

# Load test dataset
with open("nl2sparql/data/test_dataset.json") as f:
    test_data = json.load(f)

evaluator = F1Evaluator()
report = evaluator.evaluate_dataset(test_data)

print(f"Avg F1:        {report.avg_f1:.3f}")
print(f"Macro F1:      {report.macro_f1:.3f}")
print(f"Evaluated:     {report.total_evaluated}")
print(f"Skipped:       {report.total_skipped}")
```

### Evaluate with a Translator

```python
from nl2sparql.evaluation import F1Evaluator

evaluator = F1Evaluator()

# Pass a translator to generate predictions from NL questions
report = evaluator.evaluate_dataset(
    test_data,
    translator=my_translator,    # must have translate(question) method
    language="en",
)
```

### Baseline Self-Check

When no translator is provided, the evaluator uses the gold query as the prediction. This should always yield F1=1.0 for every test case (useful for verifying the test dataset):

```python
evaluator = F1Evaluator(strip_predicted_limit=True)
report = evaluator.evaluate_dataset(test_data)  # no translator = gold vs gold
assert report.avg_f1 == 1.0, "Baseline self-check failed!"
```

> **Note**: If gold queries contain `LIMIT` clauses, the self-check requires `strip_predicted_limit=True` (which strips LIMIT/OFFSET from both gold and predicted before execution). Without it, executing gold vs gold with LIMIT would still return the same truncated set and yield F1=1.0, but running with a translator that omits LIMIT would produce recall < 1. Using `strip_predicted_limit=True` is the recommended default for all evaluations.

### Integration with Component Evaluation

The F1 evaluator can be used alongside the existing component-based evaluation:

```python
from nl2sparql.evaluation import evaluate_dataset, F1Evaluator

f1_eval = F1Evaluator()
report = evaluate_dataset(
    translator=my_translator,
    f1_evaluator=f1_eval,    # adds f1_score to each TestResult
)

# Now report.test_results[i].f1_score is populated
for r in report.test_results:
    print(f"Test {r.test_id}: component={r.component_score:.2f}, F1={r.f1_score:.3f}")
```

## API Reference

### F1Evaluator

```python
class F1Evaluator:
    def __init__(
        self,
        endpoint: str = LIITA_ENDPOINT,          # SPARQL endpoint URL
        timeout: int = 60,                        # Query timeout in seconds
        max_results: int = 10000,                 # Safety limit per query
        cache_gold_results: bool = True,          # Cache gold results across calls
        strip_predicted_limit: bool = False,      # Strip LIMIT/OFFSET from both
                                                  # gold and predicted before execution
    ): ...

    def evaluate_single(
        self,
        gold_sparql: str,                    # Reference SPARQL
        predicted_sparql: str,               # Predicted SPARQL
        answer_variables: dict,              # {primary, secondary, aggregates, numeric}
        test_id: int = 0,
    ) -> F1Result: ...

    def evaluate_dataset(
        self,
        test_data: dict,                     # {"test_cases": [...]}
        translator=None,                     # Optional: generates predictions from NL
        language: str = "en",                # NL question language key
    ) -> F1Report: ...

    def prefetch_gold_results(
        self,
        test_data: dict,                     # Pre-execute all gold queries into cache
    ) -> None: ...
```

### F1Result

```python
@dataclass
class F1Result:
    test_id: int
    precision: float                          # TP / |predicted|
    recall: float                             # TP / |gold|
    f1: float                                 # harmonic mean
    gold_count: int                           # total gold tuples
    predicted_count: int                      # total predicted tuples
    true_positives: int                       # matching tuples (multiset intersection)
    aggregate_score: Optional[float] = None   # matches/total for aggregate vars
    aggregate_details: dict                   # per-variable gold vs predicted
    variable_mapping: dict                    # gold var name → predicted var name
    gold_error: Optional[str] = None          # if gold query failed
    predicted_error: Optional[str] = None     # if predicted query failed
```

### F1Report

```python
@dataclass
class F1Report:
    total_evaluated: int
    total_skipped: int                        # gold query failures
    avg_precision: float                      # micro-average precision
    avg_recall: float                         # micro-average recall
    avg_f1: float                             # micro-average F1
    macro_f1: float                           # average of per-category F1 averages
    f1_by_category: dict                      # {category: {avg_f1, count}}
    f1_by_pattern: dict                       # {pattern: {avg_f1, count}}
    results: list[F1Result]                   # per-test-case details
```

### Helper Functions

```python
# Execute a query and get full results (not just 5 like validate_endpoint)
execute_query_full(sparql, endpoint, timeout, max_results) -> QueryExecutionResult

# Normalize a value for comparison (strips whitespace, rounds numerics)
normalize_value(value, is_numeric=False) -> str

# Build gold → predicted variable name mapping
build_variable_mapping(gold_answer_vars, predicted_sparql) -> dict[str, str]

# Core F1 computation on result sets
compute_f1(gold_results, predicted_results, answer_vars, variable_mapping, numeric_vars) -> F1Result
```

## Value Normalization

Before comparing values, the evaluator normalizes them:

| Type | Normalization | Example |
|---|---|---|
| String | Strip whitespace | `" casa "` → `"casa"` |
| Numeric | Parse as float, round to 4 decimals | `"0.12345678"` → `"0.1235"` |
| Integer | Parse as float, round | `"42.0"` → `"42.0"` |

Numeric normalization is applied to variables classified as `numeric` or `aggregates`. This handles minor floating-point differences in polarity values, counts returned as decimals, etc.

## Micro vs Macro F1

The report provides two aggregate F1 scores:

**avg_f1 (micro-average)**: Simple average of F1 scores across all test cases. Every test case has equal weight regardless of category.

```
avg_f1 = (1/N) * Σ F1_i     for i = 1..N test cases
```

**macro_f1**: Average of per-category F1 averages. Each category has equal weight regardless of how many test cases it contains.

```
macro_f1 = (1/C) * Σ avg_F1_c     for c = 1..C categories
```

Macro F1 prevents categories with many test cases (e.g., "emotion") from dominating the aggregate score.

## Gold Result Caching

By default, the evaluator caches gold query results by test ID. This means:

- If you run `evaluate_dataset` multiple times (e.g., comparing models), gold queries are only executed once
- You can explicitly pre-populate the cache with `prefetch_gold_results()`
- Set `cache_gold_results=False` to disable caching (useful if the endpoint data changes)

## Worked Example

Consider a test case asking "Which Italian words express joy and have a Sicilian translation?"

**Gold query** returns:

| ?italianWord | ?emotionLabel | ?sicilianWord |
|---|---|---|
| amore | Joy | amuri |
| sole | Joy | suli |
| festa | Joy | festa |

**Predicted query** returns:

| ?word | ?emotion | ?translation |
|---|---|---|
| amore | Joy | amuri |
| sole | Joy | suli |
| vita | Joy | vita |

**Step 1**: Variable mapping (Phase 1, by category+position):
```
italianWord  → word
emotionLabel → emotion
sicilianWord → translation
```

**Step 2**: Extract answer tuples:
```
Gold tuples:      {("amore","Joy","amuri"):1, ("sole","Joy","suli"):1, ("festa","Joy","festa"):1}
Predicted tuples: {("amore","Joy","amuri"):1, ("sole","Joy","suli"):1, ("vita","Joy","vita"):1}
```

**Step 3**: Multiset intersection:
```
("amore","Joy","amuri"): min(1,1) = 1
("sole","Joy","suli"):   min(1,1) = 1
("festa","Joy","festa"): min(1,0) = 0  (not in predicted)
("vita","Joy","vita"):   min(0,1) = 0  (not in gold)

TP = 2
```

**Step 4**: Compute scores:
```
Precision = 2/3 = 0.667   (2 of 3 predicted are correct)
Recall    = 2/3 = 0.667   (2 of 3 gold answers were found)
F1        = 2 * 0.667 * 0.667 / (0.667 + 0.667) = 0.667
```

## LIMIT/OFFSET Stripping

Many gold queries in the test dataset contain `LIMIT` clauses (e.g., `LIMIT 10`). When executed literally, a gold query returning only 10 rows would make recall deflated for any predicted query that returns the full result set. Similarly, if the model adds a `LIMIT` that the gold doesn't have, precision appears perfect but recall collapses.

The solution is to strip `LIMIT` and `OFFSET` from **both** gold and predicted queries before executing them:

```python
evaluator = F1Evaluator(strip_predicted_limit=True)
```

When `strip_predicted_limit=True`:
- `LIMIT N` and `OFFSET N` are removed from the predicted query before execution
- `LIMIT N` and `OFFSET N` are also removed from gold queries when fetched (including during `prefetch_gold_results`)
- This ensures both sides return their full result set for a fair precision/recall comparison

**Recommendation**: Always use `strip_predicted_limit=True` for end-to-end evaluations where a translator generates the predictions. Use the default (`False`) only when you have manual control over both queries and neither contains LIMIT.

## Troubleshooting

### Gold Query Failures

If many test cases are skipped due to gold query failures:
- Check that the LiITA endpoint is reachable
- Some complex queries may time out — increase `timeout` (default: 60s)
- The endpoint data may have changed since test cases were created

### F1 = 0 Despite Correct Results

Common causes:
- **Variable mapping failure**: The predicted query uses variable names that can't be mapped. Check `result.variable_mapping` to debug.
- **Secondary variables in gold SELECT**: If the gold query selects secondary variables (e.g., `?italianLemma`, `?sense`) that the predicted query correctly omits, the variable mapping may fail or produce misaligned tuples. This is handled automatically — secondary variables are excluded from tuple comparison.
- **Value format mismatch**: One query returns `"42"` and the other `"42.0"`. The numeric normalizer handles this for variables classified as numeric, but only if the classification is correct.
- **LIMIT/OFFSET truncation**: If the gold query has `LIMIT 10` and the predicted query returns all results, recall will be artificially inflated. Conversely, if the predicted query has `LIMIT 10` but the gold returns 100 results, recall will be deflated. Use `strip_predicted_limit=True` to strip LIMIT/OFFSET from both sides before execution.

### Slow Evaluation

Each test case requires two HTTP requests (gold + predicted). To speed things up:
- Use `prefetch_gold_results()` to batch gold queries
- Enable `cache_gold_results=True` (default) to avoid re-executing gold queries
- Reduce `max_results` if you don't need exact counts for very large result sets

---

## Command-Line Scripts

### `scripts/run_f1_evaluation.py`

The main script for running a full F1 evaluation. It translates every NL question in the dataset, executes both gold and predicted queries against the LiITA endpoint, and saves an F1 report.

```bash
# Basic usage (Italian questions, Mistral default model)
python scripts/run_f1_evaluation.py --provider mistral

# English questions, specific model, with LIMIT stripping (recommended)
python scripts/run_f1_evaluation.py \
    --provider anthropic \
    --model claude-haiku-4-5-20251001 \
    --language en \
    --strip-limit

# With Anthropic prompt caching (reduces cost ~90% on the system prompt)
python scripts/run_f1_evaluation.py \
    --provider anthropic \
    --model claude-sonnet-4-6 \
    --strip-limit \
    --cache

# Rate limiting for free-tier APIs (e.g. Mistral: 1 req/s, 100K TPM)
python scripts/run_f1_evaluation.py \
    --provider mistral \
    --delay 1.1 \
    --tpm-limit 100000
```

Key flags:

| Flag | Description |
|---|---|
| `--provider` | LLM provider: `openai`, `anthropic`, `mistral`, `gemini`, `ollama` |
| `--model` | Model ID (uses provider default if omitted) |
| `--language` | `it` (default) or `en` |
| `--strip-limit` | Strip LIMIT/OFFSET from both gold and predicted (recommended) |
| `--cache` | Anthropic prompt caching for the system prompt (Anthropic only) |
| `--delay` | Minimum seconds between calls (RPS cap) |
| `--tpm-limit` | Tokens-per-minute budget (sliding-window cap) |
| `--no-prefetch` | Disable gold query pre-fetching |
| `--use-agent` | Use `NL2SPARQLAgent` instead of the standard translator |
| `-o` / `--output` | Output path (default: `reports/f1_report_<provider>_<model>.json`) |
| `--timeout` | SPARQL endpoint timeout in seconds (default: 60) |


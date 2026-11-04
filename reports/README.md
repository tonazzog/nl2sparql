# Evaluation reports

Reports in this folder are the ones used in the thesis: canonical gold
(`nl2sparql/data/test_dataset.json`, hypolemmas included), unified F1 evaluator,
`mistral-large-4-0`, run on 3–4 November 2026.

| File | Avg F1 | Notes |
| --- | --- | --- |
| `f1_report_mistral_mistral-large-4-0.json` | 0.780 | Standard pipeline |
| `f1_report_agent_v2_mistral_mistral-large-4-0.json` | 0.818 | LangGraph agent, v2 (commit `4d98d5a`) |

`.calls.jsonl` files have one line per LLM call with token usage; `.log` files are
the console output.

## archive/

Not used for the thesis results.

- `superseded/` – LangGraph agent v1 (avg F1 0.686): the first version, whose
  prompt forbade the CompL-it `SERVICE` block. No token log was recorded.
- `2026-03/` – March 2026 run (Claude Sonnet 4.6) on the older, uncorrected gold;
  not comparable.

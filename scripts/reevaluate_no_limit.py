#!/usr/bin/env python3
"""Re-evaluate an existing F1 report stripping LIMIT/OFFSET from predicted queries.

Reads a previously saved F1 report JSON (which contains the predicted_sparql
for every test case), strips any LIMIT/OFFSET clauses, re-executes only the
affected predicted queries, and saves a new report with corrected F1 scores.

Usage:
    python scripts/reevaluate_no_limit.py f1_report_anthropic_claude-sonnet-4-6.json
    python scripts/reevaluate_no_limit.py report.json --dataset nl2sparql/data/test_dataset.json -o report_no_limit.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sparql.config import LIITA_ENDPOINT
from nl2sparql.evaluation.f1_evaluator import (
    F1Evaluator,
    build_variable_mapping,
    compute_f1,
    execute_query_full,
    strip_limit_offset,
)
from nl2sparql.evaluation.evaluate import load_test_dataset


def _load_json_lenient(path: Path) -> dict:
    """Load a JSON file, tolerating trailing commas."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate F1 report stripping LIMIT/OFFSET from predicted queries"
    )
    parser.add_argument(
        "report",
        help="Existing F1 report JSON file (must contain predicted_sparql per result)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Test dataset JSON for gold queries and answer_variables "
             "(default: nl2sparql/data/test_dataset.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: reports/<report>_no_limit.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="SPARQL endpoint timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: report file not found: {report_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"{report_path.stem}_no_limit.json"

    # Load report
    report_data = _load_json_lenient(report_path)
    results_in = report_data["results"]
    print(f"Loaded report: {len(results_in)} results from {report_path}")

    # Load dataset
    print("Loading test dataset...")
    test_data = load_test_dataset(args.dataset)
    tc_by_id = {tc["id"]: tc for tc in test_data["test_cases"]}
    print(f"  {len(tc_by_id)} test cases loaded")

    # Pre-fetch gold results — strip LIMIT from gold queries too so both sides are comparable
    evaluator = F1Evaluator(timeout=args.timeout, strip_predicted_limit=True)
    print(f"Pre-fetching gold results ({len(tc_by_id)} queries, LIMIT stripped)...")
    evaluator.prefetch_gold_results(test_data)
    print(f"  Cached: {len(evaluator._gold_cache)}")

    # Re-evaluate
    new_results = []
    stats = {"unchanged": 0, "re_evaluated": 0, "improved": 0, "worse": 0, "same": 0}

    print(f"\nRe-evaluating queries with LIMIT clauses...")
    for entry in results_in:
        test_id = entry["test_id"]
        predicted_sparql = entry.get("predicted_sparql") or ""

        # Skip entries that already failed or have no predicted query
        if entry.get("gold_error") or entry.get("predicted_error") or not predicted_sparql:
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        stripped = strip_limit_offset(predicted_sparql)
        if stripped == predicted_sparql:
            # No LIMIT/OFFSET found — nothing to do
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        # Get test case for answer_variables
        tc = tc_by_id.get(test_id)
        if not tc:
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        # Get gold execution from cache
        gold_exec = evaluator._gold_cache.get(test_id)
        if not gold_exec or not gold_exec.success:
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        # Re-execute stripped predicted query
        pred_exec = execute_query_full(stripped, LIITA_ENDPOINT, args.timeout)

        if not pred_exec.success:
            entry_copy = dict(entry)
            entry_copy["predicted_sparql"] = stripped
            entry_copy["predicted_error"] = pred_exec.error
            new_results.append(entry_copy)
            stats["unchanged"] += 1
            continue

        # Recompute F1
        answer_vars = tc.get("answer_variables", {})
        var_mapping = build_variable_mapping(answer_vars, stripped)
        numeric_vars = set(answer_vars.get("numeric", []))
        numeric_vars.update(answer_vars.get("aggregates", []))

        f1_result = compute_f1(
            gold_results=gold_exec.results,
            predicted_results=pred_exec.results,
            answer_vars=answer_vars,
            variable_mapping=var_mapping,
            numeric_vars=numeric_vars,
        )

        old_f1 = entry["f1"]
        new_f1 = f1_result.f1
        delta = new_f1 - old_f1

        if delta > 0.001:
            stats["improved"] += 1
        elif delta < -0.001:
            stats["worse"] += 1
        else:
            stats["same"] += 1
        stats["re_evaluated"] += 1

        print(
            f"  [{test_id:4d}] F1 {old_f1:.4f} -> {new_f1:.4f}  "
            f"recall {entry['recall']:.4f} -> {f1_result.recall:.4f}  "
            f"(gold={gold_exec.result_count}, pred={pred_exec.result_count})"
        )

        new_results.append({
            "test_id": test_id,
            "f1": f1_result.f1,
            "precision": f1_result.precision,
            "recall": f1_result.recall,
            "gold_count": gold_exec.result_count,
            "predicted_count": pred_exec.result_count,
            "true_positives": f1_result.true_positives,
            "aggregate_score": f1_result.aggregate_score,
            "aggregate_details": f1_result.aggregate_details,
            "variable_mapping": f1_result.variable_mapping,
            "predicted_sparql": stripped,
            "gold_error": None,
            "predicted_error": None,
        })

    # Recompute aggregate metrics
    evaluated = [r for r in new_results if not r.get("gold_error")]
    total_evaluated = len(evaluated)
    total_skipped = len(new_results) - total_evaluated

    avg_precision = sum(r["precision"] for r in evaluated) / total_evaluated if total_evaluated else 0.0
    avg_recall    = sum(r["recall"]    for r in evaluated) / total_evaluated if total_evaluated else 0.0
    avg_f1        = sum(r["f1"]        for r in evaluated) / total_evaluated if total_evaluated else 0.0

    # Macro F1 (average of per-category averages)
    cat_f1: dict[str, list[float]] = {}
    pat_f1: dict[str, list[float]] = {}
    for r in evaluated:
        tc = tc_by_id.get(r["test_id"])
        if not tc:
            continue
        cat = tc.get("category", "unknown")
        cat_f1.setdefault(cat, []).append(r["f1"])
        for pat in tc.get("patterns", []):
            pat_f1.setdefault(pat, []).append(r["f1"])

    cat_avgs = [sum(v) / len(v) for v in cat_f1.values() if v]
    macro_f1 = sum(cat_avgs) / len(cat_avgs) if cat_avgs else 0.0

    by_category = {cat: {"avg_f1": sum(v) / len(v), "count": len(v)} for cat, v in cat_f1.items()}
    by_pattern  = {pat: {"avg_f1": sum(v) / len(v), "count": len(v)} for pat, v in pat_f1.items()}

    output_data = {
        "summary": {
            "total_evaluated": total_evaluated,
            "total_skipped": total_skipped,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "macro_f1": macro_f1,
            "note": "LIMIT/OFFSET stripped from predicted queries",
            "source_report": str(report_path),
        },
        "by_category": by_category,
        "by_pattern": by_pattern,
        "results": new_results,
    }

    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    old_summary = report_data["summary"]
    print(f"\n{'='*55}")
    print(f"RE-EVALUATION RESULTS  (LIMIT stripped)")
    print(f"{'='*55}")
    print(f"  Re-evaluated : {stats['re_evaluated']}  (had LIMIT/OFFSET)")
    print(f"  Unchanged    : {stats['unchanged']}  (no LIMIT or already failed)")
    print(f"  Improved     : {stats['improved']}  |  Same: {stats['same']}  |  Worse: {stats['worse']}")
    print(f"\n  Metric       Before     After")
    print(f"  {'-'*38}")
    print(f"  Avg F1     : {old_summary['avg_f1']:.4f}  ->  {avg_f1:.4f}")
    print(f"  Macro F1   : {old_summary['macro_f1']:.4f}  ->  {macro_f1:.4f}")
    print(f"  Precision  : {old_summary['avg_precision']:.4f}  ->  {avg_precision:.4f}")
    print(f"  Recall     : {old_summary['avg_recall']:.4f}  ->  {avg_recall:.4f}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

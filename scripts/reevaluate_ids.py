#!/usr/bin/env python3
"""Re-evaluate an F1 report for specific test IDs and recompute aggregate scores.

Two modes of operation:

1. **Re-execute mode** (default): reads the stored predicted_sparql from an
   existing report, re-executes it against the endpoint, and recomputes F1.
   Use this to verify fixes that changed gold queries or variable mappings
   without re-running the LLM.

2. **Re-translate mode** (with --provider): re-runs the LLM translator for the
   selected IDs, then evaluates the fresh predictions.  Use this when you want
   to test whether model output improved for specific cases.

In both modes, results for un-selected IDs are copied unchanged from the source
report.  Overall aggregate metrics are recomputed over all results.

Usage:
    # Re-execute stored SPARQL for IDs 2066 2067 2068
    python scripts/reevaluate_ids.py reports/f1_report_anthropic_claude-sonnet-4-6.json --ids 2066 2067 2068

    # Re-translate with fresh LLM calls for those IDs
    python scripts/reevaluate_ids.py reports/f1_report_anthropic_claude-sonnet-4-6.json \\
        --ids 2066 2067 2068 --provider anthropic --model claude-sonnet-4-6

    # Strip LIMIT from re-executed queries and save to custom path
    python scripts/reevaluate_ids.py reports/f1_report_anthropic_claude-sonnet-4-6.json \\
        --ids 2066 2067 --strip-limit -o reports/patched.json
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
    # Strip trailing commas before ] or } (not valid JSON but common in hand-edited files)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return json.loads(text)


def _recompute_aggregates(results: list[dict], tc_by_id: dict) -> dict:
    """Return summary + by_category + by_pattern dicts recomputed from results."""
    evaluated = [r for r in results if not r.get("gold_error")]
    total_evaluated = len(evaluated)
    total_skipped = len(results) - total_evaluated

    avg_precision = sum(r["precision"] for r in evaluated) / total_evaluated if total_evaluated else 0.0
    avg_recall    = sum(r["recall"]    for r in evaluated) / total_evaluated if total_evaluated else 0.0
    avg_f1        = sum(r["f1"]        for r in evaluated) / total_evaluated if total_evaluated else 0.0

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

    summary = {
        "total_evaluated": total_evaluated,
        "total_skipped": total_skipped,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "macro_f1": macro_f1,
    }
    by_category = {cat: {"avg_f1": sum(v) / len(v), "count": len(v)} for cat, v in cat_f1.items()}
    by_pattern  = {pat: {"avg_f1": sum(v) / len(v), "count": len(v)} for pat, v in pat_f1.items()}
    return summary, by_category, by_pattern


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate F1 report for specific test IDs"
    )
    parser.add_argument(
        "report",
        help="Existing F1 report JSON file (must contain predicted_sparql per result)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        required=True,
        metavar="ID",
        help="Test IDs to re-evaluate (space-separated, e.g. --ids 2066 2067 2068)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Test dataset JSON (default: nl2sparql/data/test_dataset.json)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider for re-translation (openai, anthropic, mistral, …). "
             "If omitted, re-executes stored predicted_sparql from the report.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (uses provider default if omitted)",
    )
    parser.add_argument(
        "--language",
        default="it",
        choices=["it", "en"],
        help="NL question language used for re-translation (default: it)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for re-translation (uses environment variable if omitted)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: reports/<report>_patched.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="SPARQL endpoint timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--strip-limit",
        action="store_true",
        dest="strip_limit",
        help="Strip LIMIT/OFFSET from predicted queries before executing",
    )
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: report file not found: {report_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"{report_path.stem}_patched.json"

    target_ids = set(args.ids)

    # ── Load report ────────────────────────────────────────────────────────
    report_data = _load_json_lenient(report_path)
    results_in = report_data["results"]
    print(f"Loaded report : {len(results_in)} results from {report_path}")
    print(f"Target IDs    : {sorted(target_ids)}")

    # Check all requested IDs exist in the report
    report_ids = {r["test_id"] for r in results_in}
    missing = target_ids - report_ids
    if missing:
        print(f"Warning: IDs not found in report: {sorted(missing)}")

    # ── Load dataset ───────────────────────────────────────────────────────
    print("Loading test dataset...")
    test_data = load_test_dataset(args.dataset)
    tc_by_id = {tc["id"]: tc for tc in test_data["test_cases"]}
    print(f"  {len(tc_by_id)} test cases loaded")

    # ── Pre-fetch gold results ─────────────────────────────────────────────
    evaluator = F1Evaluator(timeout=args.timeout, strip_predicted_limit=args.strip_limit)
    # Only prefetch gold for the target IDs to save time
    target_test_data = {
        "test_cases": [tc for tc in test_data["test_cases"] if tc["id"] in target_ids]
    }
    print(f"Pre-fetching gold results for {len(target_ids)} target queries...")
    evaluator.prefetch_gold_results(target_test_data)
    print(f"  Cached: {len(evaluator._gold_cache)}")

    # ── Build translator (re-translate mode) ───────────────────────────────
    translator = None
    if args.provider:
        print(f"\nInitialising translator ({args.provider} / {args.model or 'default'})...")
        try:
            from nl2sparql.generation.synthesizer import NL2SPARQL
            translator = NL2SPARQL(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                validate=True,
                fix_errors=True,
            )
        except Exception as e:
            print(f"Error initialising translator: {e}")
            sys.exit(1)
        print("  Re-translate mode: fresh LLM calls for target IDs")
    else:
        print("\nRe-execute mode: re-running stored predicted_sparql for target IDs")
    if args.strip_limit:
        print("  Strip LIMIT/OFFSET: enabled")

    # ── Re-evaluate target IDs ─────────────────────────────────────────────
    new_results = []
    stats = {"re_evaluated": 0, "unchanged": 0, "improved": 0, "worse": 0, "same": 0, "error": 0}

    for entry in results_in:
        test_id = entry["test_id"]

        if test_id not in target_ids:
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        tc = tc_by_id.get(test_id)
        if not tc:
            print(f"  [{test_id:4d}] Skipped — not found in dataset")
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        # ── Get predicted SPARQL ───────────────────────────────────────────
        if translator is not None:
            question = tc.get(f"nl_{args.language}", tc.get("nl_it", tc.get("nl_en", "")))
            try:
                translation = translator.translate(question)
                predicted_sparql = translation.sparql or ""
            except Exception as e:
                print(f"  [{test_id:4d}] Translation error: {e}")
                new_results.append({**entry, "predicted_sparql": None, "predicted_error": str(e),
                                     "f1": 0.0, "precision": 0.0, "recall": 0.0})
                stats["error"] += 1
                continue
        else:
            predicted_sparql = entry.get("predicted_sparql") or ""

        if not predicted_sparql:
            print(f"  [{test_id:4d}] Skipped — no predicted SPARQL")
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        if args.strip_limit:
            predicted_sparql = strip_limit_offset(predicted_sparql)

        # ── Get gold from cache ────────────────────────────────────────────
        gold_exec = evaluator._gold_cache.get(test_id)
        if not gold_exec or not gold_exec.success:
            print(f"  [{test_id:4d}] Skipped — gold query failed")
            new_results.append(entry)
            stats["unchanged"] += 1
            continue

        # ── Execute predicted query ────────────────────────────────────────
        pred_exec = execute_query_full(predicted_sparql, LIITA_ENDPOINT, args.timeout)

        if not pred_exec.success:
            print(f"  [{test_id:4d}] Predicted query error: {pred_exec.error}")
            new_results.append({
                **entry,
                "predicted_sparql": predicted_sparql,
                "predicted_error": pred_exec.error,
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "gold_count": gold_exec.result_count, "predicted_count": 0,
                "true_positives": 0,
            })
            stats["error"] += 1
            continue

        # ── Recompute F1 ───────────────────────────────────────────────────
        answer_vars = tc.get("answer_variables", {})
        var_mapping = build_variable_mapping(answer_vars, predicted_sparql)
        numeric_vars = set(answer_vars.get("numeric", []))
        numeric_vars.update(answer_vars.get("aggregates", []))

        f1_result = compute_f1(
            gold_results=gold_exec.results,
            predicted_results=pred_exec.results,
            answer_vars=answer_vars,
            variable_mapping=var_mapping,
            numeric_vars=numeric_vars,
        )

        old_f1 = entry.get("f1", 0.0)
        new_f1 = f1_result.f1
        delta = new_f1 - old_f1

        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        print(
            f"  [{test_id:4d}] F1 {old_f1:.4f} {arrow} {new_f1:.4f}  "
            f"P={f1_result.precision:.4f}  R={f1_result.recall:.4f}  "
            f"(gold={gold_exec.result_count}, pred={pred_exec.result_count})"
        )

        if delta > 0.001:
            stats["improved"] += 1
        elif delta < -0.001:
            stats["worse"] += 1
        else:
            stats["same"] += 1
        stats["re_evaluated"] += 1

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
            "predicted_sparql": predicted_sparql,
            "gold_error": None,
            "predicted_error": None,
        })

    # ── Recompute aggregate metrics ────────────────────────────────────────
    summary, by_category, by_pattern = _recompute_aggregates(new_results, tc_by_id)

    mode_note = (
        f"Re-translated with {args.provider}/{args.model or 'default'}"
        if translator else "Re-executed stored predicted_sparql"
    )
    if args.strip_limit:
        mode_note += ", LIMIT/OFFSET stripped"
    summary["note"] = mode_note
    summary["source_report"] = str(report_path)
    summary["patched_ids"] = sorted(target_ids)

    output_data = {
        "summary": summary,
        "by_category": by_category,
        "by_pattern": by_pattern,
        "results": new_results,
    }

    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Final summary ──────────────────────────────────────────────────────
    old_summary = report_data["summary"]
    print(f"\n{'='*60}")
    print(f"RE-EVALUATION RESULTS  ({mode_note})")
    print(f"{'='*60}")
    print(f"  Target IDs   : {len(target_ids)}")
    print(f"  Re-evaluated : {stats['re_evaluated']}")
    print(f"  Errors       : {stats['error']}")
    print(f"  Unchanged    : {stats['unchanged']}")
    if stats["re_evaluated"]:
        print(f"  Improved ↑   : {stats['improved']}  |  Same =: {stats['same']}  |  Worse ↓: {stats['worse']}")
    print(f"\n  Metric       Before       After")
    print(f"  {'-'*42}")
    print(f"  Avg F1     : {old_summary['avg_f1']:.4f}   ->   {summary['avg_f1']:.4f}")
    print(f"  Macro F1   : {old_summary['macro_f1']:.4f}   ->   {summary['macro_f1']:.4f}")
    print(f"  Precision  : {old_summary['avg_precision']:.4f}   ->   {summary['avg_precision']:.4f}")
    print(f"  Recall     : {old_summary['avg_recall']:.4f}   ->   {summary['avg_recall']:.4f}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

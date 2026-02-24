#!/usr/bin/env python3
"""Run F1 evaluation on a test dataset using a specified LLM provider.

Translates every NL question in the dataset, executes both the gold and
predicted SPARQL queries against the LiITA endpoint, and computes F1 on the
result sets. The output JSON file contains aggregate metrics AND all generated
queries for inspection.

Usage:
    python scripts/run_f1_evaluation.py --provider mistral
    python scripts/run_f1_evaluation.py --provider anthropic --model claude-haiku-4-5-20251001
    python scripts/run_f1_evaluation.py --provider openai --dataset nl2sparql/data/test_dataset_en_variations_merged.json
    python scripts/run_f1_evaluation.py --provider mistral --language en --no-prefetch
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sparql.evaluation import F1Evaluator, save_f1_report
from nl2sparql.evaluation.evaluate import load_test_dataset


class _RateLimitedTranslator:
    """Wraps any translator and enforces rate limits between calls.

    Handles two independent constraints:
      - RPS (requests per second): minimum wall-clock gap between calls
      - TPM (tokens per minute): sliding-window token budget

    The effective delay is max(rps_delay, tpm_delay) so both limits are
    respected simultaneously. Token counting uses a character-based estimate
    (chars / 4) for the question plus a fixed overhead for the system prompt,
    few-shot examples, and generated output.
    """

    # Conservative estimate of tokens that don't vary per question:
    # system prompt + constraints (~8 000) + 5 examples (~1 750) + output (~500)
    _OVERHEAD_TOKENS = 10_250

    def __init__(
        self,
        translator,
        delay: float,
        tpm_limit: int = 0,
        tokens_per_call: int = 0,
    ):
        """
        Args:
            translator:      The wrapped translator object.
            delay:           Minimum seconds between calls (RPS-based). 0 = no RPS cap.
            tpm_limit:       Tokens-per-minute budget. 0 = no TPM cap.
            tokens_per_call: Override the per-call token estimate used for TPM
                             accounting. 0 = auto (overhead + question chars/4).
        """
        self._translator = translator
        self._delay = delay
        self._tpm_limit = tpm_limit
        self._tokens_per_call_override = tokens_per_call
        self._last_call: float = 0.0
        # Sliding window: list of (monotonic_time, tokens) for the last 60 s
        self._window: list[tuple[float, int]] = []

    def _estimate_tokens(self, question: str) -> int:
        if self._tokens_per_call_override:
            return self._tokens_per_call_override
        return self._OVERHEAD_TOKENS + max(1, len(question) // 4)

    def _tpm_wait(self, tokens: int) -> float:
        """Return seconds to sleep so that adding `tokens` stays under TPM limit."""
        if not self._tpm_limit:
            return 0.0

        now = time.monotonic()
        cutoff = now - 60.0
        # Drop entries older than 60 s
        self._window = [(t, tok) for t, tok in self._window if t > cutoff]

        used = sum(tok for _, tok in self._window)
        headroom = self._tpm_limit - used

        if tokens <= headroom:
            return 0.0

        # Find the earliest entry whose removal would free enough space
        cumulative = 0
        for ts, tok in sorted(self._window):
            cumulative += tok
            if used - cumulative + self._tpm_limit >= tokens:
                # Sleep until that entry expires
                return max(0.0, (ts + 60.0) - now)

        # Worst case: wait until the whole window clears
        if self._window:
            oldest = min(ts for ts, _ in self._window)
            return max(0.0, (oldest + 60.0) - now)
        return 0.0

    def translate(self, question: str):
        tokens = self._estimate_tokens(question)

        # RPS-based delay
        rps_wait = max(0.0, self._delay - (time.monotonic() - self._last_call))

        # TPM-based delay
        tpm_wait = self._tpm_wait(tokens)

        wait = max(rps_wait, tpm_wait)
        if wait > 0:
            time.sleep(wait)

        self._last_call = time.monotonic()
        result = self._translator.translate(question)

        # Record this call in the sliding window
        if self._tpm_limit:
            self._window.append((time.monotonic(), tokens))

        return result


def main():
    parser = argparse.ArgumentParser(
        description="F1 evaluation: translate questions and compare answer sets"
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="LLM provider (openai, anthropic, mistral, gemini, ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (uses provider default if omitted)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (uses environment variable if omitted)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to test dataset JSON (default: nl2sparql/data/test_dataset.json)",
    )
    parser.add_argument(
        "--language",
        default="it",
        choices=["it", "en"],
        help="NL question language to use (default: it)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: reports/f1_report_<provider>_<model>.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="SPARQL endpoint timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help=(
            "Minimum seconds between translation calls (RPS cap). "
            "E.g. 1.1 for Mistral free tier (1 req/s). Default: 0."
        ),
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=0,
        dest="tpm_limit",
        help=(
            "Tokens-per-minute budget (TPM cap). The script will automatically "
            "pause when the sliding-window token count approaches this limit. "
            "E.g. --tpm-limit 100000 for a 100K TPM free tier. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--tokens-per-call",
        type=int,
        default=0,
        dest="tokens_per_call",
        help=(
            "Override the per-call token estimate used for TPM accounting. "
            "Default: 0 (auto: ~10 250 overhead + question length / 4)."
        ),
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable gold query pre-fetching (saves memory, slower for repeated runs)",
    )
    parser.add_argument(
        "--strip-limit",
        action="store_true",
        dest="strip_limit",
        help=(
            "Strip LIMIT (and OFFSET) clauses from predicted queries before "
            "executing them. Fixes artificially low recall when the model adds "
            "a LIMIT that truncates the result set."
        ),
    )
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Use NL2SPARQLAgent instead of the standard NL2SPARQL translator",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help=(
            "Enable Anthropic prompt caching for the system prompt "
            "(Anthropic provider only). Reduces cost ~90%% on the cached "
            "portion for repeated calls with the same system prompt."
        ),
    )

    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_tag = (args.model or "default").replace("/", "-").replace(":", "-")
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"f1_report_{args.provider}_{model_tag}.json"

    # Load dataset
    print(f"Loading dataset...")
    test_data = load_test_dataset(args.dataset)
    total = len(test_data["test_cases"])
    print(f"  {total} test cases loaded")
    if args.dataset:
        print(f"  Source: {args.dataset}")

    # Build translator
    print(f"\nInitialising translator...")
    print(f"  Provider : {args.provider}")
    print(f"  Model    : {args.model or '(provider default)'}")
    print(f"  Language : {args.language}")
    print(f"  Mode     : {'agent' if args.use_agent else 'standard'}")
    if args.cache:
        if args.provider != "anthropic":
            print("  Warning  : --cache is only supported with --provider anthropic, ignoring")
            args.cache = False
        else:
            print(f"  Caching  : system prompt caching enabled (5-min TTL)")

    try:
        if args.use_agent:
            from nl2sparql.agent import NL2SPARQLAgent
            from nl2sparql.evaluation import AgentAdapter
            agent = NL2SPARQLAgent(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
            )
            translator = AgentAdapter(agent)
        else:
            from nl2sparql.generation.synthesizer import NL2SPARQL
            translator = NL2SPARQL(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                validate=True,
                fix_errors=True,
                cache_system_prompt=args.cache,
            )
    except Exception as e:
        print(f"\nError initialising translator: {e}")
        sys.exit(1)

    # Build evaluator
    if args.strip_limit:
        print(f"  Strip LIMIT: enabled (LIMIT/OFFSET removed from predicted queries)")
    evaluator = F1Evaluator(
        timeout=args.timeout,
        cache_gold_results=not args.no_prefetch,
        strip_predicted_limit=args.strip_limit,
    )

    # Pre-fetch gold results (runs all gold queries once up front)
    if not args.no_prefetch:
        print(f"\nPre-fetching gold query results ({total} queries)...")
        evaluator.prefetch_gold_results(test_data)
        cached = len(evaluator._gold_cache)
        skipped = total - cached
        print(f"  Cached: {cached}  |  Failed/skipped: {skipped}")

    # Wrap translator with rate limiter if requested
    if args.delay > 0 or args.tpm_limit > 0:
        translator = _RateLimitedTranslator(
            translator,
            delay=args.delay,
            tpm_limit=args.tpm_limit,
            tokens_per_call=args.tokens_per_call,
        )
        est = args.tokens_per_call or (
            _RateLimitedTranslator._OVERHEAD_TOKENS + 40   # ~40 tokens avg question
        )
        print(f"\nRate limiting:")
        if args.delay > 0:
            print(f"  RPS cap : {args.delay}s between calls  "
                  f"(max {60/args.delay:.0f} calls/min)")
        if args.tpm_limit > 0:
            max_calls_tpm = args.tpm_limit / est
            safe_delay = 60.0 / max_calls_tpm
            print(f"  TPM cap : {args.tpm_limit:,} tokens/min  "
                  f"(~{est:,} tokens/call → max {max_calls_tpm:.1f} calls/min, "
                  f"effective delay ≥ {safe_delay:.1f}s)")
        effective = max(args.delay, (60.0 / (args.tpm_limit / est)) if args.tpm_limit else 0)
        total_est = len(test_data["test_cases"]) * effective
        print(f"  Estimated total run time: ~{total_est/60:.1f} min")

    # Run evaluation
    print(f"\nTranslating and evaluating...")
    report = evaluator.evaluate_dataset(
        test_data=test_data,
        translator=translator,
        language=args.language,
    )

    # Save
    save_f1_report(report, str(output_path))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"F1 EVALUATION RESULTS  —  {args.provider} / {args.model or 'default'}")
    print(f"{'='*55}")
    print(f"  Evaluated : {report.total_evaluated}")
    print(f"  Skipped   : {report.total_skipped}  (gold query failed)")
    print(f"  Avg F1    : {report.avg_f1:.4f}")
    print(f"  Macro F1  : {report.macro_f1:.4f}")
    print(f"  Precision : {report.avg_precision:.4f}")
    print(f"  Recall    : {report.avg_recall:.4f}")

    if report.f1_by_category:
        print(f"\n  By category:")
        for cat, stats in sorted(report.f1_by_category.items()):
            print(f"    {cat:<25} avg F1={stats['avg_f1']:.4f}  (n={stats['count']})")

    if report.f1_by_pattern:
        print(f"\n  By pattern (top 10):")
        sorted_pats = sorted(
            report.f1_by_pattern.items(),
            key=lambda x: -x[1]["avg_f1"],
        )[:10]
        for pat, stats in sorted_pats:
            print(f"    {pat:<28} avg F1={stats['avg_f1']:.4f}  (n={stats['count']})")

    # Distribution of F1 scores
    if report.results:
        perfect = sum(1 for r in report.results if r.f1 == 1.0)
        zeros   = sum(1 for r in report.results if r.f1 == 0.0)
        errors  = sum(1 for r in report.results if r.predicted_error)
        print(f"\n  Score distribution:")
        print(f"    F1 = 1.00 (perfect): {perfect}")
        print(f"    F1 = 0.00          : {zeros}")
        print(f"    Translation errors : {errors}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

"""Batch evaluation script for comparing multiple LLM providers/models."""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .evaluate import evaluate_dataset, save_report, EvaluationReport


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    provider: str
    model: Optional[str] = None  # None uses provider default
    name: Optional[str] = None   # Display name (auto-generated if None)

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.provider}/{self.model or 'default'}"


# Pre-defined model configurations for common comparisons
OPENAI_MODELS = [
    ModelConfig("openai", "gpt-4.1", "GPT-4.1"),
    ModelConfig("openai", "gpt-4.1-mini", "GPT-4.1-mini"),
    ModelConfig("openai", "gpt-4.1-nano", "GPT-4.1-nano"),
]

ANTHROPIC_MODELS = [
    ModelConfig("anthropic", "claude-sonnet-4-20250514", "Claude Sonnet 4"),
    ModelConfig("anthropic", "claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
]

MISTRAL_MODELS = [
    ModelConfig("mistral", "mistral-large-latest", "Mistral Large"),
    ModelConfig("mistral", "mistral-small-latest", "Mistral Small"),
]

ALL_PROVIDERS_DEFAULT = [
    ModelConfig("openai", None, "OpenAI (default)"),
    ModelConfig("anthropic", None, "Anthropic (default)"),
    ModelConfig("mistral", None, "Mistral (default)"),
    ModelConfig("gemini", None, "Gemini (default)"),
]

# Preset configurations
PRESETS = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "mistral": MISTRAL_MODELS,
    "all_defaults": ALL_PROVIDERS_DEFAULT,
    "quick": [
        ModelConfig("openai", "gpt-4.1-mini", "GPT-4.1-mini"),
        ModelConfig("anthropic", "claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
    ],
}


@dataclass
class BatchResult:
    """Result of a batch evaluation."""
    model_config: ModelConfig
    report: Optional[EvaluationReport]
    error: Optional[str]
    duration: float


def run_batch_evaluation(
    configs: list[ModelConfig],
    language: str = "it",
    validate_endpoint: bool = True,
    categories: Optional[list[str]] = None,
    patterns: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    use_agent: bool = False,
) -> list[BatchResult]:
    """
    Run evaluation for multiple model configurations.

    Args:
        configs: List of ModelConfig to evaluate
        language: "it" or "en"
        validate_endpoint: Whether to validate against endpoint
        categories: Filter by categories
        patterns: Filter by patterns
        output_dir: Directory to save individual reports (optional)
        verbose: Print progress
        use_agent: If True, use NL2SPARQLAgent instead of NL2SPARQL

    Returns:
        List of BatchResult
    """
    from ..generation.synthesizer import NL2SPARQL
    from .evaluate import AgentAdapter

    results = []

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for i, config in enumerate(configs, 1):
        mode_str = " (agent)" if use_agent else ""
        if verbose:
            print(f"\n[{i}/{len(configs)}] Evaluating {config.name}{mode_str}...")

        start_time = time.time()

        try:
            if use_agent:
                from ..agent import NL2SPARQLAgent
                agent = NL2SPARQLAgent(
                    provider=config.provider,
                    model=config.model,
                )
                translator = AgentAdapter(agent)
            else:
                translator = NL2SPARQL(
                    provider=config.provider,
                    model=config.model,
                    validate=True,
                    fix_errors=True,
                )

            report = evaluate_dataset(
                translator=translator,
                language=language,
                validate_endpoint=validate_endpoint,
                categories=categories,
                patterns=patterns,
            )

            duration = time.time() - start_time

            if verbose:
                print(f"    Syntax valid: {report.syntax_valid}/{report.total_tests} "
                      f"({100*report.syntax_valid/report.total_tests:.1f}%)")
                print(f"    Endpoint valid: {report.endpoint_valid}/{report.total_tests} "
                      f"({100*report.endpoint_valid/report.total_tests:.1f}%)")
                print(f"    Avg component score: {report.avg_component_score:.2%}")
                print(f"    Duration: {duration:.1f}s")

            # Save individual report
            if output_dir:
                safe_name = config.name.replace("/", "_").replace(" ", "_")
                report_path = output_path / f"report_{safe_name}.json"
                save_report(report, str(report_path))
                if verbose:
                    print(f"    Saved: {report_path}")

            results.append(BatchResult(
                model_config=config,
                report=report,
                error=None,
                duration=duration,
            ))

        except Exception as e:
            duration = time.time() - start_time
            if verbose:
                print(f"    ERROR: {e}")

            results.append(BatchResult(
                model_config=config,
                report=None,
                error=str(e),
                duration=duration,
            ))

    return results


def create_comparison_report(
    results: list[BatchResult],
    output_path: Optional[str] = None,
) -> dict:
    """
    Create a comparison report from batch results.

    Args:
        results: List of BatchResult from run_batch_evaluation
        output_path: Path to save comparison JSON (optional)

    Returns:
        Comparison data dictionary
    """
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models_evaluated": len(results),
        "models": [],
        "comparison": {
            "by_syntax_valid": [],
            "by_endpoint_valid": [],
            "by_component_score": [],
            "by_generation_time": [],
        },
    }

    for result in results:
        model_data = {
            "name": result.model_config.name,
            "provider": result.model_config.provider,
            "model": result.model_config.model,
            "duration": result.duration,
        }

        if result.error:
            model_data["error"] = result.error
            model_data["syntax_valid_rate"] = None
            model_data["endpoint_valid_rate"] = None
            model_data["avg_component_score"] = None
            model_data["avg_generation_time"] = None
        else:
            r = result.report
            model_data["total_tests"] = r.total_tests
            model_data["syntax_valid"] = r.syntax_valid
            model_data["syntax_valid_rate"] = r.syntax_valid / r.total_tests if r.total_tests else 0
            model_data["endpoint_valid"] = r.endpoint_valid
            model_data["endpoint_valid_rate"] = r.endpoint_valid / r.total_tests if r.total_tests else 0
            model_data["avg_component_score"] = r.avg_component_score
            model_data["avg_generation_time"] = r.avg_generation_time
            model_data["pattern_detection_accuracy"] = r.pattern_detection_accuracy

            # Per-category breakdown
            model_data["by_category"] = {
                cat: {
                    "syntax_valid": stats["syntax_valid"],
                    "total": stats["total"],
                    "rate": stats["syntax_valid"] / stats["total"] if stats["total"] else 0,
                }
                for cat, stats in r.results_by_category.items()
            }

        comparison["models"].append(model_data)

    # Sort for rankings (only successful evaluations)
    successful = [m for m in comparison["models"] if m.get("syntax_valid_rate") is not None]

    comparison["comparison"]["by_syntax_valid"] = sorted(
        [{"name": m["name"], "rate": m["syntax_valid_rate"]} for m in successful],
        key=lambda x: x["rate"],
        reverse=True
    )

    comparison["comparison"]["by_endpoint_valid"] = sorted(
        [{"name": m["name"], "rate": m["endpoint_valid_rate"]} for m in successful],
        key=lambda x: x["rate"],
        reverse=True
    )

    comparison["comparison"]["by_component_score"] = sorted(
        [{"name": m["name"], "score": m["avg_component_score"]} for m in successful],
        key=lambda x: x["score"],
        reverse=True
    )

    comparison["comparison"]["by_generation_time"] = sorted(
        [{"name": m["name"], "time": m["avg_generation_time"]} for m in successful],
        key=lambda x: x["time"]
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

    return comparison


def print_comparison(comparison: dict) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)

    print(f"\nModels evaluated: {comparison['models_evaluated']}")
    print(f"Timestamp: {comparison['timestamp']}")

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Model':<30} {'Syntax':<12} {'Endpoint':<12} {'Component':<12} {'Time':<10}")
    print("-" * 70)

    for m in comparison["models"]:
        if m.get("error"):
            print(f"{m['name']:<30} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<10}")
        else:
            syntax = f"{m['syntax_valid_rate']*100:.1f}%"
            endpoint = f"{m['endpoint_valid_rate']*100:.1f}%"
            component = f"{m['avg_component_score']*100:.1f}%"
            time_str = f"{m['avg_generation_time']:.2f}s"
            print(f"{m['name']:<30} {syntax:<12} {endpoint:<12} {component:<12} {time_str:<10}")

    print("-" * 70)

    # Rankings
    print("\nRankings:")

    print("\n  By Syntax Validity:")
    for i, item in enumerate(comparison["comparison"]["by_syntax_valid"][:5], 1):
        print(f"    {i}. {item['name']}: {item['rate']*100:.1f}%")

    print("\n  By Endpoint Success:")
    for i, item in enumerate(comparison["comparison"]["by_endpoint_valid"][:5], 1):
        print(f"    {i}. {item['name']}: {item['rate']*100:.1f}%")

    print("\n  By Component Score:")
    for i, item in enumerate(comparison["comparison"]["by_component_score"][:5], 1):
        print(f"    {i}. {item['name']}: {item['score']*100:.1f}%")

    print("\n  By Generation Speed (fastest):")
    for i, item in enumerate(comparison["comparison"]["by_generation_time"][:5], 1):
        print(f"    {i}. {item['name']}: {item['time']:.2f}s")


def batch_evaluate_cli():
    """CLI entry point for batch evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch evaluate multiple LLM models")
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration"
    )
    parser.add_argument(
        "--provider",
        action="append",
        help="Provider to evaluate (can specify multiple)"
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model to evaluate (pairs with --provider)"
    )
    parser.add_argument(
        "--language", "-l",
        choices=["it", "en"],
        default="it",
        help="Test language"
    )
    parser.add_argument(
        "--no-endpoint",
        action="store_true",
        help="Skip endpoint validation"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save reports"
    )
    parser.add_argument(
        "--comparison", "-c",
        help="Path to save comparison report"
    )

    args = parser.parse_args()

    # Build config list
    if args.preset:
        configs = PRESETS[args.preset]
    elif args.provider:
        models = args.model or [None] * len(args.provider)
        configs = [
            ModelConfig(p, m)
            for p, m in zip(args.provider, models)
        ]
    else:
        print("Error: Specify --preset or --provider")
        return

    # Run batch evaluation
    results = run_batch_evaluation(
        configs=configs,
        language=args.language,
        validate_endpoint=not args.no_endpoint,
        output_dir=args.output_dir,
    )

    # Create and print comparison
    comparison = create_comparison_report(
        results,
        output_path=args.comparison,
    )
    print_comparison(comparison)

    if args.comparison:
        print(f"\nComparison saved to: {args.comparison}")


if __name__ == "__main__":
    batch_evaluate_cli()

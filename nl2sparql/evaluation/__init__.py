"""Evaluation module for NL2SPARQL system."""

from .evaluate import (
    TestResult,
    EvaluationReport,
    load_test_dataset,
    evaluate_single,
    evaluate_dataset,
    print_report,
    save_report,
    AgentAdapter,
)

from .batch_evaluate import (
    ModelConfig,
    BatchResult,
    run_batch_evaluation,
    create_comparison_report,
    print_comparison,
    PRESETS,
)

__all__ = [
    # Single evaluation
    "TestResult",
    "EvaluationReport",
    "load_test_dataset",
    "evaluate_single",
    "evaluate_dataset",
    "print_report",
    "save_report",
    # Agent support
    "AgentAdapter",
    # Batch evaluation
    "ModelConfig",
    "BatchResult",
    "run_batch_evaluation",
    "create_comparison_report",
    "print_comparison",
    "PRESETS",
]

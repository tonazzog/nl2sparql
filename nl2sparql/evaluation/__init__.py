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

from .f1_evaluator import (
    F1Evaluator,
    F1Result,
    F1Report,
    save_f1_report,
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
    # F1 evaluation
    "F1Evaluator",
    "F1Result",
    "F1Report",
    "save_f1_report",
    # Batch evaluation
    "ModelConfig",
    "BatchResult",
    "run_batch_evaluation",
    "create_comparison_report",
    "print_comparison",
    "PRESETS",
]

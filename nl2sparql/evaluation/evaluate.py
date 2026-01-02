"""Evaluation module for NL2SPARQL system."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class TranslatorProtocol(Protocol):
    """Protocol for standard NL2SPARQL translator."""
    def translate(self, question: str) -> object: ...


class AgentValidation:
    """Validation result adapter for agent responses."""
    def __init__(self, is_valid: bool, result_count: int, error: Optional[str] = None):
        self.syntax_valid = is_valid
        self.execution_success = is_valid and result_count > 0
        self.execution_error = error
        self.result_count = result_count


class AgentResult:
    """Result adapter for agent responses to match translator interface."""
    def __init__(self, agent_result: dict):
        self.sparql = agent_result.get("sparql", "")
        self.detected_patterns = agent_result.get("detected_patterns", [])
        self.confidence = agent_result.get("confidence", 0.0)
        self.validation = AgentValidation(
            is_valid=agent_result.get("is_valid", False),
            result_count=agent_result.get("result_count", 0),
            error=None if agent_result.get("is_valid") else "; ".join(
                agent_result.get("refinement_history", [{}])[-1].get("error", "").split("; ")
                if agent_result.get("refinement_history") else []
            ) or None
        )


class AgentAdapter:
    """
    Adapter that wraps NL2SPARQLAgent to match the translator interface.

    This allows the agent to be used with the evaluation framework.

    Usage:
        from nl2sparql.agent import NL2SPARQLAgent
        from nl2sparql.evaluation import AgentAdapter, evaluate_dataset

        agent = NL2SPARQLAgent(provider="openai", model="gpt-4.1-mini")
        adapter = AgentAdapter(agent)

        report = evaluate_dataset(adapter)
    """

    def __init__(self, agent):
        """
        Initialize the adapter.

        Args:
            agent: NL2SPARQLAgent instance
        """
        self.agent = agent

    def translate(self, question: str) -> AgentResult:
        """
        Translate using the agent and return result in translator format.

        Args:
            question: Natural language question

        Returns:
            AgentResult that matches the translator interface
        """
        result = self.agent.translate(question)
        return AgentResult(result)


@dataclass
class TestResult:
    """Result of a single test case."""

    test_id: str
    category: str
    patterns: list[str]
    question: str

    # Generation results
    generated_sparql: Optional[str] = None
    generation_time: float = 0.0
    generation_error: Optional[str] = None

    # Validation results
    syntax_valid: bool = False
    endpoint_valid: Optional[bool] = None
    endpoint_error: Optional[str] = None
    result_count: Optional[int] = None

    # Component matching
    expected_components: list[str] = field(default_factory=list)
    matched_components: list[str] = field(default_factory=list)
    missing_components: list[str] = field(default_factory=list)
    component_score: float = 0.0

    # Pattern detection
    detected_patterns: list[str] = field(default_factory=list)
    pattern_detection_correct: bool = False


@dataclass
class EvaluationReport:
    """Overall evaluation report."""

    total_tests: int = 0
    successful_generations: int = 0
    syntax_valid: int = 0
    endpoint_valid: int = 0

    # By category
    results_by_category: dict = field(default_factory=dict)

    # By pattern
    results_by_pattern: dict = field(default_factory=dict)

    # Aggregate scores
    avg_generation_time: float = 0.0
    avg_component_score: float = 0.0
    pattern_detection_accuracy: float = 0.0

    # Individual results
    test_results: list[TestResult] = field(default_factory=list)


def load_test_dataset(path: Optional[str] = None) -> dict:
    """Load the test dataset."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "test_dataset.json"
    else:
        path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_components(sparql: str, expected: list[str]) -> tuple[list[str], list[str], float]:
    """
    Check which expected components are present in the generated SPARQL.

    Returns:
        Tuple of (matched, missing, score)
    """
    sparql_upper = sparql.upper()
    matched = []
    missing = []

    for component in expected:
        # Handle special cases
        if component.startswith("^") or component.endswith("$"):
            # Regex patterns - check if they appear in FILTER
            if component in sparql or component.replace("$", "") in sparql:
                matched.append(component)
            else:
                missing.append(component)
        elif component.upper() in sparql_upper or component in sparql:
            matched.append(component)
        else:
            missing.append(component)

    score = len(matched) / len(expected) if expected else 1.0
    return matched, missing, score


def evaluate_single(
    test_case: dict,
    translator,
    language: str = "it",
    validate_endpoint: bool = True,
) -> TestResult:
    """
    Evaluate a single test case.

    Args:
        test_case: Test case dictionary
        translator: NL2SPARQL translator instance
        language: "it" for Italian, "en" for English
        validate_endpoint: Whether to validate against endpoint

    Returns:
        TestResult
    """
    question = test_case.get(f"nl_{language}", test_case.get("nl_it", ""))

    result = TestResult(
        test_id=test_case["id"],
        category=test_case["category"],
        patterns=test_case["patterns"],
        question=question,
        expected_components=test_case.get("expected_components", []),
    )

    # Generate SPARQL
    start_time = time.time()
    try:
        translation = translator.translate(question)
        result.generated_sparql = translation.sparql
        result.detected_patterns = translation.detected_patterns
        result.generation_time = time.time() - start_time

        # Check syntax validity
        if translation.validation:
            result.syntax_valid = translation.validation.syntax_valid

            if validate_endpoint and translation.validation.execution_success is not None:
                result.endpoint_valid = translation.validation.execution_success
                result.endpoint_error = translation.validation.execution_error
                result.result_count = translation.validation.result_count

    except Exception as e:
        result.generation_error = str(e)
        result.generation_time = time.time() - start_time
        return result

    # Check components
    if result.generated_sparql:
        matched, missing, score = check_components(
            result.generated_sparql,
            result.expected_components
        )
        result.matched_components = matched
        result.missing_components = missing
        result.component_score = score

    # Check pattern detection
    expected_set = set(result.patterns)
    detected_set = set(result.detected_patterns)
    result.pattern_detection_correct = expected_set.issubset(detected_set)

    return result


def evaluate_dataset(
    translator,
    test_data: Optional[dict] = None,
    language: str = "it",
    validate_endpoint: bool = True,
    categories: Optional[list[str]] = None,
    patterns: Optional[list[str]] = None,
) -> EvaluationReport:
    """
    Evaluate the translator on the test dataset.

    Args:
        translator: NL2SPARQL translator instance
        test_data: Test dataset (loads default if None)
        language: "it" or "en"
        validate_endpoint: Whether to validate against endpoint
        categories: Filter by categories (e.g., ["single_pattern", "combination_2"])
        patterns: Filter by patterns (e.g., ["EMOTION_LEXICON", "TRANSLATION"])

    Returns:
        EvaluationReport
    """
    if test_data is None:
        test_data = load_test_dataset()

    test_cases = test_data["test_cases"]

    # Filter by categories
    if categories:
        test_cases = [t for t in test_cases if t["category"] in categories]

    # Filter by patterns
    if patterns:
        test_cases = [
            t for t in test_cases
            if any(p in t["patterns"] for p in patterns)
        ]

    report = EvaluationReport(total_tests=len(test_cases))

    for test_case in test_cases:
        result = evaluate_single(
            test_case,
            translator,
            language=language,
            validate_endpoint=validate_endpoint,
        )
        report.test_results.append(result)

        # Update counts
        if result.generated_sparql:
            report.successful_generations += 1
        if result.syntax_valid:
            report.syntax_valid += 1
        if result.endpoint_valid:
            report.endpoint_valid += 1

        # Update by category
        cat = result.category
        if cat not in report.results_by_category:
            report.results_by_category[cat] = {
                "total": 0, "success": 0, "syntax_valid": 0, "endpoint_valid": 0
            }
        report.results_by_category[cat]["total"] += 1
        if result.generated_sparql:
            report.results_by_category[cat]["success"] += 1
        if result.syntax_valid:
            report.results_by_category[cat]["syntax_valid"] += 1
        if result.endpoint_valid:
            report.results_by_category[cat]["endpoint_valid"] += 1

        # Update by pattern
        for pattern in result.patterns:
            if pattern not in report.results_by_pattern:
                report.results_by_pattern[pattern] = {
                    "total": 0, "success": 0, "component_scores": []
                }
            report.results_by_pattern[pattern]["total"] += 1
            if result.syntax_valid:
                report.results_by_pattern[pattern]["success"] += 1
            report.results_by_pattern[pattern]["component_scores"].append(result.component_score)

    # Calculate aggregates
    if report.test_results:
        times = [r.generation_time for r in report.test_results]
        report.avg_generation_time = sum(times) / len(times)

        scores = [r.component_score for r in report.test_results]
        report.avg_component_score = sum(scores) / len(scores)

        correct = sum(1 for r in report.test_results if r.pattern_detection_correct)
        report.pattern_detection_accuracy = correct / len(report.test_results)

    return report


def print_report(report: EvaluationReport) -> None:
    """Print evaluation report to console."""
    print("\n" + "=" * 60)
    print("NL2SPARQL EVALUATION REPORT")
    print("=" * 60)

    print(f"\nOverall Results:")
    print(f"  Total tests:           {report.total_tests}")
    print(f"  Successful generations: {report.successful_generations} ({100*report.successful_generations/report.total_tests:.1f}%)")
    print(f"  Syntax valid:          {report.syntax_valid} ({100*report.syntax_valid/report.total_tests:.1f}%)")
    print(f"  Endpoint valid:        {report.endpoint_valid} ({100*report.endpoint_valid/report.total_tests:.1f}%)")

    print(f"\nAggregate Metrics:")
    print(f"  Avg generation time:   {report.avg_generation_time:.2f}s")
    print(f"  Avg component score:   {report.avg_component_score:.2%}")
    print(f"  Pattern detection acc: {report.pattern_detection_accuracy:.2%}")

    print(f"\nResults by Category:")
    for cat, stats in report.results_by_category.items():
        success_rate = stats["syntax_valid"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {cat}: {stats['syntax_valid']}/{stats['total']} ({success_rate:.1f}%)")

    print(f"\nResults by Pattern:")
    for pattern, stats in sorted(report.results_by_pattern.items()):
        success_rate = stats["success"] / stats["total"] * 100 if stats["total"] else 0
        avg_score = sum(stats["component_scores"]) / len(stats["component_scores"]) if stats["component_scores"] else 0
        print(f"  {pattern}: {stats['success']}/{stats['total']} ({success_rate:.1f}%), component score: {avg_score:.2%}")

    # Print failures
    failures = [r for r in report.test_results if not r.syntax_valid]
    if failures:
        print(f"\nFailed Tests ({len(failures)}):")
        for r in failures[:10]:  # Show first 10
            print(f"  {r.test_id}: {r.question[:50]}...")
            if r.generation_error:
                print(f"    Error: {r.generation_error[:80]}")
            elif r.missing_components:
                print(f"    Missing: {r.missing_components}")


def save_report(report: EvaluationReport, path: str) -> None:
    """Save evaluation report to JSON file."""
    data = {
        "summary": {
            "total_tests": report.total_tests,
            "successful_generations": report.successful_generations,
            "syntax_valid": report.syntax_valid,
            "endpoint_valid": report.endpoint_valid,
            "avg_generation_time": report.avg_generation_time,
            "avg_component_score": report.avg_component_score,
            "pattern_detection_accuracy": report.pattern_detection_accuracy,
        },
        "by_category": report.results_by_category,
        "by_pattern": {
            k: {
                "total": v["total"],
                "success": v["success"],
                "avg_component_score": sum(v["component_scores"]) / len(v["component_scores"]) if v["component_scores"] else 0
            }
            for k, v in report.results_by_pattern.items()
        },
        "test_results": [
            {
                "test_id": r.test_id,
                "category": r.category,
                "patterns": r.patterns,
                "question": r.question,
                "generated_sparql": r.generated_sparql,
                "syntax_valid": r.syntax_valid,
                "endpoint_valid": r.endpoint_valid,
                "endpoint_error": r.endpoint_error,
                "result_count": r.result_count,
                "component_score": r.component_score,
                "matched_components": r.matched_components,
                "missing_components": r.missing_components,
                "detected_patterns": r.detected_patterns,
                "pattern_detection_correct": r.pattern_detection_correct,
                "generation_time": r.generation_time,
                "generation_error": r.generation_error,
            }
            for r in report.test_results
        ]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

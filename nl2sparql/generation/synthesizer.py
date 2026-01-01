"""Main NL2SPARQL synthesizer class."""

from pathlib import Path
from typing import Optional, Union

from .. import (
    QueryExample,
    RetrievalResult,
    ValidationResult,
    TranslationResult,
)
from ..config import LIITA_ENDPOINT, DATASET_PATH
from ..llm.base import get_client, LLMClient
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.patterns import infer_patterns, get_top_patterns
from ..validation.syntax import validate_syntax
from ..validation.endpoint import validate_endpoint
from ..validation.semantic import validate_semantic
from .adapters import adapt_query, synthesize_query, fix_query


class NL2SPARQL:
    """
    Main class for translating natural language questions to SPARQL queries.

    This class orchestrates the full pipeline:
    1. Pattern inference from the question
    2. Retrieval of relevant examples
    3. Query synthesis or adaptation
    4. Validation and optional fixing

    Example:
        >>> translator = NL2SPARQL(provider="openai", model="gpt-4.1")
        >>> result = translator.translate("Quali lemmi esprimono tristezza?")
        >>> print(result.sparql)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        retriever_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
        validate: bool = True,
        fix_errors: bool = True,
        max_retries: int = 3,
        endpoint: str = LIITA_ENDPOINT,
    ):
        """
        Initialize the NL2SPARQL translator.

        Args:
            provider: LLM provider ("openai", "anthropic", "mistral", "gemini", "ollama")
            model: Model name (uses provider default if not specified)
            api_key: API key (uses environment variable if not provided)
            dataset_path: Path to the example dataset
            embedding_model: Sentence-transformer model for retrieval
            retriever_weights: Weights for (semantic, bm25, pattern) scores
            validate: Whether to validate generated queries
            fix_errors: Whether to attempt to fix invalid queries
            max_retries: Maximum fix attempts
            endpoint: SPARQL endpoint for validation
        """
        # Initialize LLM client
        self.client = get_client(provider, model, api_key)

        # Initialize retriever
        self.retriever = HybridRetriever(
            dataset_path=dataset_path or DATASET_PATH,
            embedding_model=embedding_model,
            weights=retriever_weights,
        )

        # Configuration
        self.validate = validate
        self.fix_errors = fix_errors
        self.max_retries = max_retries
        self.endpoint = endpoint

        # Thresholds
        self.adaptation_threshold = 0.85  # Score above which we adapt instead of synthesize

    def translate(
        self,
        question: str,
        top_k: int = 5,
        validate: Optional[bool] = None,
        fix_errors: Optional[bool] = None,
    ) -> TranslationResult:
        """
        Translate a natural language question to SPARQL.

        Args:
            question: Natural language question (typically in Italian)
            top_k: Number of examples to retrieve
            validate: Override instance validate setting
            fix_errors: Override instance fix_errors setting

        Returns:
            TranslationResult with the SPARQL query and metadata
        """
        should_validate = validate if validate is not None else self.validate
        should_fix = fix_errors if fix_errors is not None else self.fix_errors

        # Step 1: Infer patterns from question
        inferred_patterns = infer_patterns(question)
        detected_patterns = get_top_patterns(inferred_patterns, top_k=5)

        # Step 2: Retrieve relevant examples
        retrieved = self.retriever.retrieve(
            query=question,
            user_patterns=inferred_patterns,
            top_k=top_k,
        )

        # Step 3: Decide generation strategy
        exemplars = [r.example for r in retrieved]
        best_score = retrieved[0].score if retrieved else 0.0

        if best_score >= self.adaptation_threshold and retrieved:
            # High similarity - adapt the best example
            sparql = adapt_query(
                example=retrieved[0].example,
                user_question=question,
                client=self.client,
                detected_patterns=detected_patterns,
            )
        else:
            # Synthesize from multiple examples
            sparql = synthesize_query(
                user_question=question,
                exemplars=exemplars,
                client=self.client,
                detected_patterns=detected_patterns,
            )

        # Step 4: Validate and optionally fix
        validation_result = None
        was_fixed = False
        fix_attempts = 0

        if should_validate:
            validation_result, sparql, was_fixed, fix_attempts = self._validate_and_fix(
                sparql=sparql,
                detected_patterns=detected_patterns,
                should_fix=should_fix,
            )

        # Compute confidence
        confidence = self._compute_confidence(retrieved, validation_result)

        return TranslationResult(
            question=question,
            sparql=sparql,
            validation=validation_result,
            retrieved_examples=retrieved,
            detected_patterns=detected_patterns,
            confidence=confidence,
            was_fixed=was_fixed,
            fix_attempts=fix_attempts,
        )

    def _validate_and_fix(
        self,
        sparql: str,
        detected_patterns: list[str],
        should_fix: bool,
    ) -> tuple[ValidationResult, str, bool, int]:
        """
        Validate and optionally fix a query.

        Returns:
            Tuple of (validation_result, final_sparql, was_fixed, fix_attempts)
        """
        was_fixed = False
        fix_attempts = 0

        for attempt in range(self.max_retries + 1):
            # Syntax validation
            syntax_valid, syntax_error = validate_syntax(sparql)

            if not syntax_valid:
                if should_fix and attempt < self.max_retries:
                    sparql = fix_query(
                        sparql=sparql,
                        error=syntax_error or "Syntax error",
                        client=self.client,
                        detected_patterns=detected_patterns,
                    )
                    fix_attempts += 1
                    was_fixed = True
                    continue
                else:
                    return (
                        ValidationResult(
                            syntax_valid=False,
                            syntax_error=syntax_error,
                        ),
                        sparql,
                        was_fixed,
                        fix_attempts,
                    )

            # Semantic validation
            semantic_valid, semantic_errors = validate_semantic(sparql, detected_patterns)

            # Endpoint validation (optional, may be slow)
            exec_success, exec_error, result_count, sample = validate_endpoint(
                sparql=sparql,
                endpoint=self.endpoint,
            )

            # If execution failed but syntax is OK, try to fix
            if not exec_success and should_fix and attempt < self.max_retries:
                error_msg = exec_error or "Execution failed"
                if semantic_errors:
                    error_msg += "; Semantic issues: " + "; ".join(semantic_errors)

                sparql = fix_query(
                    sparql=sparql,
                    error=error_msg,
                    client=self.client,
                    detected_patterns=detected_patterns,
                )
                fix_attempts += 1
                was_fixed = True
                continue

            # Return final validation result
            return (
                ValidationResult(
                    syntax_valid=True,
                    syntax_error=None,
                    execution_success=exec_success,
                    execution_error=exec_error,
                    result_count=result_count,
                    sample_results=sample,
                    semantic_errors=semantic_errors,
                ),
                sparql,
                was_fixed,
                fix_attempts,
            )

        # Should not reach here, but return last state
        return (
            ValidationResult(
                syntax_valid=syntax_valid,
                syntax_error=syntax_error if not syntax_valid else None,
            ),
            sparql,
            was_fixed,
            fix_attempts,
        )

    def _compute_confidence(
        self,
        retrieved: list[RetrievalResult],
        validation: Optional[ValidationResult],
    ) -> float:
        """
        Compute confidence score for the translation.

        Args:
            retrieved: Retrieved examples
            validation: Validation result

        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved:
            return 0.0

        # Base confidence from retrieval scores
        top_score = retrieved[0].score
        avg_score = sum(r.score for r in retrieved) / len(retrieved)
        retrieval_confidence = 0.6 * top_score + 0.4 * avg_score

        # Adjust for validation
        if validation:
            if validation.is_valid:
                # Boost for valid queries with results
                if validation.result_count and validation.result_count > 0:
                    retrieval_confidence = min(1.0, retrieval_confidence * 1.2)
            else:
                # Penalty for invalid queries
                retrieval_confidence *= 0.5

        return min(1.0, retrieval_confidence)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant examples without generating a query.

        Useful for inspection and debugging.

        Args:
            question: Natural language question
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        inferred_patterns = infer_patterns(question)
        return self.retriever.retrieve(
            query=question,
            user_patterns=inferred_patterns,
            top_k=top_k,
        )

    def validate_query(self, sparql: str) -> ValidationResult:
        """
        Validate a SPARQL query without generating.

        Args:
            sparql: The SPARQL query to validate

        Returns:
            ValidationResult with all validation checks
        """
        syntax_valid, syntax_error = validate_syntax(sparql)

        if not syntax_valid:
            return ValidationResult(
                syntax_valid=False,
                syntax_error=syntax_error,
            )

        semantic_valid, semantic_errors = validate_semantic(sparql)

        exec_success, exec_error, result_count, sample = validate_endpoint(
            sparql=sparql,
            endpoint=self.endpoint,
        )

        return ValidationResult(
            syntax_valid=True,
            syntax_error=None,
            execution_success=exec_success,
            execution_error=exec_error,
            result_count=result_count,
            sample_results=sample,
            semantic_errors=semantic_errors,
        )

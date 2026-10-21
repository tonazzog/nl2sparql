"""Persistent translation memory: learned examples and documented failures.

Biases retrieval and generation in the agent and synthesizer pipelines toward
previously-correct answers and away from previously-made mistakes. This store
is bias-only by design: nothing here ever skips fresh validation against the
live endpoint - a remembered match is just a strong candidate fed into the
existing retrieval/prompt machinery, not a shortcut around it.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .matching import FAILURE_RELEVANCE_THRESHOLD, NEAR_DUPLICATE_THRESHOLD, normalize_question
from .records import FailureRecord, LearnedExample, append_jsonl, read_jsonl


def _normalize_sparql(sparql: str) -> str:
    return " ".join(sparql.split())


def _example_key(ex: LearnedExample) -> tuple[str, str]:
    return (ex.nl_normalized, _normalize_sparql(ex.sparql))


def _failure_key(fr: FailureRecord) -> tuple[str, str, str]:
    return (fr.nl_normalized, _normalize_sparql(fr.sparql), fr.error)


def _dedupe(hits: list[tuple], key) -> list[tuple]:
    """Keep the first (highest-scored, if pre-sorted) hit for each content key."""
    seen = set()
    unique = []
    for record, score in hits:
        k = key(record)
        if k not in seen:
            seen.add(k)
            unique.append((record, score))
    return unique


@dataclass
class MemoryLookupResult:
    """Result of a TranslationMemory.lookup() call, ranked by similarity score."""

    examples: list[tuple[LearnedExample, float]] = field(default_factory=list)
    failures: list[tuple[FailureRecord, float]] = field(default_factory=list)


class TranslationMemory:
    """Persistent store of successful translations and documented failures."""

    def __init__(
        self,
        memory_path: Optional[Union[str, Path]] = None,
        embedding_model: Optional[str] = None,
    ):
        from ..config import DEFAULT_EMBEDDING_MODEL, DEFAULT_MEMORY_PATH

        self.path = Path(memory_path) if memory_path else DEFAULT_MEMORY_PATH
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self._examples_path = self.path / "examples.jsonl"
        self._failures_path = self.path / "failures.jsonl"

        self._examples: list[LearnedExample] = read_jsonl(self._examples_path, LearnedExample)
        self._failures: list[FailureRecord] = read_jsonl(self._failures_path, FailureRecord)

        self._lock = threading.Lock()
        self._embedding_index = None  # lazy; rebuilt on first lookup after a write
        self._index_dirty = True
        self._n_examples = 0

    # ---- id allocation ----

    @staticmethod
    def _new_id() -> int:
        """Random negative ID. Negative avoids colliding with the curated corpus
        (IDs 1..N); random (rather than min-1) avoids collisions between separate
        processes - e.g. the MCP server and CLI runs - sharing one store."""
        return -(uuid.uuid4().int >> 66)  # 62 random bits, fits a signed int64

    # ---- lazy embedding index, rebuilt on first lookup after a write ----

    def _ensure_index(self) -> None:
        if not self._index_dirty and self._embedding_index is not None:
            return

        texts = [e.nl for e in self._examples] + [f.nl for f in self._failures]
        self._n_examples = len(self._examples)

        if not texts:
            self._embedding_index = None
            self._index_dirty = False
            return

        from ..retrieval.embeddings import EmbeddingIndex

        index = EmbeddingIndex(model_name=self.embedding_model)
        index.build_index(texts)
        self._embedding_index = index
        self._index_dirty = False

    # ---- lookup (read path) ----

    def lookup(
        self,
        question: str,
        patterns: Optional[dict] = None,
        top_k: int = 3,
    ) -> MemoryLookupResult:
        """Find remembered examples/failures similar to `question`.

        Combines an O(1) exact-normalized-match tier with embedding-based
        near-duplicate matching, so both "asked again verbatim" and
        "asked again as a paraphrase" are recognized.

        Examples are only returned when they are exact matches or score at
        least NEAR_DUPLICATE_THRESHOLD, since they compete with the curated
        corpus for few-shot slots. Failures use the looser
        FAILURE_RELEVANCE_THRESHOLD: a loosely related past mistake is still
        useful "avoid this" context, but one from an unrelated question is noise.
        Records with identical content (same question and query) are returned
        once, so duplicates can't crowd out the few-shot slots.
        """
        result = MemoryLookupResult()
        if not self._examples and not self._failures:
            return result

        normalized = normalize_question(question)

        # Held for the whole read so a concurrent write can't swap the index
        # (and the example/failure split) out from under the search.
        with self._lock:
            exact_examples = [e for e in self._examples if e.nl_normalized == normalized]
            self._ensure_index()

            if self._embedding_index is None:
                result.examples = _dedupe(
                    [(e, 1.0) for e in exact_examples], _example_key
                )[:top_k]
                return result

            n_records = len(self._examples) + len(self._failures)
            scored = self._embedding_index.search(question, top_k=min(n_records, top_k * 4 + len(exact_examples)))

            example_hits: list[tuple[LearnedExample, float]] = [(e, 1.0) for e in exact_examples]
            failure_hits: list[tuple[FailureRecord, float]] = []

            for idx, score in scored:
                if idx < self._n_examples:
                    if score >= NEAR_DUPLICATE_THRESHOLD:
                        example_hits.append((self._examples[idx], float(score)))
                elif score >= FAILURE_RELEVANCE_THRESHOLD:
                    failure_hits.append((self._failures[idx - self._n_examples], float(score)))

        # Sort first so the dedupe keeps each record's best score (exact = 1.0)
        example_hits.sort(key=lambda pair: -pair[1])
        failure_hits.sort(key=lambda pair: -pair[1])
        example_hits = _dedupe(example_hits, _example_key)
        failure_hits = _dedupe(failure_hits, _failure_key)

        result.examples = example_hits[:top_k]
        result.failures = failure_hits[:top_k]
        return result

    def retrieve_as_examples(self, question: str, top_k: int = 3) -> list:
        """Wrap matching LearnedExamples as RetrievalResult, for trivial merging
        with HybridRetriever.retrieve()'s output (same return type)."""
        from .. import RetrievalResult

        hit = self.lookup(question, top_k=top_k)
        return [
            RetrievalResult(
                example=ex.to_query_example(),
                score=score,
                semantic_score=score,
                bm25_score=0.0,
                pattern_score=0.0,
            )
            for ex, score in hit.examples
        ]

    # ---- writes (called at the end of a run) ----

    def record_success(
        self,
        *,
        question: str,
        sparql: str,
        patterns: Optional[dict] = None,
        language: str = "it",
        confidence: float = 0.0,
        result_count: int = 0,
        source_pipeline: str = "synthesizer",
        provider: str = "",
        model: str = "",
    ) -> None:
        """Remember a successful translation. A success identical to one
        already stored (same normalized question and query) is not stored
        again - the existing record already carries the signal, and copies
        would only crowd the few-shot slots."""
        key = (normalize_question(question), _normalize_sparql(sparql))
        with self._lock:
            if any(_example_key(e) == key for e in self._examples):
                return
            example = LearnedExample(
                id=self._new_id(),
                nl=question,
                sparql=sparql,
                patterns=dict(patterns or {}),
                language=language,
                confidence=confidence,
                result_count=result_count,
                source_pipeline=source_pipeline,
                provider=provider or "",
                model=model or "",
            )
            self._examples.append(example)
            append_jsonl(self._examples_path, example)
            self._index_dirty = True

    def record_failure(
        self,
        *,
        question: str,
        error: str,
        sparql: str = "",
        patterns: Optional[list] = None,
        language: str = "it",
        result_count: int = 0,
        source_pipeline: str = "synthesizer",
        provider: str = "",
        model: str = "",
        run_id: str = "",
        kind: str = "unresolved",
    ) -> None:
        with self._lock:
            record = FailureRecord(
                id=self._new_id(),
                nl=question,
                error=error,
                sparql=sparql,
                detected_patterns=list(patterns or []),
                kind=kind,
                result_count=result_count,
                language=language,
                source_pipeline=source_pipeline,
                provider=provider or "",
                model=model or "",
                run_id=run_id,
            )
            self._failures.append(record)
            append_jsonl(self._failures_path, record)
            self._index_dirty = True

    def record_unresolved_run(
        self,
        *,
        question: str,
        refinement_history: list[dict],
        patterns: Optional[list] = None,
        language: str = "it",
        source_pipeline: str = "agent",
        provider: str = "",
        model: str = "",
    ) -> None:
        """Persist a whole agent run that never converged: one 'attempt'
        FailureRecord per attempt, plus one 'unresolved' record summarizing
        the run.

        `refinement_history` should include the run's final attempt: the
        agent only appends to its own history before a retry, so the last
        (failing) query is otherwise missing. The summary record carries that
        last attempt's query and error, so it stays actionable when it is
        later shown to the LLM as a past mistake."""
        run_id = uuid.uuid4().hex[:12]
        last = refinement_history[-1] if refinement_history else {}
        for attempt in refinement_history:
            self.record_failure(
                question=question,
                error=attempt.get("error", "unknown error"),
                sparql=attempt.get("sparql", ""),
                patterns=patterns,
                language=language,
                result_count=attempt.get("result_count", 0),
                source_pipeline=source_pipeline,
                provider=provider,
                model=model,
                run_id=run_id,
                kind="attempt",
            )
        self.record_failure(
            question=question,
            error=(
                f"Run did not converge after {len(refinement_history)} attempt(s). "
                f"Last error: {last.get('error') or 'unknown error'}"
            ),
            sparql=last.get("sparql", ""),
            result_count=last.get("result_count", 0),
            patterns=patterns,
            language=language,
            source_pipeline=source_pipeline,
            provider=provider,
            model=model,
            run_id=run_id,
            kind="unresolved",
        )

    # ---- introspection ----

    def stats(self) -> dict:
        return {
            "path": str(self.path),
            "examples": len(self._examples),
            "failures": len(self._failures),
        }

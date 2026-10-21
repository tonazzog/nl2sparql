"""Persisted record types for the translation memory store.

Records are stored as JSONL (one JSON object per line) - appends are O(1)
and crash-safe, and the format is human-inspectable, matching the project's
existing convention of plain JSON data files for the curated example corpus.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .matching import normalize_question

if TYPE_CHECKING:
    from .. import QueryExample


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LearnedExample:
    """A successful natural language -> SPARQL translation, remembered for reuse.

    Deliberately a superset of QueryExample (nl2sparql/__init__.py) so it can
    be wrapped as one with zero translation - see to_query_example().
    """

    id: int
    nl: str
    sparql: str
    nl_normalized: str = ""
    patterns: dict[str, float] = field(default_factory=dict)
    language: str = "it"
    confidence: float = 0.0
    result_count: int = 0
    source_pipeline: str = "synthesizer"  # "synthesizer" or "agent"
    provider: str = ""
    model: str = ""
    created_at: str = ""
    hit_count: int = 0

    def __post_init__(self):
        if not self.nl_normalized:
            self.nl_normalized = normalize_question(self.nl)
        if not self.created_at:
            self.created_at = _now_iso()

    def to_query_example(self) -> "QueryExample":
        """Wrap as a QueryExample for reuse by the existing retrieval/prompt pipeline."""
        from .. import QueryExample

        return QueryExample(id=self.id, sparql=self.sparql, nl=self.nl, patterns=self.patterns)


@dataclass
class FailureRecord:
    """A documented mistake: a translation attempt that did not succeed.

    kind="attempt" records one entry of an agent run's refinement_history;
    kind="unresolved" summarizes a whole run that never converged.
    """

    id: int
    nl: str
    error: str
    nl_normalized: str = ""
    sparql: str = ""
    detected_patterns: list[str] = field(default_factory=list)
    kind: str = "unresolved"
    result_count: int = 0
    language: str = "it"
    source_pipeline: str = "synthesizer"
    provider: str = ""
    model: str = ""
    run_id: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.nl_normalized:
            self.nl_normalized = normalize_question(self.nl)
        if not self.created_at:
            self.created_at = _now_iso()


def read_jsonl(path: Path, record_type: type) -> list:
    """Read a JSONL file into a list of `record_type` instances.

    Returns [] if the file doesn't exist. Lines that are not valid JSON (e.g.
    a partial last line left by a process killed mid-write) or that don't fit
    the record schema are skipped with a warning, so one bad line can't make
    the whole store - and every translation that uses it - fail to load.
    """
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(record_type(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                warnings.warn(f"Skipping unreadable record at {path}:{lineno} ({e})")
    return records


def append_jsonl(path: Path, record) -> None:
    """Append one dataclass record as a JSON line, creating the parent dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # If a previous write was cut off mid-line, start on a fresh line so the
    # new record isn't glued onto the partial one.
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

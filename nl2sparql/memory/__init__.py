"""Persistent translation memory for nl2sparql.

Lets the agent and synthesizer pipelines bias future translations toward
previously-correct answers and away from previously-made mistakes, across
process restarts - without ever skipping fresh validation against the live
endpoint.

Example:
    >>> from nl2sparql.memory import TranslationMemory
    >>> memory = TranslationMemory()
    >>> memory.record_success(question="...", sparql="...")
    >>> hit = memory.lookup("...")
"""

from .matching import NEAR_DUPLICATE_THRESHOLD, normalize_question
from .records import FailureRecord, LearnedExample
from .store import MemoryLookupResult, TranslationMemory

__all__ = [
    "TranslationMemory",
    "MemoryLookupResult",
    "LearnedExample",
    "FailureRecord",
    "normalize_question",
    "NEAR_DUPLICATE_THRESHOLD",
]

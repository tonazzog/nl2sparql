"""Question normalization and duplicate-matching thresholds for translation memory."""

import re

# Cosine similarity above which two questions are treated as near-duplicates
# for the purposes of memory-biased retrieval.
NEAR_DUPLICATE_THRESHOLD = 0.93

# Looser cosine similarity above which a remembered failure is considered
# relevant enough to show the LLM as a past mistake to avoid. Failures only add
# prompt context (they never replace a decision), so they can be looser than
# examples - but without any floor, mistakes from unrelated questions leak in.
FAILURE_RELEVANCE_THRESHOLD = 0.75


def normalize_question(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used for the cheap exact-match tier before falling back to embedding
    similarity - two questions that normalize identically are treated as
    the same question regardless of capitalization/punctuation.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

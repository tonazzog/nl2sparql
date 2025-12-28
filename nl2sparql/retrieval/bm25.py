"""BM25 ranking with pattern boosting."""

import re
from typing import Optional

from rank_bm25 import BM25Okapi


# Italian stopwords
ITALIAN_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "o", "ma", "se", "che", "chi", "cui", "dove", "come", "quando",
    "sono", "essere", "avere", "fare", "dire", "andare", "venire",
    "questo", "quello", "quale", "quanto", "tutto", "molto", "poco",
    "non", "piu", "anche", "solo", "gia", "ancora", "sempre", "mai",
    "perche", "cosa", "quali", "quanti", "quante",
}


def tokenize(
    text: str,
    remove_stopwords: bool = True,
    exclude_terms: Optional[set[str]] = None,
) -> list[str]:
    """
    Tokenize text for BM25 indexing.

    Args:
        text: Text to tokenize
        remove_stopwords: Whether to remove Italian stopwords
        exclude_terms: Additional terms to exclude (e.g., entity terms)

    Returns:
        List of tokens
    """
    # Lowercase and extract words
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)

    # Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ITALIAN_STOPWORDS]

    # Remove excluded terms
    if exclude_terms:
        exclude_lower = {t.lower() for t in exclude_terms}
        tokens = [t for t in tokens if t not in exclude_lower]

    return tokens


class BM25WithPatternBoost:
    """BM25 ranking with pattern-based score boosting."""

    def __init__(
        self,
        documents: list[str],
        patterns_list: list[dict[str, float]],
        pattern_boost_weight: float = 0.3,
    ):
        """
        Initialize BM25 index with pattern boosting.

        Args:
            documents: List of document texts to index
            patterns_list: List of pattern dicts for each document
            pattern_boost_weight: Weight for pattern similarity in final score
        """
        self.documents = documents
        self.patterns_list = patterns_list
        self.pattern_boost_weight = pattern_boost_weight

        # Tokenize and build BM25 index
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _compute_pattern_similarity(
        self,
        query_patterns: dict[str, float],
        doc_patterns: dict[str, float],
    ) -> float:
        """
        Compute similarity between query patterns and document patterns.

        Args:
            query_patterns: Patterns inferred from query
            doc_patterns: Patterns of the document

        Returns:
            Similarity score between 0 and 1
        """
        if not query_patterns or not doc_patterns:
            return 0.0

        # Sum of matching pattern weights
        score = 0.0
        for pattern, weight in query_patterns.items():
            if pattern in doc_patterns:
                score += weight * doc_patterns[pattern]

        # Normalize by total possible score
        max_score = sum(query_patterns.values())
        if max_score > 0:
            return min(1.0, score / max_score)
        return 0.0

    def score(
        self,
        query: str,
        query_patterns: Optional[dict[str, float]] = None,
        top_k: int = 10,
        exclude_terms: Optional[set[str]] = None,
    ) -> list[tuple[int, float, float, float]]:
        """
        Score documents for a query.

        Args:
            query: Query text
            query_patterns: Optional patterns inferred from query
            top_k: Number of top results to return
            exclude_terms: Terms to exclude from tokenization

        Returns:
            List of (doc_index, total_score, bm25_score, pattern_score) tuples
        """
        # Tokenize query
        query_tokens = tokenize(query, exclude_terms=exclude_terms)

        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_normalized = [s / max_bm25 for s in bm25_scores]

        # Compute combined scores
        results = []
        for i, (bm25_score, doc_patterns) in enumerate(
            zip(bm25_normalized, self.patterns_list)
        ):
            pattern_score = 0.0
            if query_patterns:
                pattern_score = self._compute_pattern_similarity(
                    query_patterns, doc_patterns
                )

            # Combine scores
            total_score = (
                (1 - self.pattern_boost_weight) * bm25_score
                + self.pattern_boost_weight * pattern_score
            )

            results.append((i, total_score, bm25_score, pattern_score))

        # Sort by total score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

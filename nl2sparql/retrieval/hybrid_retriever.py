"""Hybrid retriever combining semantic, BM25, and pattern-based retrieval."""

import json
from pathlib import Path
from typing import Optional, Union

from .. import QueryExample, RetrievalResult
from ..config import DATASET_PATH
from .bm25 import BM25WithPatternBoost
from .embeddings import EmbeddingIndex
from .patterns import infer_patterns, extract_entity_terms


def load_dataset(path: Union[str, Path]) -> list[QueryExample]:
    """
    Load the query dataset from JSON.

    Args:
        path: Path to the JSON dataset file

    Returns:
        List of QueryExample objects
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = QueryExample(
            id=item.get("id", 0),
            sparql=item.get("sparql", ""),
            nl=item.get("nl", ""),
            nl_variants=item.get("nl_variants", {}),
            patterns=item.get("patterns", {}),
        )
        examples.append(example)

    return examples


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Semantic similarity (sentence-transformers + FAISS)
    - Lexical matching (BM25)
    - Pattern-based boosting

    This retriever finds the most relevant example queries for a given
    natural language question, which are then used for few-shot prompting.
    """

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        query_db: Optional[list[QueryExample]] = None,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        """
        Initialize the hybrid retriever.

        Args:
            dataset_path: Path to the JSON dataset (uses default if not provided)
            query_db: Pre-loaded query examples (alternative to dataset_path)
            embedding_model: Sentence-transformer model for embeddings
            weights: Weights for (semantic, bm25, pattern) scores
        """
        # Load dataset
        if query_db is not None:
            self.query_db = query_db
        elif dataset_path is not None:
            self.query_db = load_dataset(dataset_path)
        else:
            self.query_db = load_dataset(DATASET_PATH)

        self.weights = weights
        self.w_semantic, self.w_bm25, self.w_pattern = weights

        # Build document texts for indexing
        # Combine NL and variants for better matching
        self._doc_texts = []
        for ex in self.query_db:
            text = ex.nl
            if ex.nl_variants:
                variants = " ".join(ex.nl_variants.values())
                text = f"{text} {variants}"
            self._doc_texts.append(text)

        # Initialize embedding index
        self._embedding_index = EmbeddingIndex(model_name=embedding_model)
        self._embedding_index.build_index(self._doc_texts)

        # Initialize BM25 index
        patterns_list = [ex.patterns for ex in self.query_db]
        self._bm25_index = BM25WithPatternBoost(
            documents=self._doc_texts,
            patterns_list=patterns_list,
            pattern_boost_weight=0.2,
        )

    def retrieve(
        self,
        query: str,
        user_patterns: Optional[dict[str, float]] = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant examples for a query.

        Args:
            query: Natural language query
            user_patterns: Optional pre-computed patterns (inferred if not provided)
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        # Infer patterns if not provided
        if user_patterns is None:
            user_patterns = infer_patterns(query)

        # Extract entity terms to exclude from BM25
        entity_terms = set(extract_entity_terms(query))

        # Get semantic search results
        semantic_results = self._embedding_index.search(query, top_k=top_k * 2)
        semantic_scores = {idx: score for idx, score in semantic_results}

        # Get BM25 results
        bm25_results = self._bm25_index.score(
            query,
            query_patterns=user_patterns,
            top_k=top_k * 2,
            exclude_terms=entity_terms,
        )
        bm25_scores = {idx: (total, bm25, pattern) for idx, total, bm25, pattern in bm25_results}

        # Combine all candidate indices
        all_indices = set(semantic_scores.keys()) | set(bm25_scores.keys())

        # Compute combined scores
        results = []
        for idx in all_indices:
            semantic_score = semantic_scores.get(idx, 0.0)
            bm25_total, bm25_raw, pattern_score = bm25_scores.get(idx, (0.0, 0.0, 0.0))

            # Weighted combination
            combined_score = (
                self.w_semantic * semantic_score
                + self.w_bm25 * bm25_raw
                + self.w_pattern * pattern_score
            )

            result = RetrievalResult(
                example=self.query_db[idx],
                score=combined_score,
                semantic_score=semantic_score,
                bm25_score=bm25_raw,
                pattern_score=pattern_score,
            )
            results.append(result)

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def get_example_by_id(self, example_id: int) -> Optional[QueryExample]:
        """
        Get a specific example by ID.

        Args:
            example_id: The example ID

        Returns:
            QueryExample or None if not found
        """
        for ex in self.query_db:
            if ex.id == example_id:
                return ex
        return None

"""Embedding-based semantic search using sentence-transformers and FAISS."""

from typing import Optional

import numpy as np


def resolve_local_model_path(model_name: str) -> str:
    """Resolve a sentence-transformers model name to its local cache path if fully cached.

    Loading from a resolved local directory (rather than a bare repo id) makes
    SentenceTransformer skip Hub network calls entirely, including the
    `transformers` mistral-tokenizer-regex check that ignores HF_HUB_OFFLINE
    and otherwise fails hard on networks that block huggingface.co.
    Falls back to the original model_name (unchanged online/cache behavior)
    if the model isn't already downloaded.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name

    candidates = (
        [model_name]
        if "/" in model_name
        else [f"sentence-transformers/{model_name}", model_name]
    )
    for repo_id in candidates:
        try:
            return snapshot_download(repo_id, local_files_only=True)
        except Exception:
            continue
    return model_name


class EmbeddingIndex:
    """FAISS index for semantic similarity search."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize the embedding index.

        Args:
            model_name: Sentence-transformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError:
            raise ImportError(
                "Required packages not installed. "
                "Install with: pip install sentence-transformers faiss-cpu"
            )

        self.model_name = model_name
        self.encoder = SentenceTransformer(resolve_local_model_path(model_name))
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, texts: list[str]) -> None:
        """
        Build FAISS index from texts.

        Args:
            texts: List of texts to index
        """
        import faiss

        # Encode texts
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.embeddings = embeddings

        # Build FAISS index (inner product for cosine similarity on normalized vectors)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (document_index, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Search
        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k,
        )

        # Return results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                results.append((int(idx), float(score)))

        return results

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

"""Ontology-based retrieval for discovering relevant classes and properties.

This module provides semantic search over ontology definitions to help
the agent discover appropriate classes and properties when generating SPARQL.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .embeddings import EmbeddingIndex


@dataclass
class OntologyEntry:
    """An ontology class or property with its metadata."""

    id: str  # URI
    type: Literal["class", "property"]
    label: str
    short_text: str
    detailed_text: str
    query_oriented_text: str
    searchable_text: str
    metadata: dict

    @property
    def uri(self) -> str:
        return self.id

    @property
    def prefix_local(self) -> str:
        """Get prefix:localName format (e.g., 'lexinfo:hypernym')."""
        uri = self.id
        # Common prefix mappings
        prefixes = {
            "http://www.lexinfo.net/ontology/3.0/lexinfo#": "lexinfo:",
            "http://www.w3.org/ns/lemon/ontolex#": "ontolex:",
            "http://www.w3.org/ns/lemon/vartrans#": "vartrans:",
            "http://www.w3.org/ns/lemon/lime#": "lime:",
            "http://www.w3.org/ns/lemon/synsem#": "synsem:",
            "http://www.w3.org/2004/02/skos/core#": "skos:",
            "http://purl.org/dc/terms/": "dcterms:",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
            "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
            "http://lila-erc.eu/ontologies/lila/": "lila:",
            "http://w3id.org/elita/": "elita:",
            "http://www.gsi.upm.es/ontologies/marl/ns#": "marl:",
        }
        for ns, prefix in prefixes.items():
            if uri.startswith(ns):
                return prefix + uri[len(ns):]
        return uri


@dataclass
class RetrievedOntologyEntry:
    """An ontology entry with its retrieval score."""

    entry: OntologyEntry
    score: float


class OntologyRetriever:
    """Retrieves relevant ontology classes and properties using semantic search.

    This helps the agent discover appropriate vocabulary when generating SPARQL,
    especially for semantic relations (hypernym, hyponym, etc.) and domain-specific
    properties.
    """

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize the ontology retriever.

        Args:
            catalog_path: Path to ontology.json catalog. If None, uses default.
            embedding_model: Sentence-transformer model for semantic search.
        """
        if catalog_path is None:
            catalog_path = Path(__file__).parent.parent / "data" / "ontology.json"

        self.catalog_path = Path(catalog_path)
        self.embedding_model = embedding_model

        # Load catalog
        self.entries: list[OntologyEntry] = []
        self._load_catalog()

        # Build search index
        self.index = EmbeddingIndex(model_name=embedding_model)
        self._build_index()

    def _load_catalog(self) -> None:
        """Load ontology catalog from JSON file."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Ontology catalog not found: {self.catalog_path}")

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = data.get("documents", [])

        for doc in documents:
            entry = OntologyEntry(
                id=doc["id"],
                type=doc["type"],
                label=doc.get("metadata", {}).get("label", ""),
                short_text=doc.get("short_text", ""),
                detailed_text=doc.get("detailed_text", ""),
                query_oriented_text=doc.get("query_oriented_text", ""),
                searchable_text=doc.get("searchable_text", ""),
                metadata=doc.get("metadata", {}),
            )
            self.entries.append(entry)

    def _build_index(self) -> None:
        """Build semantic search index from searchable_text."""
        if not self.entries:
            return

        # Use searchable_text for embedding (optimized for search)
        texts = [entry.searchable_text for entry in self.entries]
        self.index.build_index(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        entry_type: Literal["all", "class", "property"] = "all",
    ) -> list[RetrievedOntologyEntry]:
        """
        Retrieve ontology entries relevant to the query.

        Args:
            query: Natural language query (e.g., "broader meaning", "part of speech")
            top_k: Maximum number of results
            entry_type: Filter by type ("all", "class", "property")

        Returns:
            List of retrieved entries with scores, sorted by relevance.
        """
        if not self.entries:
            return []

        # Get more results if filtering by type
        search_k = top_k * 3 if entry_type != "all" else top_k

        # Semantic search
        results = self.index.search(query, top_k=search_k)

        retrieved = []
        for idx, score in results:
            if idx < len(self.entries):
                entry = self.entries[idx]

                # Filter by type if specified
                if entry_type != "all" and entry.type != entry_type:
                    continue

                retrieved.append(RetrievedOntologyEntry(entry=entry, score=score))

                if len(retrieved) >= top_k:
                    break

        return retrieved

    def retrieve_properties(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedOntologyEntry]:
        """Retrieve only properties relevant to the query."""
        return self.retrieve(query, top_k=top_k, entry_type="property")

    def retrieve_classes(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedOntologyEntry]:
        """Retrieve only classes relevant to the query."""
        return self.retrieve(query, top_k=top_k, entry_type="class")

    def format_for_prompt(
        self,
        entries: list[RetrievedOntologyEntry],
        include_examples: bool = True,
    ) -> str:
        """
        Format retrieved entries for inclusion in an LLM prompt.

        Args:
            entries: Retrieved ontology entries
            include_examples: Whether to include SPARQL usage examples

        Returns:
            Formatted string for prompt inclusion
        """
        if not entries:
            return ""

        lines = ["## Relevant Ontology Terms\n"]

        for item in entries:
            entry = item.entry
            prefix_local = entry.prefix_local

            if entry.type == "property":
                lines.append(f"### Property: {prefix_local}")
                lines.append(f"- URI: <{entry.uri}>")
                lines.append(f"- Description: {entry.short_text}")

                # Add domain/range if available
                domains = entry.metadata.get("domains", [])
                ranges = entry.metadata.get("ranges", [])
                if domains:
                    domain_str = ", ".join(d.split("#")[-1] for d in domains)
                    lines.append(f"- Domain (subject type): {domain_str}")
                if ranges:
                    range_str = ", ".join(r.split("#")[-1] for r in ranges)
                    lines.append(f"- Range (object type): {range_str}")

                # Add inverse if available
                inverses = entry.metadata.get("inverses", [])
                if inverses:
                    inv_str = ", ".join(i.split("#")[-1] for i in inverses)
                    lines.append(f"- Inverse property: {inv_str}")

                if include_examples:
                    lines.append(f"- SPARQL pattern: ?subject {prefix_local} ?object")

            else:  # class
                lines.append(f"### Class: {prefix_local}")
                lines.append(f"- URI: <{entry.uri}>")
                lines.append(f"- Description: {entry.short_text}")

                if include_examples:
                    lines.append(f"- SPARQL pattern: ?x a {prefix_local}")

            lines.append("")

        return "\n".join(lines)

    def get_entry_by_uri(self, uri: str) -> OntologyEntry | None:
        """Get an ontology entry by its URI."""
        for entry in self.entries:
            if entry.uri == uri:
                return entry
        return None

    @property
    def num_entries(self) -> int:
        """Total number of entries in the catalog."""
        return len(self.entries)

    @property
    def num_properties(self) -> int:
        """Number of properties in the catalog."""
        return sum(1 for e in self.entries if e.type == "property")

    @property
    def num_classes(self) -> int:
        """Number of classes in the catalog."""
        return sum(1 for e in self.entries if e.type == "class")

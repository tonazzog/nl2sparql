"""MCP Server implementation for NL2SPARQL.

This module provides the main MCP server class that registers tools and resources
for NL-to-SPARQL translation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource

from .tools import (
    handle_translate,
    handle_infer_patterns,
    handle_retrieve_examples,
    handle_search_ontology,
    handle_get_constraints,
    handle_validate_sparql,
    handle_execute_sparql,
    handle_fix_case_sensitivity,
    handle_check_variable_reuse,
    handle_fix_service_clause,
    handle_record_translation,
)
from .resources import (
    get_ontology_catalog,
    get_base_constraints,
    get_server_config,
)

if TYPE_CHECKING:
    from ..retrieval import HybridRetriever, OntologyRetriever
    from ..generation.synthesizer import NL2SPARQL
    from ..memory import TranslationMemory


# Default SPARQL endpoint
DEFAULT_ENDPOINT = "https://liita.it/sparql"


_WORKFLOW_INSTRUCTIONS = """\
Tools for translating natural language questions (Italian or English) into SPARQL
for the LiITA linguistic knowledge base.

Building a query yourself: call retrieve_examples for the question, use
search_ontology / get_constraints as needed, then check the query with
validate_sparql / execute_sparql and show the user the results."""

_TRANSLATE_INSTRUCTIONS = """
Alternatively, translate produces a validated query in one call."""

_MEMORY_INSTRUCTIONS = """

Translation memory - this server remembers past translations across sessions:
- retrieve_examples also returns remembered_examples (queries that answered the
  same or a near-identical question before - prefer adapting them) and
  known_failures (past mistakes on similar questions - do not repeat them).
- Call record_translation only after showing the user results, never
  mid-attempt, and record at most one success per question:
  * Success: ask the user in one line whether to save the query to memory, and
    record it only if they agree (or if they asked you to save automatically).
    The server can only check that a query runs and returns rows - the user is
    the judge of whether the rows actually answer the question.
  * Skip recording if you reused a remembered_examples query unchanged; it is
    already stored. If you changed it in any way (e.g. a different filter value
    or suffix), record it - it is a distinct mapping.
  * Failure: record without asking when the user points out a mistake, or when
    an executed query turned out conceptually wrong (wrong property direction,
    graph, or entity). Describe what was wrong in 'error', then record the
    corrected query as a success once the user confirms it. Do not record
    syntax slips you fixed immediately."""


def _build_instructions(config: "MCPConfig") -> str:
    """Server instructions sent to the MCP client at initialization: how the
    tools fit together, including when to write to translation memory."""
    text = _WORKFLOW_INSTRUCTIONS
    if config.translate_enabled:
        text += _TRANSLATE_INSTRUCTIONS
    if config.memory_enabled:
        text += _MEMORY_INSTRUCTIONS
    return text


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""

    provider: str = "openai"
    model: str | None = None
    endpoint: str = DEFAULT_ENDPOINT
    timeout: int = 30
    api_key: str | None = None
    memory_path: str | None = None
    memory_enabled: bool = True
    # False hides the translate tool - the only tool that calls an LLM provider -
    # for clients that write SPARQL themselves (e.g. Claude Desktop)
    translate_enabled: bool = True

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Create config from environment variables."""
        return cls(
            provider=os.environ.get("NL2SPARQL_PROVIDER", "openai"),
            model=os.environ.get("NL2SPARQL_MODEL"),
            endpoint=os.environ.get("NL2SPARQL_ENDPOINT", DEFAULT_ENDPOINT),
            timeout=int(os.environ.get("NL2SPARQL_TIMEOUT", "30")),
            memory_path=os.environ.get("NL2SPARQL_MEMORY_PATH"),
            memory_enabled=os.environ.get("NL2SPARQL_MEMORY_ENABLED", "1") != "0",
            translate_enabled=os.environ.get("NL2SPARQL_TRANSLATE_ENABLED", "1") != "0",
        )


class NL2SPARQLMCPServer:
    """MCP Server exposing NL2SPARQL capabilities as tools.

    The server provides tools for:
    - Full NL-to-SPARQL translation (using configured LLM)
    - Pattern inference
    - Example retrieval
    - Ontology search
    - SPARQL validation and execution
    - Query auto-fixing utilities

    Example:
        >>> server = NL2SPARQLMCPServer(MCPConfig(provider="anthropic"))
        >>> asyncio.run(server.run())
    """

    def __init__(self, config: MCPConfig | None = None):
        """Initialize the MCP server.

        Args:
            config: Server configuration. If None, uses defaults from environment.
        """
        self.config = config or MCPConfig.from_env()
        self.server = Server("nl2sparql", instructions=_build_instructions(self.config))
        self._setup_handlers()

        # Thread pool for running blocking operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Lazy-loaded components
        self._translator: NL2SPARQL | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._ontology_retriever: OntologyRetriever | None = None
        self._memory: TranslationMemory | None = None

    @property
    def memory(self) -> "TranslationMemory | None":
        """Lazy-load the persistent translation memory (None if disabled)."""
        if not self.config.memory_enabled:
            return None
        if self._memory is None:
            from ..memory import TranslationMemory

            self._memory = TranslationMemory(memory_path=self.config.memory_path)
        return self._memory

    @property
    def translator(self) -> "NL2SPARQL":
        """Lazy-load the NL2SPARQL translator."""
        if self._translator is None:
            from ..generation.synthesizer import NL2SPARQL

            try:
                self._translator = NL2SPARQL(
                    provider=self.config.provider,
                    model=self.config.model,
                    api_key=self.config.api_key,
                    validate=True,
                    fix_errors=True,
                    max_retries=3,
                    endpoint=self.config.endpoint,
                    memory=self.memory,
                )
            except ValueError as e:
                # Re-raise with more context about setting env vars in Claude Desktop
                raise ValueError(
                    f"{e}\n\nFor Claude Desktop, add the API key to your config:\n"
                    f'{{"env": {{"{self.config.provider.upper()}_API_KEY": "your-key"}}}}'
                ) from e
        return self._translator

    @property
    def hybrid_retriever(self) -> "HybridRetriever":
        """Lazy-load the hybrid retriever."""
        if self._hybrid_retriever is None:
            from ..retrieval import HybridRetriever

            self._hybrid_retriever = HybridRetriever()
        return self._hybrid_retriever

    @property
    def ontology_retriever(self) -> "OntologyRetriever":
        """Lazy-load the ontology retriever."""
        if self._ontology_retriever is None:
            from ..retrieval import OntologyRetriever

            self._ontology_retriever = OntologyRetriever()
        return self._ontology_retriever

    def _setup_handlers(self):
        """Register tool and resource handlers with the MCP server."""

        # ========== TOOL LISTING ==========

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            tools = [
                Tool(
                    name="translate",
                    description=(
                        "Translate a natural language question to a SPARQL query for the LiITA "
                        "(Linking Italian) linguistic knowledge base. Returns the generated SPARQL "
                        "query along with validation results and confidence score."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question in Italian or English",
                            },
                            "language": {
                                "type": "string",
                                "enum": ["it", "en"],
                                "default": "it",
                                "description": "Question language (auto-detected if not specified)",
                            },
                        },
                        "required": ["question"],
                    },
                ),
                Tool(
                    name="infer_patterns",
                    description=(
                        "Analyze a natural language question to detect query patterns. "
                        "Returns patterns like EMOTION_LEXICON, TRANSLATION, SEMANTIC_RELATION, etc. "
                        "with confidence scores. Useful for understanding query complexity."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question to analyze",
                            },
                            "threshold": {
                                "type": "number",
                                "default": 0.3,
                                "description": "Minimum score threshold for patterns",
                            },
                        },
                        "required": ["question"],
                    },
                ),
                Tool(
                    name="retrieve_examples",
                    description=(
                        "Retrieve similar SPARQL query examples for few-shot learning. "
                        "Uses hybrid retrieval (semantic + BM25 + pattern matching) to find "
                        "the most relevant examples from a curated dataset. When translation "
                        "memory is enabled, also returns 'remembered_examples' (queries that "
                        "previously answered the same or a near-identical question - prefer "
                        "adapting these) and 'known_failures' (past mistakes on similar "
                        "questions - avoid repeating them)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question",
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 5,
                                "description": "Number of examples to retrieve",
                            },
                            "patterns": {
                                "type": "object",
                                "description": "Pre-computed patterns (optional, from infer_patterns)",
                            },
                        },
                        "required": ["question"],
                    },
                ),
                Tool(
                    name="search_ontology",
                    description=(
                        "Search the ontology catalog for relevant RDF classes and properties. "
                        "Uses semantic search to find terms matching the query. Returns URIs, "
                        "descriptions, domain/range info, and SPARQL usage patterns."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (e.g., 'broader meaning', 'part of speech')",
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 10,
                                "description": "Number of results to return",
                            },
                            "entry_type": {
                                "type": "string",
                                "enum": ["all", "class", "property"],
                                "default": "all",
                                "description": "Filter by entry type",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_constraints",
                    description=(
                        "Get domain-specific constraint documentation for detected patterns. "
                        "Returns rules for EMOTION, TRANSLATION, SEMANTIC queries, etc. "
                        "along with required SPARQL prefixes and system prompt."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Pattern names (e.g., ['EMOTION_LEXICON', 'TRANSLATION'])",
                            },
                        },
                        "required": ["patterns"],
                    },
                ),
                Tool(
                    name="validate_sparql",
                    description=(
                        "Validate a SPARQL query with syntax, semantic, and endpoint checks. "
                        "Returns detailed validation results including errors and sample results."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sparql": {
                                "type": "string",
                                "description": "SPARQL query to validate",
                            },
                            "patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Detected patterns for semantic validation",
                            },
                            "check_syntax": {
                                "type": "boolean",
                                "default": True,
                                "description": "Perform syntax validation",
                            },
                            "check_semantic": {
                                "type": "boolean",
                                "default": True,
                                "description": "Perform semantic validation",
                            },
                            "check_endpoint": {
                                "type": "boolean",
                                "default": True,
                                "description": "Execute against endpoint",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Query timeout in seconds",
                            },
                        },
                        "required": ["sparql"],
                    },
                ),
                Tool(
                    name="execute_sparql",
                    description=(
                        "Execute a SPARQL query against the LiITA endpoint. "
                        "Returns the full result set (up to limit) with variable bindings."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sparql": {
                                "type": "string",
                                "description": "SPARQL query to execute",
                            },
                            "timeout": {
                                "type": "integer",
                                "default": 30,
                                "description": "Query timeout in seconds",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Maximum results to return",
                            },
                        },
                        "required": ["sparql"],
                    },
                ),
                Tool(
                    name="fix_case_sensitivity",
                    description=(
                        "Auto-fix case-sensitive string filters in SPARQL. "
                        "Converts FILTER(STR(?x) = 'value') to case-insensitive REGEX. "
                        "Useful when queries return 0 results due to case mismatch."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sparql": {
                                "type": "string",
                                "description": "SPARQL query to fix",
                            },
                        },
                        "required": ["sparql"],
                    },
                ),
                Tool(
                    name="check_variable_reuse",
                    description=(
                        "Check for variable reuse bugs in SPARQL. "
                        "Detects when a variable is incorrectly used both as a URI "
                        "(subject of a triple) and as a literal (in writtenRep)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sparql": {
                                "type": "string",
                                "description": "SPARQL query to check",
                            },
                        },
                        "required": ["sparql"],
                    },
                ),
                Tool(
                    name="fix_service_clause",
                    description=(
                        "Remove SERVICE clause wrapper from SPARQL query. "
                        "Use this when you get permission errors like 'SPARUL LOAD SERVICE DATA access denied' "
                        "or other federated query errors. The fix removes the SERVICE wrapper and keeps "
                        "the triple patterns inside, executing them against the local LiITA endpoint."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sparql": {
                                "type": "string",
                                "description": "SPARQL query with SERVICE clause to fix",
                            },
                        },
                        "required": ["sparql"],
                    },
                ),
                Tool(
                    name="record_translation",
                    description=(
                        "Save the outcome of a translation you built yourself to the persistent "
                        "translation memory, so future questions benefit from it (see "
                        "retrieve_examples). Call it only after showing the user results - "
                        "never mid-attempt - and record at most one success per question. "
                        "outcome='success': only after the user confirms the results answer "
                        "the question (ask in one line, e.g. 'Save this query to memory?'), "
                        "unless they asked you to save automatically. Skip it if you reused a "
                        "remembered_examples query unchanged - it is already stored; do record "
                        "it if you changed the query in any way, even just a filter value. "
                        "outcome='failure' (no need to ask): when the user points out a mistake, "
                        "or an executed query turned out conceptually wrong (wrong property "
                        "direction, graph, entity); put what was wrong in 'error', then record "
                        "the corrected query as a success once confirmed. Do not record syntax "
                        "slips you fixed immediately. A claimed success is re-executed against "
                        "the endpoint and only stored as a success if it returns at least one row."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The user's original natural language question",
                            },
                            "sparql": {
                                "type": "string",
                                "description": "The SPARQL query being recorded",
                            },
                            "outcome": {
                                "type": "string",
                                "enum": ["success", "failure"],
                                "description": "Whether the query correctly answers the question",
                            },
                            "error": {
                                "type": "string",
                                "description": "For failures: what was wrong with the query",
                            },
                        },
                        "required": ["question", "sparql", "outcome"],
                    },
                ),
            ]
            if not self.config.translate_enabled:
                tools = [t for t in tools if t.name != "translate"]
            return tools

        # ========== TOOL CALLING ==========

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            try:
                result = await self._handle_tool(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                error_result = {"error": str(e), "tool": name}
                return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

        # ========== RESOURCE LISTING ==========

        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="liita://ontology/catalog",
                    name="Ontology Catalog",
                    description="Full ontology catalog with classes and properties",
                    mimeType="application/json",
                ),
                Resource(
                    uri="liita://constraints/base",
                    name="Base Constraints",
                    description="Base system prompt and SPARQL prefixes",
                    mimeType="text/plain",
                ),
                Resource(
                    uri="liita://config",
                    name="Server Configuration",
                    description="Current server configuration",
                    mimeType="application/json",
                ),
            ]

        # ========== RESOURCE READING ==========

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            if uri == "liita://ontology/catalog":
                return json.dumps(get_ontology_catalog(), indent=2)
            elif uri == "liita://constraints/base":
                return get_base_constraints()
            elif uri == "liita://config":
                return json.dumps(get_server_config(self.config), indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _handle_tool(self, name: str, arguments: dict) -> dict[str, Any]:
        """Route tool calls to appropriate handlers.

        Heavy operations (translate, retrieve, validate) are run in a thread pool
        to avoid blocking the async event loop.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as dictionary
        """
        loop = asyncio.get_event_loop()

        if name == "translate":
            if not self.config.translate_enabled:
                raise ValueError(
                    "The translate tool is disabled on this server (--no-translate). "
                    "Write the SPARQL with retrieve_examples / execute_sparql instead."
                )
            # Heavy: LLM call + validation - run in thread
            return await loop.run_in_executor(
                self._executor,
                partial(handle_translate, arguments, self.translator)
            )

        elif name == "infer_patterns":
            # Light operation
            return handle_infer_patterns(arguments)

        elif name == "retrieve_examples":
            # Heavy: embedding computation - run in thread
            return await loop.run_in_executor(
                self._executor,
                partial(handle_retrieve_examples, arguments, self.hybrid_retriever, self.memory)
            )

        elif name == "search_ontology":
            # Heavy: embedding computation - run in thread
            return await loop.run_in_executor(
                self._executor,
                partial(handle_search_ontology, arguments, self.ontology_retriever)
            )

        elif name == "get_constraints":
            # Light operation
            return handle_get_constraints(arguments)

        elif name == "validate_sparql":
            # Heavy: network calls - run in thread
            return await loop.run_in_executor(
                self._executor,
                partial(
                    handle_validate_sparql,
                    arguments,
                    endpoint=self.config.endpoint,
                    timeout=self.config.timeout,
                )
            )

        elif name == "execute_sparql":
            # Heavy: network calls - run in thread
            return await loop.run_in_executor(
                self._executor,
                partial(
                    handle_execute_sparql,
                    arguments,
                    endpoint=self.config.endpoint,
                    default_timeout=self.config.timeout,
                )
            )

        elif name == "fix_case_sensitivity":
            # Light operation
            return handle_fix_case_sensitivity(arguments)

        elif name == "check_variable_reuse":
            # Light operation
            return handle_check_variable_reuse(arguments)

        elif name == "fix_service_clause":
            # Light operation
            return handle_fix_service_clause(arguments)

        elif name == "record_translation":
            # Heavy: verification query + embedding on next lookup - run in thread
            return await loop.run_in_executor(
                self._executor,
                partial(
                    handle_record_translation,
                    arguments,
                    self.memory,
                    endpoint=self.config.endpoint,
                    timeout=self.config.timeout,
                )
            )

        else:
            raise ValueError(f"Unknown tool: {name}")

    def preload_models(self) -> None:
        """Load the embedding models before the stdio loop starts.

        On Windows, loading the sentence-transformer models for the first time
        while the stdio transport is already reading from stdin can block
        indefinitely (observed 2026-11-17: infer_patterns and retrieve_examples
        never answered, validate_sparql did). Loading them up front avoids it
        and also makes the first tool calls fast.
        """
        from ..retrieval.patterns import infer_patterns_hybrid

        infer_patterns_hybrid("warm up")
        _ = self.hybrid_retriever
        _ = self.ontology_retriever

    async def run(self):
        """Run the MCP server with stdio transport."""
        self.preload_models()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def run_server(config: MCPConfig | None = None):
    """Entry point for running the MCP server.

    Args:
        config: Server configuration. If None, uses defaults from environment.

    Example:
        >>> import asyncio
        >>> from nl2sparql.mcp import run_server, MCPConfig
        >>> config = MCPConfig(provider="anthropic")
        >>> asyncio.run(run_server(config))
    """
    server = NL2SPARQLMCPServer(config)
    await server.run()

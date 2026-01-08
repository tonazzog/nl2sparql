"""MCP (Model Context Protocol) server for NL2SPARQL.

This module provides an MCP server that exposes NL2SPARQL capabilities as tools
that can be called by MCP-compatible LLM clients (e.g., Claude Desktop).

Example usage with Claude Desktop:

    # claude_desktop_config.json
    {
        "mcpServers": {
            "nl2sparql": {
                "command": "nl2sparql",
                "args": ["mcp", "serve", "--provider", "anthropic"]
            }
        }
    }

The server exposes tools for:
- Full NL-to-SPARQL translation (using configured LLM)
- Pattern inference and example retrieval
- Ontology search
- SPARQL validation and execution
- Query auto-fixing utilities
"""

from .server import NL2SPARQLMCPServer, MCPConfig, run_server

__all__ = [
    "NL2SPARQLMCPServer",
    "MCPConfig",
    "run_server",
]

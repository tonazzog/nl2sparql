"""Entry point for running the MCP server as a module.

Usage:
    python -m nl2sparql.mcp serve --provider ollama --model llama3
    python -m nl2sparql.mcp config

This allows Claude Desktop to use the full Python path:
    {
        "mcpServers": {
            "nl2sparql": {
                "command": "C:/path/to/python.exe",
                "args": ["-m", "nl2sparql.mcp", "serve", "--provider", "anthropic"]
            }
        }
    }
"""

import argparse
import asyncio
import sys

from .server import NL2SPARQLMCPServer, MCPConfig


def main():
    """Main entry point for the MCP server CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m nl2sparql.mcp",
        description="NL2SPARQL MCP Server"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the MCP server with stdio transport"
    )
    serve_parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="LLM provider for translation (default: openai)"
    )
    serve_parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (uses provider default if not specified)"
    )
    serve_parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key for the LLM provider (can also use env vars)"
    )
    serve_parser.add_argument(
        "--endpoint", "-e",
        default="https://liita.it/sparql",
        help="SPARQL endpoint URL"
    )
    serve_parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Query timeout in seconds (default: 30)"
    )

    # config command
    config_parser = subparsers.add_parser(
        "config",
        help="Show server configuration"
    )
    config_parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="LLM provider"
    )

    args = parser.parse_args()

    if args.command == "serve":
        config = MCPConfig(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            endpoint=args.endpoint,
            timeout=args.timeout,
        )
        server = NL2SPARQLMCPServer(config)
        asyncio.run(server.run())

    elif args.command == "config":
        from .resources import get_server_config
        import json

        config = MCPConfig(provider=args.provider)
        print(json.dumps(get_server_config(config), indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Agent-based Gradio Web UI for NL2SPARQL.

This app uses two LLMs:
- Orchestrator: Decides which tools to call (can be a smaller/cheaper model)
- Translator: Expert SPARQL generator (used by the translate tool)

The orchestrator can call any of the MCP tools and sees their results,
allowing for dynamic, self-correcting query generation.

Usage:
    python scripts/gradio_app_agent.py
    python scripts/gradio_app_agent.py --orchestrator-provider anthropic --orchestrator-model claude-3-haiku-20240307
    python scripts/gradio_app_agent.py --translator-provider mistral --translator-model mistral-large-latest
"""

import argparse
import json
from typing import Any

import gradio as gr

from nl2sparql import LIITA_ENDPOINT
from nl2sparql.llm.base import get_client, LLMClient
from nl2sparql.mcp.tools import (
    handle_infer_patterns,
    handle_retrieve_examples,
    handle_get_constraints,
    handle_validate_sparql,
    handle_execute_sparql,
    handle_fix_case_sensitivity,
    handle_check_variable_reuse,
    handle_fix_service_clause,
    handle_fix_graph_clause,
)
from nl2sparql.retrieval import HybridRetriever


# =============================================================================
# Tool Definitions (OpenAI function calling format)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "infer_patterns",
            "description": "Analyze a natural language question to detect query patterns like EMOTION_LEXICON, TRANSLATION, SEMANTIC_RELATION, etc. Use this FIRST to understand what kind of query is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The natural language question to analyze"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_examples",
            "description": "Retrieve similar SPARQL query examples from the dataset. Use this to find examples that can guide query generation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The natural language question"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of examples to retrieve (default: 3)",
                        "default": 3
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_constraints",
            "description": "Get domain-specific constraints and rules for detected patterns. Provides SPARQL prefixes and query templates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of pattern names (e.g., ['EMOTION_LEXICON', 'TRANSLATION'])"
                    }
                },
                "required": ["patterns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sparql",
            "description": "Generate a SPARQL query using the expert translator LLM. Call this after gathering context from other tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The original natural language question"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context gathered from other tools (patterns, example, constraints)"
                    }
                },
                "required": ["question", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_sparql",
            "description": "Validate a SPARQL query for syntax, semantic correctness, and execute against the endpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query to validate"
                    }
                },
                "required": ["sparql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sparql",
            "description": "Execute a SPARQL query and return results. Use after validation succeeds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query to execute"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["sparql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fix_query",
            "description": "Apply automatic fixes to a SPARQL query (SERVICE clause removal, GRAPH clause remova, case sensitivity, variable reuse check).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query to fix"
                    }
                },
                "required": ["sparql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return the final SPARQL query and explanation to the user. Call this when you have a valid, working query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The final SPARQL query"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of what the query does"
                    }
                },
                "required": ["sparql", "explanation"]
            }
        }
    }
]


# =============================================================================
# Global State
# =============================================================================

orchestrator_client: LLMClient | None = None
translator_client: LLMClient | None = None
hybrid_retriever: HybridRetriever | None = None


def init_clients(
    orchestrator_provider: str,
    orchestrator_model: str | None,
    orchestrator_api_key: str | None,
    translator_provider: str,
    translator_model: str | None,
    translator_api_key: str | None,
):
    """Initialize both LLM clients."""
    global orchestrator_client, translator_client, hybrid_retriever

    orchestrator_client = get_client(orchestrator_provider, orchestrator_model, api_key=orchestrator_api_key)
    translator_client = get_client(translator_provider, translator_model, api_key=translator_api_key)
    hybrid_retriever = HybridRetriever()


# =============================================================================
# Tool Execution
# =============================================================================

def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool and return its result."""

    if name == "infer_patterns":
        return handle_infer_patterns(arguments)

    elif name == "retrieve_examples":
        result = handle_retrieve_examples(arguments, hybrid_retriever)
        # Simplify for context
        examples = []
        for ex in result.get("examples", [])[:3]:
            examples.append({
                "nl": ex["nl"],
                "sparql": ex["sparql"],
                "score": ex["score"],
                "patterns": list(ex.get("patterns", {}).keys())
            })
        return {"examples": examples}

    elif name == "get_constraints":
        return handle_get_constraints(arguments)

    elif name == "generate_sparql":
        # Use the translator LLM to generate SPARQL
        return _generate_sparql(arguments["question"], arguments["context"])

    elif name == "validate_sparql":
        arguments['validate_syntax'] = False
        return handle_validate_sparql(
            arguments,
            endpoint=LIITA_ENDPOINT,
            timeout=30
        )

    elif name == "execute_sparql":
        return handle_execute_sparql(
            arguments,
            endpoint=LIITA_ENDPOINT,
            default_timeout=30
        )

    elif name == "fix_query":
        sparql = arguments["sparql"]
        changes = []

        # Apply SERVICE fix
        service_result = handle_fix_service_clause({"sparql": sparql})
        if service_result["was_modified"]:
            sparql = service_result["sparql"]
            changes.append(f"Removed SERVICE clause")

        # Apply GRAPH fix
        graph_result = handle_fix_graph_clause({"sparql": sparql})
        if graph_result["was_modified"]:
            sparql = graph_result["sparql"]
            changes.append(f"Removed GRAPH clause")

        # Apply case sensitivity fix
        case_result = handle_fix_case_sensitivity({"sparql": sparql})
        if case_result["was_modified"]:
            sparql = case_result["sparql"]
            changes.extend(case_result["changes"])

        # Check variable reuse
        var_result = handle_check_variable_reuse({"sparql": sparql})

        return {
            "sparql": sparql,
            "changes": changes,
            "warnings": var_result.get("issues", [])
        }

    elif name == "final_answer":
        return arguments  # Just pass through

    else:
        return {"error": f"Unknown tool: {name}"}


def _generate_sparql(question: str, context: str) -> dict[str, Any]:
    """Use the translator LLM to generate SPARQL."""

    system_prompt = """You are an expert SPARQL query generator for the LiITA (Linking Italian) linguistic knowledge base.

Generate a valid SPARQL query based on the user's question and the provided context.

CRITICAL: return ONLY the SPARQL query, do not add comments

MANDATORY PREFIXES (use these EXACT URIs - do NOT modify them):
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX elita: <http://w3id.org/elita/>
PREFIX lexinfo: <http://www.lexinfo.net/ontology/3.0/lexinfo#>
PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
PREFIX lime: <http://www.w3.org/ns/lemon/lime#>
PREFIX marl: <http://www.gsi.upm.es/ontologies/marl/ns#>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX vartrans: <http://www.w3.org/ns/lemon/vartrans#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

CRITICAL: Copy the prefixes EXACTLY as shown above. Common mistakes to AVOID:
- elita: must be <http://w3id.org/elita/> NOT <http://w3id.org/elita/ontology#>
- marl: must be <http://www.gsi.upm.es/ontologies/marl/ns#> NOT <http://www.w3.org/ns/marl#>

CRITICAL RULES:
1. Translation direction is ALWAYS Italian → Dialect (never dialect → Italian)
2. For multi-dialect queries, use DIFFERENT Italian lexical entry variables for each dialect
3. Use dcterms:isPartOf with LemmaBank URI to identify dialect lemmas (NO GRAPH clauses)
4. Never reuse a variable for both a URI and a literal value
5. Variables bound in SERVICE can be used outside, but NOT vice versa

SERVICE BLOCK RULES (VERY IMPORTANT):
6. Only use SERVICE block when querying CompL-it for definitions or semantic relations
7. The ONLY valid SERVICE endpoint is: SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/>
8. NEVER use localhost, made-up URLs, or any other SERVICE endpoints
9. For EMOTION queries (ELITA), do NOT use SERVICE - emotions are in GRAPH <http://w3id.org/elita>
10. For TRANSLATION queries (dialects), do NOT use SERVICE - translations are in the main LiITA data

LINKING LIITA TO COMPL-IT (CRITICAL):
11. When starting from a LiITA lemma and needing CompL-it data (definitions, semantic relations):
    - Use the SAME variable name (?writtenRep) in both GRAPH and SERVICE blocks
    - The shared variable creates a NATURAL JOIN - no FILTER needed
    - NEVER use FILTER(STR(?x) = STR(?y)) to compare variables across SERVICE boundaries


---

Always maintain SPARQL correctness and follow ALL mandatory patterns for the query categories involved.

"""

    user_prompt = f"""Question: {question}

Context:
{context}

Generate the SPARQL query:"""

    try:
        response = translator_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        # Extract SPARQL from response
        sparql = response.strip()
        if "```sparql" in sparql:
            sparql = sparql.split("```sparql")[1].split("```")[0].strip()
        elif "```" in sparql:
            sparql = sparql.split("```")[1].split("```")[0].strip()

        return {"sparql": sparql}

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Orchestrator Loop
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestrator for a natural language to SPARQL translation system.

Your job is to:
1. Analyze the user's question using available tools
2. Gather context (patterns, examples, domain rules)
3. Generate a SPARQL query using the translator
4. Validate the query and fix any issues
5. Return the final working query

Available tools:
- infer_patterns: Detect query patterns (call FIRST)
- retrieve_examples: Find similar examples
- get_constraints: Get domain rules
- generate_sparql: Generate SPARQL using expert LLM
- validate_sparql: Check query validity
- execute_sparql: Run query and get results
- fix_query: Auto-fix common issues
- final_answer: Return final result (call LAST)

Strategy:
1. Start with infer_patterns to understand the query type
2. Use retrieve_examples to find similar queries
3. Use get_constraints to enforce specific domain rules
4. Call generate_sparql with gathered context
5. Validate and fix if necessary
6. Call final_answer with the working query

Be efficient - don't call unnecessary tools. Focus on getting a working query."""


def run_orchestrator_streaming(question: str, max_iterations: int = 10):
    """
    Run the orchestrator loop with streaming updates.

    Yields:
        (current_sparql, log_markdown, explanation) after each tool call
    """
    messages = [
        {"role": "user", "content": question}
    ]

    tool_calls_log = []
    current_sparql = ""
    explanation = ""

    for iteration in range(max_iterations):
        # Call orchestrator LLM with tools
        try:
            response = orchestrator_client.generate_with_tools(
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                messages=messages,
                tools=TOOLS,
                temperature=0.1,
            )
        except Exception as e:
            tool_calls_log.append({
                "tool": "error",
                "error": str(e)
            })
            yield current_sparql, format_tool_log(tool_calls_log), explanation
            return

        # Check if we have tool calls
        if not response.get("tool_calls"):
            # No more tool calls - we're done
            if response.get("content"):
                explanation = response["content"]
            yield current_sparql, format_tool_log(tool_calls_log), explanation
            return

        # Process each tool call
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]

            # Log the call
            log_entry = {
                "iteration": iteration + 1,
                "tool": tool_name,
                "arguments": tool_args,
                "status": "running",
            }
            tool_calls_log.append(log_entry)

            # Yield immediately to show "running" state
            yield current_sparql, format_tool_log(tool_calls_log), explanation

            # Execute the tool
            try:
                result = execute_tool(tool_name, tool_args)
                log_entry["result"] = result
                log_entry["status"] = "done"

                # Check for final_answer
                if tool_name == "final_answer":
                    current_sparql = result.get("sparql", "")
                    explanation = result.get("explanation", "")
                    yield current_sparql, format_tool_log(tool_calls_log), explanation
                    return

                # Update sparql if generate_sparql was called
                if tool_name == "generate_sparql" and "sparql" in result:
                    current_sparql = result["sparql"]

            except Exception as e:
                result = {"error": str(e)}
                log_entry["error"] = str(e)
                log_entry["status"] = "error"

            # Yield after tool execution completes
            yield current_sparql, format_tool_log(tool_calls_log), explanation

            # Add tool result to messages for next iteration
            messages.append({
                "role": "assistant",
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "name": tool_name,
                "content": json.dumps(result, indent=2)
            })

    yield current_sparql, format_tool_log(tool_calls_log), explanation


def format_tool_log(tool_calls_log: list[dict]) -> str:
    """Format tool calls log as markdown."""
    log_output = "## Tool Calls\n\n"

    for call in tool_calls_log:
        tool = call.get("tool", "unknown")
        iteration = call.get("iteration", "?")
        status = call.get("status", "done")

        if tool == "error":
            log_output += f"### ❌ Error\n\n{call.get('error', 'Unknown error')}\n\n"
            continue

        # Status indicator
        if status == "running":
            icon = "⏳"
        elif status == "error":
            icon = "❌"
        else:
            icon = "✅"

        log_output += f"### {icon} {iteration}. {tool}\n\n"

        # Show arguments (simplified)
        args = call.get("arguments", {})
        if args:
            args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
            log_output += f"**Args:** {args_str}\n\n"

        # Show result (only if done)
        if status == "running":
            log_output += "*Running...*\n\n"
        elif "error" in call:
            log_output += f"**Error:** {call['error']}\n\n"
        else:
            result = call.get("result", {})
            if result:
                if tool == "infer_patterns":
                    patterns = result.get("top_patterns", [])
                    log_output += f"**Patterns:** {', '.join(patterns)}\n\n"
                elif tool == "retrieve_examples":
                    examples = result.get("examples", [])
                    log_output += f"**Found:** {len(examples)} examples\n\n"
                elif tool == "generate_sparql":
                    if "sparql" in result:
                        log_output += f"**Generated query** ({len(result['sparql'])} chars)\n\n"
                    else:
                        log_output += f"**Error:** {result.get('error', 'Unknown')}\n\n"
                elif tool == "validate_sparql":
                    valid = result.get("is_valid", False)
                    v_icon = "✅" if valid else "❌"
                    log_output += f"**Valid:** {v_icon}\n\n"
                    if result.get("endpoint", {}).get("result_count"):
                        log_output += f"**Results:** {result['endpoint']['result_count']}\n\n"
                elif tool == "fix_query":
                    changes = result.get("changes", [])
                    if changes:
                        log_output += f"**Fixes:** {', '.join(changes)}\n\n"
                    else:
                        log_output += "**No fixes needed**\n\n"
                elif tool == "final_answer":
                    log_output += "**Completed**\n\n"
                else:
                    log_output += f"**Result:** {json.dumps(result)[:100]}...\n\n"

        log_output += "---\n\n"

    return log_output


# =============================================================================
# Gradio Interface
# =============================================================================

def process_question(question: str):
    """Process a question through the orchestrator with streaming updates."""

    if not question.strip():
        yield "", "Please enter a question", ""
        return

    if not orchestrator_client or not translator_client:
        yield "", "Error: Clients not initialized", ""
        return

    try:
        # Stream updates from orchestrator
        for sparql, log, explanation in run_orchestrator_streaming(question):
            explanation_output = ""
            if explanation:
                explanation_output = f"## Explanation\n\n{explanation}"
            yield sparql, log, explanation_output

    except Exception as e:
        import traceback
        yield "", f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```", ""


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""

    with gr.Blocks(title="NL2SPARQL Agent for LiITA") as app:
        gr.Markdown("""
        # NL2SPARQL Agent

        This app uses two LLMs:
        - **Orchestrator**: Decides which tools to call
        - **Translator**: Expert SPARQL generator

        Watch the tool calls in real-time as the agent works through your question.
        """)

        with gr.Row():
            question_input = gr.Textbox(
                label="Natural Language Question",
                placeholder="e.g., Quali lemmi esprimono tristezza?",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("Generate SPARQL", variant="primary", scale=1)

        with gr.Row():
            # Left: SPARQL output
            with gr.Column(scale=1):
                sparql_output = gr.Code(
                    label="Generated SPARQL",
                    language="sql",
                    lines=15,
                )
                explanation_output = gr.Markdown(label="Explanation")

            # Right: Tool calls log
            with gr.Column(scale=1):
                log_output = gr.Markdown(label="Tool Calls")

        submit_btn.click(
            process_question,
            inputs=[question_input],
            outputs=[sparql_output, log_output, explanation_output],
        )

        gr.Examples(
            examples=[
                ["Quali lemmi esprimono tristezza?"],
                ["Find Sicilian translations of 'casa'"],
                ["Quali sono gli iperonimi di 'cane'?"],
                ["Find definitions of words starting with 'anti'"],
                ["Quanti sensi ha la parola 'banco'?"],
            ],
            inputs=[question_input],
        )

        gr.Markdown("""
        ---
        **NL2SPARQL Agent** | Dual-LLM Architecture |
        [Documentation](https://github.com/tonazzog/nl2sparql)
        """)

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Agent-based Gradio UI for NL2SPARQL")

    # Orchestrator config
    parser.add_argument(
        "--orchestrator-provider", "-op",
        default="mistral",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="Orchestrator LLM provider (default: mistral)"
    )
    parser.add_argument(
        "--orchestrator-model", "-om",
        default=None,
        help="Orchestrator model (uses provider default if not specified)"
    )
    parser.add_argument(
        "--orchestrator-api-key", "-ok",
        default=None,
        help="Orchestrator API key (uses environment variable if not specified)"
    )

    # Translator config
    parser.add_argument(
        "--translator-provider", "-tp",
        default="mistral",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="Translator LLM provider (default: mistral)"
    )
    parser.add_argument(
        "--translator-model", "-tm",
        default=None,
        help="Translator model (uses provider default if not specified)"
    )
    parser.add_argument(
        "--translator-api-key", "-tk",
        default=None,
        help="Translator API key (uses environment variable if not specified)"
    )

    # Server config
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run on (default: 7861)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NL2SPARQL Agent - Dual LLM Architecture")
    print("=" * 60)
    print(f"Orchestrator: {args.orchestrator_provider} / {args.orchestrator_model or 'default'}")
    print(f"Translator:   {args.translator_provider} / {args.translator_model or 'default'}")
    print()

    print("Initializing LLM clients...")
    init_clients(
        args.orchestrator_provider,
        args.orchestrator_model,
        args.orchestrator_api_key,
        args.translator_provider,
        args.translator_model,
        args.translator_api_key,
    )

    print("Starting web UI...")
    app = create_ui()
    app.launch(
        share=args.share,
        server_port=args.port,
        server_name="127.0.0.1",
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()

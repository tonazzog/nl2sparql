#!/usr/bin/env python3
"""
Gradio Web UI for NL2SPARQL.

A simple web interface for translating natural language questions
to SPARQL queries for the LiITA linguistic knowledge base.

Usage:
    python scripts/gradio_app.py
    python scripts/gradio_app.py --provider mistral --model mistral-small-latest
    python scripts/gradio_app.py --provider ollama --model llama3 --share

Features:
- Translate natural language to SPARQL
- View detected patterns and confidence scores
- Execute queries and see results
- Search the ontology catalog
"""

import argparse
import json

import gradio as gr

from nl2sparql import NL2SPARQL, LIITA_ENDPOINT
from nl2sparql.mcp.tools import (
    handle_infer_patterns,
    handle_search_ontology,
    handle_execute_sparql,
    handle_fix_case_sensitivity,
)
from nl2sparql.retrieval import OntologyRetriever


# Global instances (initialized on startup)
translator: NL2SPARQL | None = None
ontology_retriever: OntologyRetriever | None = None


def init_translator(provider: str, model: str | None, api_key: str | None):
    """Initialize the translator."""
    global translator
    translator = NL2SPARQL(
        provider=provider,
        model=model,
        api_key=api_key,
        validate=True,
        fix_errors=True,
        max_retries=2,
    )


def init_ontology_retriever():
    """Initialize the ontology retriever."""
    global ontology_retriever
    ontology_retriever = OntologyRetriever()


def translate_question(question: str) -> tuple[str, str, str, str]:
    """Translate a natural language question to SPARQL.

    Returns: (sparql, patterns_info, validation_info, results_preview)
    """
    if not translator:
        return "Error: Translator not initialized", "", "", ""

    if not question.strip():
        return "Please enter a question", "", "", ""

    try:
        result = translator.translate(question)

        # Format patterns
        patterns_info = f"**Detected Patterns:** {', '.join(result.detected_patterns) or 'None'}\n\n"
        patterns_info += f"**Confidence:** {result.confidence:.2%}\n\n"
        if result.was_fixed:
            patterns_info += f"**Query was auto-fixed** ({result.fix_attempts} attempts)"

        # Format validation
        if result.validation:
            v = result.validation
            if v.is_valid:
                validation_info = f"✅ **Valid Query**\n\n"
                validation_info += f"**Results:** {v.result_count or 0} rows"
            else:
                validation_info = f"❌ **Invalid Query**\n\n"
                if v.syntax_error:
                    validation_info += f"**Syntax Error:** {v.syntax_error}\n\n"
                if v.execution_error:
                    validation_info += f"**Execution Error:** {v.execution_error}\n\n"
                if v.semantic_errors:
                    validation_info += f"**Semantic Issues:** {', '.join(v.semantic_errors)}"
        else:
            validation_info = "Validation not performed"

        # Format sample results
        results_preview = ""
        if result.validation and result.validation.sample_results:
            results_preview = "**Sample Results:**\n\n"
            for i, row in enumerate(result.validation.sample_results[:5]):
                results_preview += f"{i+1}. {row}\n"

        return result.sparql, patterns_info, validation_info, results_preview

    except Exception as e:
        return f"Error: {str(e)}", "", "", ""


def analyze_patterns(question: str) -> str:
    """Analyze patterns in a question without generating SPARQL."""
    if not question.strip():
        return "Please enter a question"

    try:
        result = handle_infer_patterns({"question": question, "threshold": 0.2})

        output = f"**Complexity:** {result['complexity']}\n\n"
        output += "**Top Patterns:**\n"
        for pattern in result['top_patterns']:
            score = result['patterns'].get(pattern, 0)
            output += f"- {pattern}: {score:.2%}\n"

        output += "\n**All Pattern Scores:**\n"
        for pattern, score in sorted(result['patterns'].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            output += f"- {pattern}: {score:.2%} {bar}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


def search_ontology(query: str, entry_type: str, top_k: int) -> str:
    """Search the ontology catalog."""
    if not ontology_retriever:
        return "Error: Ontology retriever not initialized"

    if not query.strip():
        return "Please enter a search query"

    try:
        result = handle_search_ontology(
            {"query": query, "top_k": top_k, "entry_type": entry_type.lower()},
            ontology_retriever
        )

        output = f"**Found {len(result['entries'])} entries:**\n\n"
        for entry in result['entries']:
            output += f"### {entry['prefix_local']}\n"
            output += f"- **Type:** {entry['type']}\n"
            output += f"- **URI:** `{entry['uri']}`\n"
            output += f"- **Description:** {entry['description']}\n"
            if entry.get('sparql_pattern'):
                output += f"- **SPARQL Pattern:** `{entry['sparql_pattern']}`\n"
            output += f"- **Score:** {entry['score']:.3f}\n\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


def execute_query(sparql: str, limit: int) -> str:
    """Execute a SPARQL query."""
    if not sparql.strip():
        return "Please enter a SPARQL query"

    try:
        result = handle_execute_sparql(
            {"sparql": sparql, "limit": limit},
            endpoint=LIITA_ENDPOINT,
            default_timeout=30
        )

        if result['success']:
            output = f"**Success!** {result['result_count']} results\n\n"
            output += f"**Variables:** {', '.join(result['variables'])}\n\n"
            output += "**Results:**\n\n"

            # Format as table
            if result['results']:
                # Header
                output += "| " + " | ".join(result['variables']) + " |\n"
                output += "| " + " | ".join(["---"] * len(result['variables'])) + " |\n"
                # Rows
                for row in result['results']:
                    values = [str(row.get(v, ""))[:50] for v in result['variables']]
                    output += "| " + " | ".join(values) + " |\n"

            return output
        else:
            return f"**Error:** {result['error']}"

    except Exception as e:
        return f"Error: {str(e)}"


def fix_case_sensitivity(sparql: str) -> tuple[str, str]:
    """Fix case-sensitive filters in a SPARQL query."""
    if not sparql.strip():
        return sparql, "Please enter a SPARQL query"

    try:
        result = handle_fix_case_sensitivity({"sparql": sparql})

        if result['was_modified']:
            info = "**Changes made:**\n"
            for change in result['changes']:
                info += f"- {change}\n"
        else:
            info = "No case-sensitive filters found to fix."

        return result['sparql'], info

    except Exception as e:
        return sparql, f"Error: {str(e)}"


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""

    with gr.Blocks(title="NL2SPARQL - LiITA Query Generator") as app:
        gr.Markdown("""
        # NL2SPARQL for LiITA

        Translate natural language questions into SPARQL queries for the
        [LiITA (Linking Italian)](https://liita.it) linguistic knowledge base.
        """)

        with gr.Tabs():
            # Tab 1: Translate
            with gr.TabItem("Translate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        question_input = gr.Textbox(
                            label="Natural Language Question",
                            placeholder="e.g., Quali lemmi esprimono tristezza?",
                            lines=2,
                        )
                        translate_btn = gr.Button("Translate to SPARQL", variant="primary")

                        with gr.Row():
                            patterns_output = gr.Markdown(label="Patterns")
                            validation_output = gr.Markdown(label="Validation")

                    with gr.Column(scale=1):
                        sparql_output = gr.Code(
                            label="Generated SPARQL",
                            language="sql",  # SPARQL not supported, SQL is closest
                            lines=15,
                        )
                        results_output = gr.Markdown(label="Sample Results")

                translate_btn.click(
                    translate_question,
                    inputs=[question_input],
                    outputs=[sparql_output, patterns_output, validation_output, results_output],
                )

                gr.Examples(
                    examples=[
                        ["Quali lemmi esprimono tristezza?"],
                        ["Trova le traduzioni siciliane di 'casa'"],
                        ["What are the hypernyms of 'dog'?"],
                        ["Trova i sinonimi di 'bello'"],
                        ["Lemmi che indicano parti del corpo"],
                    ],
                    inputs=[question_input],
                )

            # Tab 2: Pattern Analysis
            with gr.TabItem("Analyze Patterns"):
                pattern_input = gr.Textbox(
                    label="Question to Analyze",
                    placeholder="Enter a question to see detected patterns",
                    lines=2,
                )
                analyze_btn = gr.Button("Analyze Patterns")
                pattern_output = gr.Markdown()

                analyze_btn.click(
                    analyze_patterns,
                    inputs=[pattern_input],
                    outputs=[pattern_output],
                )

            # Tab 3: Ontology Search
            with gr.TabItem("Search Ontology"):
                with gr.Row():
                    onto_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., emotion, translation, hypernym",
                    )
                    onto_type = gr.Radio(
                        choices=["All", "Class", "Property"],
                        value="All",
                        label="Entry Type",
                    )
                    onto_k = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Number of Results",
                    )
                search_btn = gr.Button("Search Ontology")
                onto_output = gr.Markdown()

                search_btn.click(
                    search_ontology,
                    inputs=[onto_query, onto_type, onto_k],
                    outputs=[onto_output],
                )

            # Tab 4: Execute Query
            with gr.TabItem("Execute SPARQL"):
                exec_sparql = gr.Code(
                    label="SPARQL Query",
                    language="sql",
                    lines=10,
                )
                with gr.Row():
                    exec_limit = gr.Slider(
                        minimum=1, maximum=100, value=20, step=1,
                        label="Result Limit",
                    )
                    exec_btn = gr.Button("Execute Query", variant="primary")
                exec_output = gr.Markdown()

                exec_btn.click(
                    execute_query,
                    inputs=[exec_sparql, exec_limit],
                    outputs=[exec_output],
                )

            # Tab 5: Fix Query
            with gr.TabItem("Fix Query"):
                gr.Markdown("Fix case-sensitive string filters in SPARQL queries.")
                fix_input = gr.Code(
                    label="SPARQL Query to Fix",
                    language="sql",
                    lines=8,
                )
                fix_btn = gr.Button("Fix Case Sensitivity")
                fix_output = gr.Code(
                    label="Fixed SPARQL",
                    language="sql",
                    lines=8,
                )
                fix_info = gr.Markdown()

                fix_btn.click(
                    fix_case_sensitivity,
                    inputs=[fix_input],
                    outputs=[fix_output, fix_info],
                )

        gr.Markdown("""
        ---
        **NL2SPARQL** - Natural Language to SPARQL for LiITA |
        [Documentation](https://github.com/tonazzog/nl2sparql)
        """)

    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio Web UI for NL2SPARQL")
    parser.add_argument(
        "--provider", "-p",
        default="mistral",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="LLM provider (default: mistral)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key (uses environment variable if not specified)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NL2SPARQL Gradio Web UI")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print()

    print("Initializing translator...")
    init_translator(args.provider, args.model, args.api_key)
    print("Initializing ontology retriever...")
    init_ontology_retriever()
    print("Starting web UI...")

    app = create_ui()
    app.launch(
        share=args.share,
        server_port=args.port,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()

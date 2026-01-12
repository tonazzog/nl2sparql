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
    handle_check_variable_reuse,
    handle_fix_service_clause,
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


def translate_question(question: str) -> tuple[str, str, str, str, str]:
    """Translate a natural language question to SPARQL.

    Returns: (sparql, pipeline_info, patterns_info, validation_info, results_preview)
    """
    if not translator:
        return "Error: Translator not initialized", "", "", "", ""

    if not question.strip():
        return "Please enter a question", "", "", "", ""

    try:
        result = translator.translate(question)

        # Format pipeline info (generation strategy + retrieved examples)
        strategy_icon = "🔄" if result.generation_strategy == "adapt" else "🔨"
        pipeline_info = f"### Generation Strategy\n\n"
        pipeline_info += f"{strategy_icon} **{result.generation_strategy.upper()}**"
        if result.generation_strategy == "adapt":
            pipeline_info += " (adapting best matching example)\n\n"
        else:
            pipeline_info += " (synthesizing from multiple examples)\n\n"

        # Add retrieved examples
        pipeline_info += "### Retrieved Examples\n\n"
        if result.retrieved_examples:
            for i, ex in enumerate(result.retrieved_examples[:3]):
                score_bar = "█" * int(ex.score * 10)
                pipeline_info += f"**{i+1}. Example #{ex.example.id}** (score: {ex.score:.3f}) {score_bar}\n\n"
                nl_preview = ex.example.nl[:80] + ('...' if len(ex.example.nl) > 80 else '')
                pipeline_info += f"   *\"{nl_preview}\"*\n\n"
                if ex.example.patterns:
                    pipeline_info += f"   Patterns: {', '.join(ex.example.patterns.keys())}\n\n"
        else:
            pipeline_info += "No examples retrieved\n\n"

        # Add fix info if applicable
        if result.was_fixed:
            pipeline_info += f"### Auto-Fix Applied\n\n"
            pipeline_info += f"Query was fixed in **{result.fix_attempts}** attempt(s)\n"

        # Format patterns with scores
        patterns_info = f"### Detected Patterns\n\n"
        if result.pattern_scores:
            sorted_patterns = sorted(result.pattern_scores.items(), key=lambda x: -x[1])
            for pattern, score in sorted_patterns:
                bar = "█" * int(score * 10)
                highlight = "**" if pattern in result.detected_patterns else ""
                patterns_info += f"{highlight}{pattern}{highlight}: {score:.2%} {bar}\n\n"
        else:
            patterns_info += "No patterns detected\n"

        patterns_info += f"### Confidence\n\n"
        conf_bar = "█" * int(result.confidence * 10) + "░" * (10 - int(result.confidence * 10))
        patterns_info += f"**{result.confidence:.1%}** [{conf_bar}]\n"

        # Format validation
        if result.validation:
            v = result.validation
            if not v.is_valid:
                # Red ❌ for syntax or execution errors
                validation_info = f"❌ **Invalid Query**\n\n"
                if v.syntax_error:
                    validation_info += f"**Syntax Error:** {v.syntax_error}\n\n"
                if v.execution_error:
                    validation_info += f"**Execution Error:** {v.execution_error}\n\n"
            elif v.has_results:
                # Green ✅ for successful execution with results
                validation_info = f"✅ **Valid Query**\n\n"
                validation_info += f"**Results:** {v.result_count} rows"
            else:
                # Yellow ⚠️ for successful execution with no results
                validation_info = f"⚠️ **Query Executed** (0 results)\n\n"
                validation_info += "The query is syntactically correct but returned no data."

            # Add semantic warnings if any (shown for all cases)
            if v.has_warnings:
                validation_info += f"\n\n**Semantic Warnings:** {', '.join(v.semantic_errors)}"
        else:
            validation_info = "Validation not performed"

        # Format sample results
        results_preview = ""
        if result.validation and result.validation.sample_results:
            results_preview = "**Sample Results:**\n\n"
            for i, row in enumerate(result.validation.sample_results[:5]):
                results_preview += f"{i+1}. {row}\n"

        return result.sparql, pipeline_info, patterns_info, validation_info, results_preview

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", "", "", "", ""


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

def retrieve_examples(question: str, top_k: int) -> str:
    """Retrieve similar examples from the dataset without generating a query."""
    if not translator:
        return "Error: Translator not initialized"

    if not question.strip():
        return "Please enter a question"

    try:
        # Use the translator's retrieve method
        results = translator.retrieve(question, top_k=int(top_k))

        if not results:
            return "No examples found for this question."

        output = f"## Retrieved {len(results)} Examples\n\n"

        for i, r in enumerate(results):
            ex = r.example
            output += f"---\n\n"
            output += f"### Example #{ex.id}\n\n"

            # Scores
            output += f"**Scores:** "
            output += f"Combined: {r.score:.3f} | "
            output += f"Semantic: {r.semantic_score:.3f} | "
            output += f"BM25: {r.bm25_score:.3f} | "
            output += f"Pattern: {r.pattern_score:.3f}\n\n"

            # Score visualization
            score_bar = "█" * int(r.score * 20) + "░" * (20 - int(r.score * 20))
            output += f"[{score_bar}] {r.score:.1%}\n\n"

            # Natural language question
            output += f"**Question:** {ex.nl}\n\n"

            # Variants if available
            if ex.nl_variants:
                for lang, variant in ex.nl_variants.items():
                    output += f"**Question ({lang.upper()}):** {variant}\n\n"

            # Patterns
            if ex.patterns:
                patterns_str = ", ".join([f"{p} ({s:.0%})" for p, s in ex.patterns.items()])
                output += f"**Patterns:** {patterns_str}\n\n"

            # SPARQL query
            output += f"**SPARQL Query:**\n\n```sparql\n{ex.sparql}\n```\n\n"

        return output

    except Exception as e:
        import traceback
        return f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def fix_query_issues(sparql: str) -> tuple[str, str]:
    """Apply all available fixes to a SPARQL query.

    Fixes applied:
    1. SERVICE clause removal (for federated query permission errors)
    2. Case-sensitive filters (convert to case-insensitive REGEX)
    3. Variable reuse detection (warns but doesn't auto-fix)
    """
    if not sparql.strip():
        return sparql, "Please enter a SPARQL query"

    try:
        current_sparql = sparql
        all_changes = []
        fixes_applied = []
        warnings = []

        # Fix 1: SERVICE clause removal
        service_result = handle_fix_service_clause({"sparql": current_sparql})
        if service_result['was_modified']:
            current_sparql = service_result['sparql']
            fixes_applied.append("🔧 **SERVICE Clause**")
            all_changes.append(f"Removed SERVICE clause for `<{service_result['service_uri']}>`")

        # Fix 2: Case sensitivity
        case_result = handle_fix_case_sensitivity({"sparql": current_sparql})
        if case_result['was_modified']:
            current_sparql = case_result['sparql']
            fixes_applied.append("🔧 **Case Sensitivity**")
            all_changes.extend(case_result['changes'])

        # Check 3: Variable reuse (detection only, no auto-fix)
        var_result = handle_check_variable_reuse({"sparql": current_sparql})
        if var_result['has_issues']:
            warnings.append("⚠️ **Variable Reuse Issues**")
            for issue in var_result['issues']:
                warnings.append(f"  - {issue}")

        # Build output info
        info = ""

        if fixes_applied:
            info += "### Fixes Applied\n\n"
            for fix in fixes_applied:
                info += f"{fix}\n\n"
            info += "**Changes:**\n\n"
            for change in all_changes:
                info += f"- {change}\n"
            info += "\n"
        else:
            info += "### No Fixes Needed\n\nThe query appears to be clean.\n\n"

        if warnings:
            info += "### Warnings\n\n"
            for warning in warnings:
                info += f"{warning}\n"
            info += "\n*Variable reuse issues require manual fixing.*\n"

        return current_sparql, info

    except Exception as e:
        import traceback
        return sparql, f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
      
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
                    question_input = gr.Textbox(
                        label="Natural Language Question",
                        placeholder="e.g., Quali lemmi esprimono tristezza?",
                        lines=2,
                        scale=4,
                    )
                translate_btn = gr.Button("Translate to SPARQL", variant="primary")

                with gr.Row():
                    # Left column: SPARQL output
                    with gr.Column(scale=2):
                        sparql_output = gr.Code(
                            label="Generated SPARQL",
                            language="sql",  # SPARQL not supported, SQL is closest
                            lines=12,
                        )
                        with gr.Row():
                            validation_output = gr.Markdown(label="Validation")
                            results_output = gr.Markdown(label="Sample Results")

                    # Right column: Pipeline details
                    with gr.Column(scale=1):
                        with gr.Accordion("Pipeline Details", open=True):
                            pipeline_output = gr.Markdown(label="Pipeline")
                        with gr.Accordion("Pattern Analysis", open=True):
                            patterns_output = gr.Markdown(label="Patterns")

                translate_btn.click(
                    translate_question,
                    inputs=[question_input],
                    outputs=[sparql_output, pipeline_output, patterns_output, validation_output, results_output],
                )

                gr.Examples(
                    examples=[
                        ["Quali lemmi esprimono tristezza?"],
                        ["Trova le traduzioni siciliane di 'figlio'"],
                        ["What are the hypernyms of 'dog'?"],
                        ["Lemmi che indicano parti del corpo"],
                        ["Trova i sinonimi di 'veloce' con le loro definizioni"],
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

            # Tab 3: Retrieve Examples
            with gr.TabItem("Retrieve Examples"):
                gr.Markdown("""
                ### Example Retrieval

                Find similar examples from the training dataset using hybrid retrieval
                (semantic similarity + BM25 + pattern matching). This shows what examples
                would be used for few-shot learning when generating a query.
                """)
                with gr.Row():
                    retrieve_input = gr.Textbox(
                        label="Natural Language Question",
                        placeholder="e.g., Quali lemmi esprimono tristezza?",
                        lines=2,
                        scale=4,
                    )
                    retrieve_k = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Number of Examples",
                        scale=1,
                    )
                retrieve_btn = gr.Button("Retrieve Examples", variant="primary")
                retrieve_output = gr.Markdown()

                retrieve_btn.click(
                    retrieve_examples,
                    inputs=[retrieve_input, retrieve_k],
                    outputs=[retrieve_output],
                )

            # Tab 4: Search Ontology
            with gr.TabItem("Search Ontology"):
                with gr.Row():
                    onto_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., emotion, translation, hypernym",
                    )
                    onto_type = gr.Radio(
                        choices=["All", "Class", "Property"],
                        value="Property",
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

            # Tab 5: Execute Query
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

            # Tab 6: Fix Query
            with gr.TabItem("Fix Query"):
                gr.Markdown("""
                ### SPARQL Query Fixer

                This tool applies multiple fixes to problematic SPARQL queries:

                | Fix | Description |
                |-----|-------------|
                | **SERVICE Clause** | Removes federated SERVICE wrappers that cause permission errors |
                | **Case Sensitivity** | Converts `FILTER(STR(?x) = "value")` to case-insensitive REGEX |
                | **Variable Reuse** | Detects variables used both as URIs and literals (warning only) |
                """)
                fix_input = gr.Code(
                    label="SPARQL Query to Fix",
                    language="sql",
                    lines=10,
                )
                fix_btn = gr.Button("Apply All Fixes", variant="primary")
                with gr.Row():
                    fix_output = gr.Code(
                        label="Fixed SPARQL",
                        language="sql",
                        lines=10,
                    )
                    fix_info = gr.Markdown(label="Fix Report")

                fix_btn.click(
                    fix_query_issues,
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

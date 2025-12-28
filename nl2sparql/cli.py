"""Command-line interface for nl2sparql."""

import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__, AVAILABLE_PROVIDERS, LIITA_ENDPOINT


@click.group(invoke_without_command=True)
@click.option(
    "--version", "-v",
    is_flag=True,
    help="Show version and exit.",
)
@click.pass_context
def main(ctx: click.Context, version: bool):
    """
    NL2SPARQL: Translate natural language questions to SPARQL queries.

    Designed for the LiITA (Linked Italian) linguistic knowledge base.

    \b
    Examples:
        nl2sparql translate "Quali lemmi esprimono tristezza?"
        nl2sparql translate -p anthropic "Trova traduzioni siciliane"
        nl2sparql validate query.sparql
        nl2sparql list-models
    """
    if version:
        click.echo(f"nl2sparql version {__version__}")
        ctx.exit()

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("question")
@click.option(
    "--provider", "-p",
    type=click.Choice(list(AVAILABLE_PROVIDERS.keys())),
    default="openai",
    help="LLM provider to use.",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model name (uses provider default if not specified).",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file for the SPARQL query.",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate the generated query.",
)
@click.option(
    "--fix/--no-fix",
    default=True,
    help="Attempt to fix invalid queries.",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    help="Maximum fix attempts.",
)
@click.option(
    "--endpoint",
    type=str,
    default=LIITA_ENDPOINT,
    help="SPARQL endpoint for validation.",
)
@click.option(
    "--verbose", "-V",
    is_flag=True,
    help="Show detailed output.",
)
def translate(
    question: str,
    provider: str,
    model: Optional[str],
    output: Optional[str],
    validate: bool,
    fix: bool,
    max_retries: int,
    endpoint: str,
    verbose: bool,
):
    """
    Translate a natural language question to SPARQL.

    \b
    Examples:
        nl2sparql translate "Quali lemmi esprimono tristezza?"
        nl2sparql translate -p anthropic -m claude-sonnet-4-20250514 "Trova verbi con gioia"
        nl2sparql translate "Definizione di casa" -o query.sparql
    """
    from .generation.synthesizer import NL2SPARQL

    try:
        if verbose:
            click.echo(f"Provider: {provider}")
            click.echo(f"Model: {model or 'default'}")
            click.echo(f"Question: {question}")
            click.echo()

        # Initialize translator
        translator = NL2SPARQL(
            provider=provider,
            model=model,
            validate=validate,
            fix_errors=fix,
            max_retries=max_retries,
            endpoint=endpoint,
        )

        # Translate
        with click.progressbar(
            length=1,
            label="Translating",
            show_eta=False,
        ) as bar:
            result = translator.translate(question)
            bar.update(1)

        # Show results
        if verbose:
            click.echo()
            click.echo(f"Detected patterns: {', '.join(result.detected_patterns)}")
            click.echo(f"Confidence: {result.confidence:.2f}")
            click.echo(f"Retrieved {len(result.retrieved_examples)} examples")

            if result.was_fixed:
                click.echo(f"Query was fixed ({result.fix_attempts} attempts)")

            if result.validation:
                if result.validation.is_valid:
                    click.secho("Validation: PASSED", fg="green")
                    if result.validation.result_count is not None:
                        click.echo(f"Results: {result.validation.result_count}")
                else:
                    click.secho("Validation: FAILED", fg="red")
                    if result.validation.syntax_error:
                        click.echo(f"Syntax error: {result.validation.syntax_error}")
                    if result.validation.execution_error:
                        click.echo(f"Execution error: {result.validation.execution_error}")
                    if result.validation.semantic_errors:
                        click.echo("Semantic issues:")
                        for err in result.validation.semantic_errors:
                            click.echo(f"  - {err}")

            click.echo()

        # Output query
        click.echo("--- SPARQL Query ---")
        click.echo(result.sparql)
        click.echo("-------------------")

        # Save to file if requested
        if output:
            Path(output).write_text(result.sparql, encoding="utf-8")
            click.echo(f"Query saved to: {output}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@main.command("validate")
@click.argument("file_or_query")
@click.option(
    "--endpoint",
    type=str,
    default=LIITA_ENDPOINT,
    help="SPARQL endpoint for validation.",
)
def validate_command(file_or_query: str, endpoint: str):
    """
    Validate a SPARQL query.

    FILE_OR_QUERY can be a file path or a SPARQL query string.

    \b
    Examples:
        nl2sparql validate query.sparql
        nl2sparql validate "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
    """
    from .validation.syntax import validate_syntax
    from .validation.endpoint import validate_endpoint
    from .validation.semantic import validate_semantic

    # Determine if input is a file or query string
    if Path(file_or_query).exists():
        sparql = Path(file_or_query).read_text(encoding="utf-8")
        click.echo(f"Validating file: {file_or_query}")
    else:
        sparql = file_or_query
        click.echo("Validating query string")

    click.echo()

    # Syntax validation
    syntax_valid, syntax_error = validate_syntax(sparql)
    if syntax_valid:
        click.secho("Syntax: OK", fg="green")
    else:
        click.secho(f"Syntax: FAILED - {syntax_error}", fg="red")
        sys.exit(1)

    # Semantic validation
    semantic_valid, semantic_errors = validate_semantic(sparql)
    if semantic_valid:
        click.secho("Semantic: OK", fg="green")
    else:
        click.secho("Semantic: WARNINGS", fg="yellow")
        for err in semantic_errors:
            click.echo(f"  - {err}")

    # Endpoint validation
    exec_success, exec_error, result_count, sample = validate_endpoint(
        sparql=sparql,
        endpoint=endpoint,
    )

    if exec_success:
        click.secho(f"Execution: OK ({result_count} results)", fg="green")
        if sample:
            click.echo("Sample results:")
            for row in sample[:3]:
                click.echo(f"  {row}")
    else:
        click.secho(f"Execution: FAILED - {exec_error}", fg="red")

    click.echo()

    if syntax_valid and exec_success:
        click.secho("Query is VALID", fg="green", bold=True)
    else:
        click.secho("Query has ISSUES", fg="red", bold=True)
        sys.exit(1)


@main.command("list-models")
def list_models():
    """List available LLM providers and models."""
    click.echo("Available LLM Providers and Models:")
    click.echo()

    for provider, config in AVAILABLE_PROVIDERS.items():
        default = config["default_model"]
        models = config["models"]

        click.secho(f"{provider}:", fg="cyan", bold=True)
        for model in models:
            if model == default:
                click.echo(f"  - {model} (default)")
            else:
                click.echo(f"  - {model}")
        click.echo()


@main.command("retrieve")
@click.argument("question")
@click.option(
    "--top-k", "-k",
    type=int,
    default=5,
    help="Number of examples to retrieve.",
)
def retrieve_command(question: str, top_k: int):
    """
    Retrieve similar example queries (without generating).

    Useful for debugging and understanding retrieval.
    """
    from .retrieval.hybrid_retriever import HybridRetriever
    from .retrieval.patterns import infer_patterns

    retriever = HybridRetriever()
    patterns = infer_patterns(question)

    click.echo(f"Question: {question}")
    click.echo(f"Inferred patterns: {patterns}")
    click.echo()

    results = retriever.retrieve(
        query=question,
        user_patterns=patterns,
        top_k=top_k,
    )

    for i, result in enumerate(results, 1):
        click.echo(f"--- Result {i} (score: {result.score:.3f}) ---")
        click.echo(f"ID: {result.example.id}")
        click.echo(f"NL: {result.example.nl}")
        click.echo(f"Patterns: {result.example.patterns}")
        click.echo(f"Scores: semantic={result.semantic_score:.3f}, "
                   f"bm25={result.bm25_score:.3f}, pattern={result.pattern_score:.3f}")
        click.echo()


if __name__ == "__main__":
    main()

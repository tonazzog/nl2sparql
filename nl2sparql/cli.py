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


@main.command("evaluate")
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
    help="Model name.",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["it", "en"]),
    default="it",
    help="Test language (Italian or English).",
)
@click.option(
    "--category", "-c",
    type=click.Choice(["single_pattern", "combination_2", "combination_3", "complex"]),
    multiple=True,
    default=None,
    help="Filter by test category (can specify multiple).",
)
@click.option(
    "--pattern",
    type=str,
    multiple=True,
    default=None,
    help="Filter by pattern (can specify multiple).",
)
@click.option(
    "--no-endpoint",
    is_flag=True,
    help="Skip endpoint validation.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Save report to JSON file.",
)
@click.option(
    "--agent",
    is_flag=True,
    help="Use the agentic workflow (NL2SPARQLAgent) instead of the standard translator.",
)
def evaluate_command(
    provider: str,
    model: Optional[str],
    language: str,
    category: tuple,
    pattern: tuple,
    no_endpoint: bool,
    output: Optional[str],
    agent: bool,
):
    """
    Evaluate the system on the test dataset.

    \b
    Examples:
        nl2sparql evaluate
        nl2sparql evaluate -p anthropic -l en
        nl2sparql evaluate --agent  # Use agentic workflow
        nl2sparql evaluate -c single_pattern -c combination_2
        nl2sparql evaluate --pattern EMOTION_LEXICON --pattern TRANSLATION
        nl2sparql evaluate -o report.json
    """
    from .generation.synthesizer import NL2SPARQL
    from .evaluation import evaluate_dataset, print_report, save_report, AgentAdapter

    mode_str = "Agent" if agent else "Standard"
    click.echo(f"NL2SPARQL Evaluation ({mode_str} Mode)")
    click.echo(f"Provider: {provider}, Model: {model or 'default'}")
    click.echo(f"Language: {language}")
    click.echo()

    # Initialize translator or agent
    if agent:
        from .agent import NL2SPARQLAgent
        nl2sparql_agent = NL2SPARQLAgent(
            provider=provider,
            model=model,
        )
        translator = AgentAdapter(nl2sparql_agent)
    else:
        translator = NL2SPARQL(
            provider=provider,
            model=model,
            validate=True,
            fix_errors=True,
        )

    # Run evaluation
    categories = list(category) if category else None
    patterns = list(pattern) if pattern else None

    click.echo("Running evaluation...")
    report = evaluate_dataset(
        translator=translator,
        language=language,
        validate_endpoint=not no_endpoint,
        categories=categories,
        patterns=patterns,
    )

    # Print report
    print_report(report)

    # Save if requested
    if output:
        save_report(report, output)
        click.echo(f"\nReport saved to: {output}")


@main.command("batch-evaluate")
@click.option(
    "--preset", "-p",
    type=click.Choice(["openai", "anthropic", "mistral", "all_defaults", "quick"]),
    default=None,
    help="Use a preset model configuration.",
)
@click.option(
    "--provider",
    multiple=True,
    help="Provider to evaluate (can specify multiple).",
)
@click.option(
    "--model",
    multiple=True,
    help="Model to evaluate (pairs with --provider).",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["it", "en"]),
    default="it",
    help="Test language.",
)
@click.option(
    "--no-endpoint",
    is_flag=True,
    help="Skip endpoint validation.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default=None,
    help="Directory to save individual reports.",
)
@click.option(
    "--comparison", "-c",
    type=click.Path(),
    default=None,
    help="Path to save comparison report.",
)
@click.option(
    "--agent",
    is_flag=True,
    help="Use the agentic workflow (NL2SPARQLAgent) instead of the standard translator.",
)
def batch_evaluate_command(
    preset: Optional[str],
    provider: tuple,
    model: tuple,
    language: str,
    no_endpoint: bool,
    output_dir: Optional[str],
    comparison: Optional[str],
    agent: bool,
):
    """
    Evaluate multiple LLM models and compare results.

    \\b
    Examples:
        nl2sparql batch-evaluate -p quick
        nl2sparql batch-evaluate -p openai -o ./reports
        nl2sparql batch-evaluate --agent -p quick  # Use agentic workflow
        nl2sparql batch-evaluate --provider openai --provider anthropic
        nl2sparql batch-evaluate -p all_defaults -c comparison.json

    \\b
    Available presets:
        quick         - GPT-4.1-mini + Claude 3.5 Haiku (fast comparison)
        openai        - All OpenAI models (GPT-4.1, GPT-4.1-mini, GPT-4.1-nano)
        anthropic     - All Anthropic models (Claude Sonnet 4, Claude 3.5 Haiku)
        mistral       - All Mistral models (Large, Small)
        all_defaults  - Default model from each provider
    """
    from .evaluation import (
        ModelConfig,
        run_batch_evaluation,
        create_comparison_report,
        print_comparison,
        PRESETS,
    )

    # Build config list
    mode_str = "Agent" if agent else "Standard"
    if preset:
        configs = PRESETS[preset]
        click.echo(f"Using preset: {preset} ({len(configs)} models) - {mode_str} Mode")
    elif provider:
        models = list(model) if model else [None] * len(provider)
        if len(models) < len(provider):
            models.extend([None] * (len(provider) - len(models)))
        configs = [
            ModelConfig(p, m)
            for p, m in zip(provider, models)
        ]
        click.echo(f"Evaluating {len(configs)} custom configuration(s) - {mode_str} Mode")
    else:
        click.secho("Error: Specify --preset or --provider", fg="red", err=True)
        sys.exit(1)

    click.echo(f"Language: {language}")
    click.echo()

    # Run batch evaluation
    results = run_batch_evaluation(
        configs=configs,
        language=language,
        validate_endpoint=not no_endpoint,
        output_dir=output_dir,
        verbose=True,
        use_agent=agent,
    )

    # Create and print comparison
    comp = create_comparison_report(
        results,
        output_path=comparison,
    )
    print_comparison(comp)

    if comparison:
        click.echo(f"\nComparison saved to: {comparison}")

    if output_dir:
        click.echo(f"Individual reports saved to: {output_dir}")


AGENT_PROVIDERS = ["openai", "anthropic", "mistral", "gemini", "ollama"]


@main.command("agent")
@click.argument("question")
@click.option(
    "--provider", "-p",
    type=click.Choice(AGENT_PROVIDERS),
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
    "--language", "-l",
    type=click.Choice(["it", "en"]),
    default="it",
    help="Question language.",
)
@click.option(
    "--verbose", "-V",
    is_flag=True,
    help="Show detailed progress.",
)
@click.option(
    "--stream", "-s",
    is_flag=True,
    help="Stream the translation process step by step.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Save query to file.",
)
def agent_command(
    question: str,
    provider: str,
    model: Optional[str],
    language: str,
    verbose: bool,
    stream: bool,
    output: Optional[str],
):
    """
    Translate using the agentic LangGraph workflow.

    This command uses an intelligent agent that can:
    - Analyze query complexity
    - Decompose complex queries
    - Execute and verify results
    - Self-correct based on errors
    - Explore schema when needed

    \\b
    Examples:
        nl2sparql agent "Find all nouns expressing sadness"
        nl2sparql agent -p anthropic "Trova aggettivi con traduzioni"
        nl2sparql agent -p openai -m gpt-4.1 "Complex query here"
        nl2sparql agent --stream "Show step-by-step progress"
    """
    try:
        from .agent import NL2SPARQLAgent
    except ImportError as e:
        click.secho(
            f"Agent requires additional dependencies: {e}\n"
            "Install with: pip install liita-nl2sparql[agent]",
            fg="red",
            err=True
        )
        sys.exit(1)

    if verbose:
        click.echo(f"Provider: {provider}")
        if model:
            click.echo(f"Model: {model}")

    agent = NL2SPARQLAgent(provider=provider, model=model)

    if stream:
        click.echo(f"Translating: {question}")
        click.echo("=" * 60)

        final_state = None
        for node_name, state in agent.stream(question, language):
            final_state = state  # Keep track of accumulated state
            click.secho(f"\n[{node_name}]", fg="cyan", bold=True)

            if node_name == "analyze":
                click.echo(f"  Patterns: {state.get('detected_patterns', [])}")
                click.echo(f"  Complexity: {state.get('complexity', 'unknown')}")

            elif node_name == "plan":
                tasks = state.get("sub_tasks", [])
                if len(tasks) > 1:
                    click.echo("  Sub-tasks:")
                    for i, task in enumerate(tasks, 1):
                        click.echo(f"    {i}. {task}")

            elif node_name == "retrieve":
                examples = state.get("retrieved_examples", [])
                click.echo(f"  Retrieved {len(examples)} examples")

            elif node_name == "generate":
                attempts = state.get("generation_attempts", 0)
                click.echo(f"  Generation attempt: {attempts}")

            elif node_name == "execute":
                count = state.get("result_count", 0)
                error = state.get("execution_error")
                if error:
                    click.secho(f"  Error: {error}", fg="red")
                else:
                    click.echo(f"  Results: {count}")

            elif node_name == "verify":
                is_valid = state.get("is_valid", False)
                if is_valid:
                    click.secho("  Valid!", fg="green")
                else:
                    errors = state.get("validation_errors", [])
                    click.secho(f"  Issues: {errors}", fg="yellow")

            elif node_name == "refine":
                click.echo("  Preparing to retry...")

            elif node_name == "explore":
                props = state.get("discovered_properties", [])
                click.echo(f"  Discovered {len(props)} properties")

            elif node_name == "output":
                click.echo("  Finalizing...")

        # Get final result from accumulated state (no need to run workflow again)
        result = agent.get_final_result(final_state)

    else:
        # Non-streaming mode
        if verbose:
            click.echo(f"Translating: {question}")
            click.echo("-" * 50)

        result = agent.translate(question, language, verbose=verbose)

    # Display result
    click.echo("\n" + "=" * 60)

    if result["is_valid"]:
        click.secho("SUCCESS", fg="green", bold=True)
    else:
        click.secho("BEST EFFORT (validation issues)", fg="yellow", bold=True)

    click.echo(f"Confidence: {result['confidence']:.0%}")
    click.echo(f"Attempts: {result['attempts']}")
    click.echo(f"Results: {result['result_count']}")
    click.echo(f"Patterns: {result['detected_patterns']}")

    click.echo("\nGenerated SPARQL:")
    click.echo("-" * 60)
    click.echo(result["sparql"])

    if result["refinement_history"]:
        click.echo(f"\nRefinement history ({len(result['refinement_history'])} attempts):")
        for i, attempt in enumerate(result["refinement_history"], 1):
            click.echo(f"  {i}. Error: {attempt.get('error', 'unknown')[:80]}")

    if output:
        Path(output).write_text(result["sparql"])
        click.echo(f"\nQuery saved to: {output}")


@main.command("agent-viz")
def agent_viz_command():
    """
    Show the agent workflow graph as Mermaid diagram.

    Copy the output to https://mermaid.live to visualize.
    """
    try:
        from .agent import get_graph_visualization
    except ImportError as e:
        click.secho(
            f"Agent requires additional dependencies: {e}\n"
            "Install with: pip install liita-nl2sparql[agent]",
            fg="red",
            err=True
        )
        sys.exit(1)

    mermaid = get_graph_visualization()

    if mermaid:
        click.echo("Copy this Mermaid diagram to https://mermaid.live to visualize:\n")
        click.echo(mermaid)
    else:
        click.secho("Could not generate visualization.", fg="red")


@main.command("generate-synthetic")
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="Output file path (e.g., synthetic_data.jsonl).",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["jsonl", "json", "alpaca", "sharegpt", "hf"]),
    default="jsonl",
    help="Output format.",
)
@click.option(
    "--num-variations", "-n",
    type=int,
    default=5,
    help="Number of NL variations per seed example.",
)
@click.option(
    "--num-combinations", "-c",
    type=int,
    default=10,
    help="Number of pattern combination questions to generate.",
)
@click.option(
    "--max-pairs", "-m",
    type=int,
    default=None,
    help="Maximum total pairs to generate.",
)
@click.option(
    "--min-results",
    type=int,
    default=1,
    help="Minimum query results required for valid pair.",
)
@click.option(
    "--provider", "-p",
    type=click.Choice(AGENT_PROVIDERS),
    default="openai",
    help="LLM provider to use.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (uses provider default if not specified).",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["it", "en"]),
    default="it",
    help="Language for generated questions.",
)
@click.option(
    "--no-agent",
    is_flag=True,
    help="Use standard translator instead of agent for SPARQL generation.",
)
@click.option(
    "--include-seeds/--no-seeds",
    default=True,
    help="Include original seed examples in output.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress progress output.",
)
def generate_synthetic_command(
    output: str,
    format: str,
    num_variations: int,
    num_combinations: int,
    max_pairs: Optional[int],
    min_results: int,
    provider: str,
    model: Optional[str],
    language: str,
    no_agent: bool,
    include_seeds: bool,
    quiet: bool,
):
    """
    Generate synthetic training data for fine-tuning.

    Creates (NL question, SPARQL query) pairs by:
    1. Generating NL variations of seed examples
    2. Creating pattern combination questions
    3. Validating all generated SPARQL against the endpoint

    \\b
    Examples:
        nl2sparql generate-synthetic -o data.jsonl
        nl2sparql generate-synthetic -o data.json -f json -n 10
        nl2sparql generate-synthetic -o train.jsonl -m 500 -p anthropic
        nl2sparql generate-synthetic -o alpaca.json -f alpaca --no-seeds

    \\b
    Output formats:
        jsonl    - JSON Lines (one record per line, with metadata)
        json     - JSON array (with metadata)
        alpaca   - Alpaca format for fine-tuning (instruction/input/output)
        sharegpt - ShareGPT format for chat fine-tuning
        hf       - HuggingFace datasets format (directory)
    """
    try:
        from .synthetic import SyntheticDataGenerator
    except ImportError as e:
        click.secho(
            f"Synthetic generation requires agent dependencies: {e}\n"
            "Install with: pip install liita-nl2sparql[agent]",
            fg="red",
            err=True
        )
        sys.exit(1)

    if not quiet:
        click.echo("NL2SPARQL Synthetic Data Generator")
        click.echo("=" * 60)
        click.echo(f"Provider: {provider}, Model: {model or 'default'}")
        click.echo(f"Mode: {'Standard translator' if no_agent else 'Agent'}")
        click.echo(f"Variations per seed: {num_variations}")
        click.echo(f"Pattern combinations: {num_combinations}")
        if max_pairs:
            click.echo(f"Max pairs: {max_pairs}")
        click.echo(f"Output: {output} ({format})")
        click.echo()

    generator = SyntheticDataGenerator(
        provider=provider,
        model=model,
        use_agent=not no_agent,
    )

    pairs, stats = generator.generate_dataset(
        num_variations_per_seed=num_variations,
        num_combinations=num_combinations,
        min_results=min_results,
        max_pairs=max_pairs,
        language=language,
        include_seeds=include_seeds,
        verbose=not quiet,
    )

    generator.save_dataset(pairs, output, format=format)

    if not quiet:
        click.echo()
        click.secho(f"Successfully generated {len(pairs)} training pairs!", fg="green")
        click.echo(f"Output saved to: {output}")

        # Show summary by method
        by_method = {}
        for p in pairs:
            by_method[p.generation_method] = by_method.get(p.generation_method, 0) + 1
        click.echo(f"\nBreakdown by method:")
        for method, count in sorted(by_method.items()):
            click.echo(f"  {method}: {count}")


if __name__ == "__main__":
    main()

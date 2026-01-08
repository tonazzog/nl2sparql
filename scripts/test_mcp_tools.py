#!/usr/bin/env python3
"""
Direct test script for NL2SPARQL MCP tools.

This script tests the MCP tool handlers directly without running the MCP server.
Much simpler for local testing and debugging.

Usage:
    python scripts/test_mcp_tools.py
    python scripts/test_mcp_tools.py --provider anthropic
    python scripts/test_mcp_tools.py --tool translate --question "Find words expressing sadness"
"""

import argparse
import json
import sys
import time


def test_infer_patterns():
    """Test the infer_patterns tool."""
    from nl2sparql.mcp.tools import handle_infer_patterns

    print("\n" + "=" * 60)
    print("Testing: infer_patterns")
    print("=" * 60)

    questions = [
        "Trova le traduzioni siciliane di 'casa'",
        "Quali lemmi esprimono tristezza?",
        "What are the hypernyms of 'dog'?",
        "Definizione di amore",
    ]

    for q in questions:
        result = handle_infer_patterns({"question": q})
        print(f"\nQuestion: {q}")
        print(f"  Top patterns: {result['top_patterns']}")
        print(f"  Complexity: {result['complexity']}")


def test_retrieve_examples():
    """Test the retrieve_examples tool."""
    from nl2sparql.mcp.tools import handle_retrieve_examples
    from nl2sparql.retrieval import HybridRetriever

    print("\n" + "=" * 60)
    print("Testing: retrieve_examples")
    print("=" * 60)

    retriever = HybridRetriever()

    result = handle_retrieve_examples(
        {"question": "Quali lemmi esprimono tristezza?", "top_k": 3},
        retriever
    )

    print(f"\nRetrieved {len(result['examples'])} examples:")
    for ex in result['examples']:
        print(f"\n  ID: {ex['id']}, Score: {ex['score']:.3f}")
        print(f"  NL: {ex['nl'][:60]}...")
        print(f"  SPARQL: {ex['sparql'][:60]}...")


def test_search_ontology():
    """Test the search_ontology tool."""
    from nl2sparql.mcp.tools import handle_search_ontology
    from nl2sparql.retrieval import OntologyRetriever

    print("\n" + "=" * 60)
    print("Testing: search_ontology")
    print("=" * 60)

    retriever = OntologyRetriever()

    queries = [
        ("emotion feeling", "property"),
        ("broader meaning hypernym", "property"),
        ("lexical entry", "class"),
    ]

    for query, entry_type in queries:
        result = handle_search_ontology(
            {"query": query, "top_k": 3, "entry_type": entry_type},
            retriever
        )

        print(f"\nQuery: '{query}' (type: {entry_type})")
        for entry in result['entries']:
            print(f"  {entry['prefix_local']}: {entry['description'][:50]}...")


def test_get_constraints():
    """Test the get_constraints tool."""
    from nl2sparql.mcp.tools import handle_get_constraints

    print("\n" + "=" * 60)
    print("Testing: get_constraints")
    print("=" * 60)

    patterns_list = [
        ["EMOTION_LEXICON"],
        ["TRANSLATION"],
        ["EMOTION_LEXICON", "TRANSLATION"],
    ]

    for patterns in patterns_list:
        result = handle_get_constraints({"patterns": patterns})
        print(f"\nPatterns: {patterns}")
        print(f"  Constraints length: {len(result['constraints'])} chars")
        print(f"  Prefixes length: {len(result['prefixes'])} chars")
        print(f"  Preview: {result['constraints'][:100]}...")


def test_fix_case_sensitivity():
    """Test the fix_case_sensitivity tool."""
    from nl2sparql.mcp.tools import handle_fix_case_sensitivity

    print("\n" + "=" * 60)
    print("Testing: fix_case_sensitivity")
    print("=" * 60)

    queries = [
        'SELECT ?x WHERE { FILTER(STR(?x) = "rabbia") }',
        'SELECT ?x WHERE { FILTER(?label = "Tristezza") }',
        'SELECT ?x WHERE { ?x rdfs:label ?label . FILTER(STR(?label) = "amore") }',
    ]

    for sparql in queries:
        result = handle_fix_case_sensitivity({"sparql": sparql})
        print(f"\nOriginal: {sparql[:60]}...")
        print(f"  Modified: {result['was_modified']}")
        if result['changes']:
            print(f"  Changes: {result['changes']}")


def test_check_variable_reuse():
    """Test the check_variable_reuse tool."""
    from nl2sparql.mcp.tools import handle_check_variable_reuse

    print("\n" + "=" * 60)
    print("Testing: check_variable_reuse")
    print("=" * 60)

    queries = [
        # Bad: ?word used as both URI and literal
        """SELECT ?word WHERE {
            ?word ontolex:canonicalForm [ ontolex:writtenRep ?word ] .
        }""",
        # Good: different variables
        """SELECT ?entry ?written WHERE {
            ?entry ontolex:canonicalForm [ ontolex:writtenRep ?written ] .
        }""",
    ]

    for sparql in queries:
        result = handle_check_variable_reuse({"sparql": sparql})
        print(f"\nQuery: {sparql[:50]}...")
        print(f"  Has issues: {result['has_issues']}")
        if result['issues']:
            for issue in result['issues']:
                print(f"  Issue: {issue[:80]}...")


def test_validate_sparql():
    """Test the validate_sparql tool."""
    from nl2sparql.mcp.tools import handle_validate_sparql

    print("\n" + "=" * 60)
    print("Testing: validate_sparql")
    print("=" * 60)

    # Valid query
    valid_query = """
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
        SELECT ?lemma ?written WHERE {
            GRAPH <http://liita.it/data> {
                ?lemma a lila:Lemma ;
                       ontolex:writtenRep ?written .
            }
        } LIMIT 5
    """

    # Invalid query (syntax error)
    invalid_query = "SELECT ?x WHERE { ?x }"

    for sparql, name in [(valid_query, "Valid"), (invalid_query, "Invalid")]:
        print(f"\n{name} query:")
        result = handle_validate_sparql(
            {"sparql": sparql, "check_endpoint": name == "Valid"},
            endpoint="https://liita.it/sparql",
            timeout=30
        )
        print(f"  Is valid: {result['is_valid']}")
        print(f"  Syntax: {result.get('syntax', {})}")
        if 'endpoint' in result:
            print(f"  Endpoint: success={result['endpoint']['success']}, "
                  f"count={result['endpoint']['result_count']}")


def test_execute_sparql():
    """Test the execute_sparql tool."""
    from nl2sparql.mcp.tools import handle_execute_sparql

    print("\n" + "=" * 60)
    print("Testing: execute_sparql")
    print("=" * 60)

    sparql = """
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
        SELECT ?lemma ?written WHERE {
            GRAPH <http://liita.it/data> {
                ?lemma a lila:Lemma ;
                       ontolex:writtenRep ?written .
            }
        } LIMIT 5
    """

    result = handle_execute_sparql(
        {"sparql": sparql, "limit": 5},
        endpoint="https://liita.it/sparql",
        default_timeout=30
    )

    print(f"\nSuccess: {result['success']}")
    print(f"Result count: {result['result_count']}")
    print(f"Variables: {result['variables']}")
    if result['results']:
        print("Results:")
        for row in result['results'][:3]:
            print(f"  {row}")


def test_translate(provider: str, model: str | None = None):
    """Test the translate tool (requires LLM)."""
    from nl2sparql.mcp.tools import handle_translate
    from nl2sparql import NL2SPARQL

    print("\n" + "=" * 60)
    print(f"Testing: translate (provider={provider})")
    print("=" * 60)

    translator = NL2SPARQL(
        provider=provider,
        model=model,
        validate=True,
        fix_errors=True,
    )

    questions = [
        "Quali lemmi esprimono tristezza?",
        "Find the Sicilian translations of 'house'",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        print("Translating...")
        start = time.time()
        result = handle_translate({"question": q}, translator)
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Patterns: {result['detected_patterns']}")
        print(f"  Results: {result['result_count']}")
        print(f"  SPARQL:\n{result['sparql']}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct test script for NL2SPARQL MCP tools"
    )
    parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="LLM provider for translate tool (default: openai)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--tool", "-t",
        default=None,
        choices=[
            "infer_patterns", "retrieve_examples", "search_ontology",
            "get_constraints", "fix_case_sensitivity", "check_variable_reuse",
            "validate_sparql", "execute_sparql", "translate"
        ],
        help="Test only a specific tool"
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Custom question for translate/infer_patterns tool"
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip the translate test (which requires LLM API)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NL2SPARQL MCP Tools - Direct Test")
    print("=" * 60)

    if args.tool:
        # Run single tool test
        if args.tool == "translate":
            if args.question:
                from nl2sparql.mcp.tools import handle_translate
                from nl2sparql import NL2SPARQL
                translator = NL2SPARQL(provider=args.provider, model=args.model)
                result = handle_translate({"question": args.question}, translator)
                print(json.dumps(result, indent=2))
            else:
                test_translate(args.provider, args.model)
        elif args.tool == "infer_patterns":
            if args.question:
                from nl2sparql.mcp.tools import handle_infer_patterns
                result = handle_infer_patterns({"question": args.question})
                print(json.dumps(result, indent=2))
            else:
                test_infer_patterns()
        elif args.tool == "retrieve_examples":
            test_retrieve_examples()
        elif args.tool == "search_ontology":
            test_search_ontology()
        elif args.tool == "get_constraints":
            test_get_constraints()
        elif args.tool == "fix_case_sensitivity":
            test_fix_case_sensitivity()
        elif args.tool == "check_variable_reuse":
            test_check_variable_reuse()
        elif args.tool == "validate_sparql":
            test_validate_sparql()
        elif args.tool == "execute_sparql":
            test_execute_sparql()
    else:
        # Run all tests
        test_infer_patterns()
        test_retrieve_examples()
        test_search_ontology()
        test_get_constraints()
        test_fix_case_sensitivity()
        test_check_variable_reuse()
        test_validate_sparql()
        test_execute_sparql()

        if not args.skip_translate:
            test_translate(args.provider, args.model)
        else:
            print("\n(Skipping translate test)")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

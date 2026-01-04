"""Node implementations for the NL2SPARQL LangGraph agent."""

import json
import re
from typing import Any

from .state import NL2SPARQLState


def get_llm(
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0,
    tier: str = "default",
    api_key: str | None = None,
):
    """
    Get LLM client for the specified provider.

    Args:
        provider: LLM provider ("openai", "anthropic", "mistral", "gemini", "ollama")
        model: Model name (uses provider default if None)
        temperature: Sampling temperature
        tier: Model tier - "fast" for cheaper/faster, "default" for standard capability
        api_key: API key (uses environment variable if None)

    Returns:
        LangChain chat model instance
    """
    # Default models per provider and tier
    DEFAULT_MODELS = {
        "openai": {"fast": "gpt-4.1-mini", "default": "gpt-4.1"},
        "anthropic": {"fast": "claude-3-5-haiku-latest", "default": "claude-sonnet-4-20250514"},
        "mistral": {"fast": "mistral-small-latest", "default": "mistral-large-latest"},
        "gemini": {"fast": "gemini-1.5-flash", "default": "gemini-1.5-pro"},
        "ollama": {"fast": "llama3.2", "default": "llama3.2"},
    }

    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: {list(DEFAULT_MODELS.keys())}")

    # Use provided model or default based on tier
    if model is None:
        model = DEFAULT_MODELS[provider][tier]

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            kwargs = {"model": model, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            return ChatOpenAI(**kwargs)
        except ImportError:
            raise ImportError("langchain-openai required. Install with: pip install langchain-openai")

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            kwargs = {"model": model, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            return ChatAnthropic(**kwargs)
        except ImportError:
            raise ImportError("langchain-anthropic required. Install with: pip install langchain-anthropic")

    elif provider == "mistral":
        try:
            from langchain_mistralai import ChatMistralAI
            kwargs = {"model": model, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            return ChatMistralAI(**kwargs)
        except ImportError:
            raise ImportError("langchain-mistralai required. Install with: pip install langchain-mistralai")

    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            kwargs = {"model": model, "temperature": temperature}
            if api_key:
                kwargs["google_api_key"] = api_key
            return ChatGoogleGenerativeAI(**kwargs)
        except ImportError:
            raise ImportError("langchain-google-genai required. Install with: pip install langchain-google-genai")

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            # Ollama doesn't use API keys
            return ChatOllama(model=model, temperature=temperature)
        except ImportError:
            raise ImportError("langchain-ollama required. Install with: pip install langchain-ollama")


def analyze_question(state: NL2SPARQLState) -> dict[str, Any]:
    """Analyze the question to understand what's needed."""
    from langchain_core.messages import HumanMessage, SystemMessage

    # Use fast tier for analysis (cheaper, faster)
    llm = get_llm(
        provider=state["provider"],
        model=state["model"],
        tier="fast",
        api_key=state["api_key"],
    )

    analysis_prompt = f"""Analyze this natural language question for SPARQL translation to the LiITA linguistic knowledge base.

Question: {state["question"]}
Language: {state["language"]}

Determine:
1. patterns: List of patterns needed. Choose from:
   - EMOTION_LEXICON (emotions, feelings)
   - TRANSLATION (dialect translations - Sicilian, Parmigiano)
   - SENSE_DEFINITION (word definitions from CompL-it)
   - SEMANTIC_RELATION (hypernyms, hyponyms, meronyms)
   - POS_FILTER (part of speech filtering - nouns, verbs, etc.)
   - MORPHO_REGEX (word patterns - starts with, ends with)
   - COUNT_ENTITIES (counting queries)

2. complexity: "simple" (1 pattern), "moderate" (2 patterns), "complex" (3+ patterns or multi-dialect)

3. requires_service: true ONLY if needs CompL-it data (definitions, semantic relations)
   - TRUE for: definitions, hypernyms, hyponyms, meronyms, semantic relations
   - FALSE for: emotions (ELITA), translations (dialects), POS filters, morphology

4. requires_translation: true if needs dialect translations

5. dialects: list of dialects needed, e.g., ["sicilian", "parmigiano"]

Respond in JSON format only:
{{"patterns": [...], "complexity": "...", "requires_service": true/false, "requires_translation": true/false, "dialects": [...]}}"""

    response = llm.invoke([
        SystemMessage(content="You are an expert at analyzing linguistic queries for the LiITA knowledge base."),
        HumanMessage(content=analysis_prompt)
    ])

    try:
        # Extract JSON from response
        content = response.content
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        analysis = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        # Fallback to basic analysis
        analysis = {
            "patterns": ["POS_FILTER"],
            "complexity": "simple",
            "requires_service": False,
            "requires_translation": False,
            "dialects": []
        }

    return {
        "detected_patterns": analysis.get("patterns", []),
        "complexity": analysis.get("complexity", "simple"),
        "requires_service": analysis.get("requires_service", False),
        "requires_translation": analysis.get("requires_translation", False),
        "dialects_needed": analysis.get("dialects", []),
    }


def plan_query(state: NL2SPARQLState) -> dict[str, Any]:
    """Break complex queries into sub-tasks."""
    from langchain_core.messages import HumanMessage

    if state["complexity"] == "simple":
        return {"sub_tasks": [state["question"]], "current_task_index": 0}

    # Use fast tier for planning
    llm = get_llm(
        provider=state["provider"],
        model=state["model"],
        tier="fast",
        api_key=state["api_key"],
    )

    planning_prompt = f"""Break this complex query into logical SPARQL construction steps:

Question: {state["question"]}
Patterns needed: {state["detected_patterns"]}
Dialects: {state["dialects_needed"]}
Needs CompL-it SERVICE: {state["requires_service"]}

Create a step-by-step plan. Each step should describe what SPARQL patterns to add.

Example for "Find Italian adjectives with Sicilian and Parmigiano translations and definitions":
1. "Query CompL-it SERVICE for Italian adjectives with definitions"
2. "Link to LiITA lemma and filter by POS"
3. "Add Sicilian translation via separate Italian lexical entry"
4. "Add Parmigiano translation via separate Italian lexical entry"

Return as JSON array of step descriptions:
["step1", "step2", ...]"""

    response = llm.invoke([HumanMessage(content=planning_prompt)])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        sub_tasks = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        sub_tasks = [state["question"]]

    return {
        "sub_tasks": sub_tasks,
        "current_task_index": 0
    }


def retrieve_examples(state: NL2SPARQLState) -> dict[str, Any]:
    """Retrieve relevant examples and constraints."""
    from ..retrieval.hybrid_retriever import HybridRetriever
    from ..constraints.prompt_builder import get_constraints_for_patterns

    retriever = HybridRetriever()

    # Get current sub-task or full question
    current_query = state["question"]
    if state["sub_tasks"]:
        idx = min(state["current_task_index"], len(state["sub_tasks"]) - 1)
        current_query = state["sub_tasks"][idx]

    # Retrieve examples
    examples = retriever.retrieve(current_query, top_k=5)

    # Build constraints based on detected patterns
    constraints = get_constraints_for_patterns(state["detected_patterns"])

    return {
        "retrieved_examples": [
            {"nl": ex.example.nl, "sparql": ex.example.sparql, "score": ex.score}
            for ex in examples
        ],
        "relevant_constraints": constraints
    }


def generate_sparql(state: NL2SPARQLState) -> dict[str, Any]:
    """Generate SPARQL query using LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage

    # Use default tier for generation (more capable model)
    llm = get_llm(
        provider=state["provider"],
        model=state["model"],
        tier="default",
        api_key=state["api_key"],
    )

    # Build refinement context from history
    refinement_context = ""
    if state["refinement_history"]:
        refinement_context = "\n\n## PREVIOUS ATTEMPTS (learn from these errors!):\n"
        for i, attempt in enumerate(state["refinement_history"][-3:], 1):
            refinement_context += f"\n### Attempt {i}:\n"
            refinement_context += f"Query:\n```sparql\n{attempt.get('sparql', 'N/A')[:500]}\n```\n"
            refinement_context += f"Error: {attempt.get('error', 'Unknown')}\n"
            refinement_context += f"Results: {attempt.get('result_count', 0)}\n"

    # Build examples text
    examples_text = ""
    for ex in state["retrieved_examples"][:3]:
        examples_text += f"\n### Example:\nQuestion: {ex['nl']}\n```sparql\n{ex['sparql']}\n```\n"

    # Include discovered schema if available
    schema_context = ""
    if state["discovered_properties"]:
        schema_context = f"\n\n## Discovered Properties:\n{', '.join(state['discovered_properties'][:20])}"

    # Explicit SERVICE instruction based on requires_service flag
    if state["requires_service"]:
        service_instruction = """- This query REQUIRES a SERVICE block to query CompL-it
- Use: SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> { ... }"""
    else:
        service_instruction = """- DO NOT use any SERVICE block for this query
- All data needed is in the main LiITA graphs (no external federation needed)"""

    generation_prompt = f"""Generate a SPARQL query for the LiITA knowledge base.

## CRITICAL CONSTRAINTS
{state["relevant_constraints"]}

## Similar Examples
{examples_text}
{schema_context}
{refinement_context}

## Question to translate:
{state["question"]}

## Requirements:
- Patterns needed: {state["detected_patterns"]}
- Dialects: {state["dialects_needed"]}

## SERVICE BLOCK INSTRUCTION:
{service_instruction}

Generate ONLY the complete SPARQL query. No explanations."""

    system_prompt = """You are an expert SPARQL query generator for the LiITA linguistic knowledge base.

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

    CORRECT pattern:
    GRAPH <http://liita.it/data> {
        ?lemma a lila:Lemma ; ontolex:writtenRep ?writtenRep .
    }
    SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
        ?word ontolex:canonicalForm [ ontolex:writtenRep ?writtenRep ] ;
              ontolex:sense [ skos:definition ?definition ] .
    }

    WRONG - causes "variable not assigned" error:
    GRAPH <http://liita.it/data> {
        ?lemma a lila:Lemma ; ontolex:writtenRep ?lilaRep .
    }
    SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {
        ?word ontolex:canonicalForm [ ontolex:writtenRep ?complRep ] .
        FILTER(STR(?complRep) = STR(?lilaRep))  # ERROR: ?lilaRep not visible inside SERVICE!
    }"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=generation_prompt)
    ])

    # Extract SPARQL from response
    sparql = response.content.strip()

    # Remove markdown code blocks if present
    if "```sparql" in sparql:
        sparql = sparql.split("```sparql")[1].split("```")[0].strip()
    elif "```" in sparql:
        sparql = sparql.split("```")[1].split("```")[0].strip()

    return {
        "generated_sparql": sparql,
        "generation_attempts": state.get("generation_attempts", 0) + 1
    }


def execute_query(state: NL2SPARQLState) -> dict[str, Any]:
    """Execute the SPARQL query against the endpoint."""
    from ..validation.endpoint import validate_endpoint
    from ..validation.syntax import validate_syntax

    # First check syntax
    syntax_valid, syntax_error = validate_syntax(state["generated_sparql"])

    if not syntax_valid:
        return {
            "execution_result": None,
            "result_count": 0,
            "execution_error": f"Syntax error: {syntax_error}"
        }

    # Execute against endpoint
    success, error, count, results = validate_endpoint(
        state["generated_sparql"],
        timeout=30
    )

    return {
        "execution_result": results,
        "result_count": count or 0,
        "execution_error": error
    }


def verify_results(state: NL2SPARQLState) -> dict[str, Any]:
    """Verify that results match the question semantically."""
    from langchain_core.messages import HumanMessage

    validation_errors = []

    # Check for execution errors
    if state["execution_error"]:
        validation_errors.append(state["execution_error"])

    # Check for empty results
    if state["result_count"] == 0:
        validation_errors.append("Query returned no results - may need different approach")

    # Semantic verification for non-empty results
    if state["result_count"] > 0 and state["execution_result"] and len(validation_errors) == 0:
        # Use fast tier for verification
        llm = get_llm(
            provider=state["provider"],
            model=state["model"],
            tier="fast",
            api_key=state["api_key"],
        )

        sample_results = state["execution_result"][:5]

        verify_prompt = f"""Verify if these SPARQL results correctly answer the question.

Question: {state["question"]}

Sample results (first 5 of {state["result_count"]}):
{json.dumps(sample_results, indent=2, ensure_ascii=False)}

Check:
1. Do the results contain the expected type of data?
2. Are variable names meaningful for the question?
3. Any obvious issues?

Respond with JSON only:
{{"valid": true, "issues": []}}
or
{{"valid": false, "issues": ["issue1", "issue2"]}}"""

        response = llm.invoke([HumanMessage(content=verify_prompt)])

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            verification = json.loads(content.strip())

            if not verification.get("valid", True):
                validation_errors.extend(verification.get("issues", []))
        except (json.JSONDecodeError, IndexError):
            pass  # Skip verification if parsing fails

    is_valid = len(validation_errors) == 0

    # Calculate confidence
    if is_valid:
        confidence = min(1.0, 0.5 + (state["result_count"] / 100) * 0.5)
    else:
        confidence = 0.3

    return {
        "is_valid": is_valid,
        "validation_errors": validation_errors,
        "final_sparql": state["generated_sparql"] if is_valid else "",
        "confidence": confidence,
        "explanation": (
            f"Query returned {state['result_count']} results."
            if is_valid
            else f"Issues: {'; '.join(validation_errors)}"
        )
    }


def refine_query(state: NL2SPARQLState) -> dict[str, Any]:
    """Record the failed attempt and prepare for retry."""

    # Create refinement entry
    refinement_entry = {
        "sparql": state["generated_sparql"],
        "error": "; ".join(state["validation_errors"]),
        "result_count": state["result_count"]
    }

    return {
        "refinement_history": [refinement_entry],
        # Reset for next attempt
        "generated_sparql": "",
        "execution_result": None,
        "execution_error": None,
        "is_valid": False,
        "validation_errors": []
    }


def explore_schema(state: NL2SPARQLState) -> dict[str, Any]:
    """Explore the knowledge base schema for relevant properties."""
    from ..validation.endpoint import validate_endpoint

    # Query for available properties
    schema_queries = [
        # LiLA properties
        """
        SELECT DISTINCT ?property WHERE {
            ?s ?property ?o .
            FILTER(STRSTARTS(STR(?property), "http://lila-erc.eu/ontologies/lila/"))
        } LIMIT 30
        """,
        # OntoLex properties
        """
        SELECT DISTINCT ?property WHERE {
            ?s ?property ?o .
            FILTER(STRSTARTS(STR(?property), "http://www.w3.org/ns/lemon/ontolex#"))
        } LIMIT 30
        """
    ]

    discovered = []

    for query in schema_queries:
        success, error, count, results = validate_endpoint(query.strip(), timeout=15)
        if success and results:
            for r in results:
                prop = r.get("property", "")
                if prop and prop not in discovered:
                    discovered.append(prop)

    return {
        "discovered_properties": discovered[:30],
        "schema_explored": True
    }


def output_result(state: NL2SPARQLState) -> dict[str, Any]:
    """Prepare final output."""

    # If we have a valid result, use it
    if state["is_valid"] and state["final_sparql"]:
        return {
            "explanation": f"Successfully generated query with {state['result_count']} results after {state['generation_attempts']} attempt(s)."
        }

    # If we exhausted attempts, return best effort
    if state["generated_sparql"]:
        return {
            "final_sparql": state["generated_sparql"],
            "confidence": 0.2,
            "explanation": f"Best effort after {state['generation_attempts']} attempts. Issues: {'; '.join(state['validation_errors'])}"
        }

    return {
        "explanation": "Failed to generate a valid query."
    }

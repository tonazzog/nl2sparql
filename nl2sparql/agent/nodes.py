"""Node implementations for the NL2SPARQL LangGraph agent."""

import json
import re
from typing import Any

from .state import NL2SPARQLState


class _TextContentChat:
    """Wraps a LangChain chat model so that response.content is always the answer text.

    Reasoning models return content as a list of chunks (thinking + text); the
    nodes below expect a string, so the thinking chunks are dropped.
    """

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, messages, **kwargs):
        from ..llm.mistral_client import mistral_text

        import time

        # Rate limits (429) and transient 5xx: exponential backoff, up to ~10 minutes in total
        delay = 2.0
        for attempt in range(10):
            try:
                response = self._llm.invoke(messages, **kwargs)
                break
            except Exception as e:  # httpx.HTTPStatusError wrapped by langchain_mistralai
                msg = str(e)
                transient = any(code in msg for code in ("429", "500", "502", "503", "504")) or "Rate limit" in msg
                if not transient or attempt == 9:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        if not isinstance(response.content, str):
            response.content = mistral_text(response.content)
        _log_langchain_call(response, self._llm)
        return response


def _log_langchain_call(response, llm) -> None:
    """Append token usage of a LangChain call to $MISTRAL_LOG (same format as the
    standard client, so the agent's cost can be measured instead of estimated)."""
    import os
    import time

    path = os.environ.get("MISTRAL_LOG")
    if not path:
        return
    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or {}
    details = token_usage.get("completion_tokens_details") or {}
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "requested_model": getattr(llm, "model", None),
        "model_used": meta.get("model_name") or meta.get("model"),
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens") if isinstance(details, dict) else None,
        "finish_reason": meta.get("finish_reason"),
        "duration_s": None,
        "source": "agent",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _parse_json(content: str):
    """Parse the first JSON object/array in an LLM answer (tolerates code fences and
    surrounding prose). Raises ValueError if none can be parsed."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(content):
        if ch in "{[":
            try:
                return decoder.raw_decode(content[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in LLM answer")


# Patterns whose data live in CompL-it (queried through SERVICE): senses and sense
# inventories, definitions, semantic and lexical relations.
SERVICE_PATTERNS = {
    "SENSE_DEFINITION", "SENSE_COUNT", "SEMANTIC_RELATION", "LEXICAL_RELATION", "SERVICE_INTEGRATION",
}


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
        "openai": {"fast": "gpt-6-luna", "default": "gpt-6.1-terra"},
        "anthropic": {"fast": "claude-haiku-4-5-20251001", "default": "claude-sonnet-4-6"},
        "mistral": {"fast": "mistral-large-4-0", "default": "mistral-large-4-0"},
        "gemini": {"fast": "gemini-3.5-flash-lite", "default": "gemini-3.6-flash"},
        "ollama": {"fast": "llama3.1", "default": "llama3.1"},
        "claudecode": {"fast": "claude-haiku-4-5-20251001", "default": "claude-sonnet-4-6"},
    }

    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: {list(DEFAULT_MODELS.keys())}")

    # Use provided model or default based on tier
    if model is None:
        model = DEFAULT_MODELS[provider][tier]

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            from ..llm.openai_client import is_openai_reasoning_model
            kwargs = {"model": model}
            # Reasoning models (gpt-5*, gpt-6*, o-series) accept only the default temperature
            if not is_openai_reasoning_model(model):
                kwargs["temperature"] = temperature
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
            # top_p=1 is required with greedy sampling; reasoning calls can take minutes
            kwargs = {"model": model, "temperature": temperature, "top_p": 1.0, "timeout": 600}
            if api_key:
                kwargs["api_key"] = api_key
            # Reasoning models (e.g. mistral-large-4) return content as a list of chunks
            return _TextContentChat(ChatMistralAI(**kwargs))
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

    elif provider == "claudecode":
        # Claude Code CLI; temperature is not exposed by the CLI and is ignored
        from ..llm.claudecode_client import ClaudeCodeChat
        return ClaudeCodeChat(model=model)


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
   - EMOTION_LEXICON (emotions, feelings, polarity)
   - TRANSLATION (dialect translations - Sicilian, Parmigiano)
   - SENSE_DEFINITION (word definitions from CompL-it)
   - SENSE_COUNT (number of senses, polysemy)
   - SEMANTIC_RELATION (hypernyms, hyponyms, meronyms, part-of relations)
   - LEXICAL_RELATION (synonyms, antonyms)
   - POS_FILTER (part of speech filtering - nouns, verbs, adjectives, etc.)
   - MORPHO_REGEX (word patterns - starts with, ends with, contains)
   - LEXICAL_FORM (inflected forms, conjugations, declensions, gender, number)
   - ETYMOLOGY (word origins, derivations)
   - COUNT_ENTITIES (counting queries)

2. complexity: "simple" (1 pattern), "moderate" (2 patterns), "complex" (3+ patterns or multi-dialect)

3. requires_service: true ONLY if needs CompL-it data (senses, definitions, semantic relations, synonyms)
   - TRUE for: senses and number of senses (polysemy), definitions, hypernyms, hyponyms, meronyms, synonyms, antonyms
   - FALSE for: emotions (ELITA), translations (dialects), POS filters, morphology, Latin

4. requires_translation: true if needs dialect translations

5. dialects: list of dialects needed, e.g., ["sicilian", "parmigiano"]

Respond in JSON format only:
{{"patterns": [...], "complexity": "...", "requires_service": true/false, "requires_translation": true/false, "dialects": [...]}}"""

    response = llm.invoke([
        SystemMessage(content="You are an expert at analyzing linguistic queries for the LiITA knowledge base."),
        HumanMessage(content=analysis_prompt)
    ])

    # Rule-based pattern detection, the same used by the standard pipeline. It is
    # combined with the LLM analysis so that one misclassification by the LLM
    # cannot remove a pattern (and its constraints) from the rest of the run.
    from ..retrieval.patterns import infer_patterns, get_top_patterns

    rule_patterns = infer_patterns(state["question"])
    rule_top = get_top_patterns(rule_patterns, top_k=5)

    try:
        analysis = _parse_json(response.content)
        if not isinstance(analysis, dict):
            raise ValueError("analysis is not a JSON object")
    except ValueError:
        # Fallback: rule-based analysis (not a fixed guess)
        analysis = {
            "patterns": rule_top,
            "complexity": "simple" if len(rule_top) <= 1 else "moderate",
            "requires_service": False,
            "requires_translation": any(p in ("TRANSLATION", "MULTI_TRANSLATION") for p in rule_top),
            "dialects": [],
        }

    llm_patterns = [p for p in analysis.get("patterns", []) if isinstance(p, str)]
    detected = list(dict.fromkeys(llm_patterns + rule_top))
    # SERVICE is suggested if either the LLM or a CompL-it pattern says so
    requires_service = bool(analysis.get("requires_service", False)) or any(
        p in SERVICE_PATTERNS for p in detected
    )

    return {
        "detected_patterns": detected,
        "rule_patterns": rule_patterns,
        "complexity": analysis.get("complexity", "simple"),
        "requires_service": requires_service,
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
        sub_tasks = _parse_json(response.content)
        if not isinstance(sub_tasks, list) or not sub_tasks:
            raise ValueError("plan is not a JSON array")
        sub_tasks = [str(s) for s in sub_tasks]
    except ValueError:
        sub_tasks = [state["question"]]

    return {
        "sub_tasks": sub_tasks,
        "current_task_index": 0
    }


# Retrievers load a sentence-transformer model and embed the whole example/ontology
# set, so they are built once per process and reused (as the MCP server does).
# Re-creating them on every node call leaked hundreds of MB per question.
_RETRIEVERS: dict[str, Any] = {}


def _get_retriever(kind: str):
    if kind not in _RETRIEVERS:
        if kind == "hybrid":
            from ..retrieval.hybrid_retriever import HybridRetriever
            _RETRIEVERS[kind] = HybridRetriever()
        else:
            from ..retrieval.ontology_retriever import OntologyRetriever
            _RETRIEVERS[kind] = OntologyRetriever()
    return _RETRIEVERS[kind]


def retrieve_examples(state: NL2SPARQLState) -> dict[str, Any]:
    """Retrieve relevant examples and constraints."""
    from ..constraints.prompt_builder import get_constraints_for_patterns

    retriever = _get_retriever("hybrid")

    # Retrieve on the question itself, with the rule-based pattern scores, exactly as
    # the standard pipeline does. (Retrieving on the first plan step, as before,
    # matched examples against a paraphrase like "Query CompL-it SERVICE for ...".)
    examples = retriever.retrieve(
        query=state["question"],
        user_patterns=state.get("rule_patterns") or None,
        top_k=5,
    )

    retrieved_examples = [
        {
            "nl": ex.example.nl,
            "sparql": ex.example.sparql,
            "score": ex.score,
            "patterns": dict(ex.example.patterns or {}),
            "nl_variants": dict(ex.example.nl_variants or {}),
        }
        for ex in examples
    ]

    # Merge in examples remembered from past runs (translation memory), if any.
    # Kept in a separate state field (memory_examples) rather than seeding
    # retrieved_examples directly, since this node overwrites that field on
    # every call - merging here is what actually lets a memory hit reach the
    # generation prompt.
    if state.get("memory_examples"):
        retrieved_examples = sorted(
            retrieved_examples + state["memory_examples"],
            key=lambda e: e["score"],
            reverse=True,
        )[:5]

    # Build constraints based on detected patterns
    constraints = get_constraints_for_patterns(state["detected_patterns"])

    return {
        "retrieved_examples": retrieved_examples,
        "relevant_constraints": constraints
    }


def generate_sparql(state: NL2SPARQLState) -> dict[str, Any]:
    """Generate a SPARQL query with the LLM.

    The prompt is the standard pipeline's synthesis prompt (same system prompt,
    constraints, prefixes and few-shot examples), extended with what only the agent
    has: a soft hint about CompL-it, the plan, schema found by exploration and the
    previous failed attempts. The agent therefore differs from the standard
    pipeline only by its execute-verify-refine loop.
    """
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage, SystemMessage

    from ..constraints.prompt_builder import build_synthesis_prompt

    llm = get_llm(
        provider=state["provider"],
        model=state["model"],
        tier="default",
        api_key=state["api_key"],
    )

    exemplars = [
        SimpleNamespace(
            nl=ex["nl"],
            sparql=ex["sparql"],
            patterns=ex.get("patterns") or {},
            nl_variants=ex.get("nl_variants") or {},
        )
        for ex in state["retrieved_examples"][:5]
    ]

    # Schema discovered by ontology exploration (system prompt, as in the standard pipeline)
    schema_context = ""
    if state.get("schema_context"):
        schema_context = state["schema_context"]
    elif state["discovered_properties"]:
        schema_context = "## Discovered Properties:\n" + ", ".join(state["discovered_properties"][:20])

    system_prompt, user_prompt = build_synthesis_prompt(
        user_question=state["question"],
        exemplars=exemplars,
        detected_patterns=state["detected_patterns"],
        schema_context=schema_context,
    )

    # --- agent-specific context, appended after the standard task description ---
    extra = []

    # Soft hint (never a prohibition): the examples and constraints above can override it
    if state["requires_service"]:
        extra.append(
            "## HINT ON DATA SOURCES:\n"
            "This question most likely needs CompL-it data (senses, number of senses, definitions, "
            "semantic or lexical relations). CompL-it is queried with "
            "SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> { ... }."
        )
    else:
        extra.append(
            "## HINT ON DATA SOURCES:\n"
            "This question can probably be answered from the LiITA graphs alone. Use a SERVICE block "
            "for CompL-it only if the examples or constraints above show that it is needed."
        )

    if state["dialects_needed"]:
        extra.append(f"## DIALECTS NEEDED:\n{', '.join(map(str, state['dialects_needed']))}")

    sub_tasks = state.get("sub_tasks") or []
    if len(sub_tasks) > 1:
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(sub_tasks, 1))
        extra.append(f"## SUGGESTED PLAN:\n{steps}")

    own_attempts = state["refinement_history"][-3:]
    if own_attempts:
        lines = ["## PREVIOUS ATTEMPTS (they failed - do not repeat them, fix the cause):"]
        for i, attempt in enumerate(own_attempts, 1):
            lines.append(f"\n### Attempt {i}:")
            lines.append(f"```sparql\n{attempt.get('sparql', 'N/A')}\n```")
            lines.append(f"Problem: {attempt.get('error', 'Unknown')}")
            lines.append(f"Rows returned: {attempt.get('result_count', 0)}")
        extra.append("\n".join(lines))

    if extra:
        user_prompt = user_prompt.rstrip()
        marker = "**SPARQL Query**:"
        if user_prompt.endswith(marker):
            user_prompt = user_prompt[: -len(marker)].rstrip()
        user_prompt += "\n\n" + "\n\n".join(extra) + "\n\n" + marker + "\n"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    sparql = response.content.strip()
    if "```sparql" in sparql:
        sparql = sparql.split("```sparql")[1].split("```")[0].strip()
    elif "```" in sparql:
        sparql = sparql.split("```")[1].split("```")[0].strip()

    return {
        "generated_sparql": sparql,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
    }


def _check_variable_reuse(sparql: str) -> str | None:
    """Check for common variable reuse bugs in SPARQL.

    Detects when a variable is used both as a URI (subject/object position)
    and as a literal (in writtenRep). This is a common LLM mistake.

    Returns error message if bug found, None otherwise.
    """
    # Find variables used in writtenRep (these should be literals)
    literal_vars = set(re.findall(r'writtenRep\s+(\?\w+)', sparql))

    # Find variables used as subjects (at start of triple pattern)
    # Pattern: newline/brace followed by ?var followed by space/predicate
    subject_vars = set(re.findall(r'(?:^|\{|\.\s*)\s*(\?\w+)\s+(?:a\s|ontolex:|lexinfo:|skos:|lila:|rdf:|rdfs:)', sparql, re.MULTILINE))

    # Find overlap - variables used both as literal and as subject/URI
    reused_vars = literal_vars & subject_vars

    if reused_vars:
        return f"Variable reuse error: {', '.join(reused_vars)} used both as URI and literal. Use different variable names."

    return None


def _fix_case_sensitive_filters(sparql: str) -> tuple[str, bool]:
    """Fix case-sensitive string equality filters to case-insensitive regex.

    Converts patterns like:
        FILTER(STR(?var) = "string")
        FILTER(?var = "string")

    To case-insensitive regex:
        FILTER(REGEX(STR(?var), "^string$", "i"))

    This is a targeted fix that doesn't change query structure.
    Used when a query returns 0 results due to case mismatch (e.g., "rabbia" vs "Rabbia").

    Returns:
        tuple: (fixed_sparql, was_modified)
    """
    modified = False

    # Pattern 1: FILTER(STR(?var) = "string")
    pattern1 = r'FILTER\s*\(\s*STR\s*\(\s*(\?\w+)\s*\)\s*=\s*"([^"]+)"\s*\)'

    def replace1(match):
        nonlocal modified
        modified = True
        var = match.group(1)
        value = match.group(2)
        # Escape regex special characters in the value
        escaped_value = re.escape(value)
        return f'FILTER(REGEX(STR({var}), "^{escaped_value}$", "i"))'

    sparql = re.sub(pattern1, replace1, sparql)

    # Pattern 2: FILTER(?var = "string") - direct variable comparison
    pattern2 = r'FILTER\s*\(\s*(\?\w+)\s*=\s*"([^"]+)"\s*\)'

    def replace2(match):
        nonlocal modified
        modified = True
        var = match.group(1)
        value = match.group(2)
        escaped_value = re.escape(value)
        return f'FILTER(REGEX(STR({var}), "^{escaped_value}$", "i"))'

    sparql = re.sub(pattern2, replace2, sparql)

    return sparql, modified


def is_service_error(error_message: str) -> bool:
    """True if the endpoint refused a federated SERVICE call (a permission problem).

    Only permission-type errors are matched. Timeouts and 5xx are transient and are
    retried as they are (see _is_transient_error); query-shape errors that merely
    mention "service" (e.g. Virtuoso SP031) are left to the refinement loop.
    """
    if not error_message:
        return False
    error_lower = error_message.lower()
    indicators = [
        "sr186",          # Virtuoso: No permission to execute procedure (e.g. SPARQL_SD_PROBE)
        "sr619",          # Virtuoso: SPARUL LOAD SERVICE DATA access denied
        "load service data",
        "access denied",
        "no permission",
        "permission denied",
    ]
    return any(indicator in error_lower for indicator in indicators)


def _is_transient_error(error_message: str) -> bool:
    """True for errors worth retrying unchanged (timeouts, gateway errors, connection drops)."""
    if not error_message:
        return False
    error_lower = error_message.lower()
    return any(s in error_lower for s in (
        "timeout", "timed out", "502", "503", "504", "bad gateway", "service unavailable",
        "connection error", "connection reset", "connection aborted", "temporarily",
    ))


_AGGREGATE_RE = re.compile(r"\b(COUNT|SUM)\s*\(", re.IGNORECASE)


def _effective_count(sparql: str, count: int | None, sample: list | None) -> int:
    """Row count, except that a single row whose numeric values are all 0 from a
    COUNT/SUM query counts as an empty answer (a count of 0 is still one row)."""
    count = count or 0
    if count == 1 and sample and _AGGREGATE_RE.search(sparql):
        numeric = []
        for value in sample[0].values():
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                continue
        if numeric and all(v == 0 for v in numeric):
            return 0
    return count


def _run_on_endpoint(sparql: str):
    """Execute on the endpoint (GET for normal-size queries, like the evaluator),
    retrying transient failures twice."""
    import time

    from ..validation.endpoint import validate_endpoint

    success, error, count, results = validate_endpoint(sparql, timeout=30, method="auto")
    for delay in (5, 15):
        if not (error and _is_transient_error(error)):
            break
        time.sleep(delay)
        success, error, count, results = validate_endpoint(sparql, timeout=30, method="auto")
    return success, error, count, results


def execute_query(state: NL2SPARQLState) -> dict[str, Any]:
    """Execute the SPARQL query against the endpoint."""
    from ..validation.syntax import validate_syntax

    sparql = state["generated_sparql"]

    # Check for variable reuse bug (same var for URI and literal)
    reuse_error = _check_variable_reuse(sparql)
    if reuse_error:
        return {
            "execution_result": None,
            "result_count": 0,
            "execution_error": reuse_error
        }

    # First check syntax
    syntax_valid, syntax_error = validate_syntax(sparql)

    if not syntax_valid:
        return {
            "execution_result": None,
            "result_count": 0,
            "execution_error": f"Syntax error: {syntax_error}"
        }

    success, error, count, results = _run_on_endpoint(sparql)

    # The endpoint refused the federated call. Removing the SERVICE block would only
    # query LiITA, which does not contain CompL-it's senses, definitions and
    # relations, so such a result is never accepted: the error goes back to the
    # refinement loop instead.
    if error and is_service_error(error):
        return {
            "execution_result": None,
            "result_count": 0,
            "execution_error": (
                f"Endpoint error: the federated SERVICE call to CompL-it was refused ({error[:300]}). "
                "Do not drop the SERVICE block (CompL-it data are not in LiITA); restructure the query instead."
            ),
        }

    count = _effective_count(sparql, count, results) if success and not error else count

    # If no results and no error, try fixing case-sensitive filters
    # This handles cases like "rabbia" vs "Rabbia" without regenerating the whole query
    if success and not error and not count:
        fixed_sparql, was_fixed = _fix_case_sensitive_filters(sparql)
        if was_fixed:
            success2, error2, count2, results2 = _run_on_endpoint(fixed_sparql)
            count2 = _effective_count(fixed_sparql, count2, results2) if success2 and not error2 else 0
            if success2 and count2 > 0:
                return {
                    "execution_result": results2,
                    "result_count": count2,
                    "execution_error": None,
                    "generated_sparql": fixed_sparql,
                }

    return {
        "execution_result": results,
        "result_count": count or 0,
        "execution_error": error
    }


def verify_results(state: NL2SPARQLState) -> dict[str, Any]:
    """Verify query execution results.

    Checks for:
    1. Technical errors (syntax, execution) - triggers refine
    2. Zero results after minor fixes - triggers explore/refine

    Does NOT perform semantic verification of results because LLMs lack
    domain knowledge and over-aggressive verification introduces bugs.
    """
    validation_errors = []
    has_technical_error = False

    # Check for actual execution errors (technical failures)
    if state["execution_error"]:
        error_msg = state["execution_error"].lower()
        if any(err in error_msg for err in ["syntax", "parse", "timeout", "connection", "500", "error"]):
            validation_errors.append(state["execution_error"])
            has_technical_error = True

    # Query is syntactically valid if no technical errors
    syntax_valid = not has_technical_error

    # Store first syntactically valid query as fallback
    # (only on first attempt, when first_valid_sparql is empty)
    first_valid_sparql = state.get("first_valid_sparql", "")
    if syntax_valid and not first_valid_sparql and state["generated_sparql"]:
        first_valid_sparql = state["generated_sparql"]

    # Zero results (after minor fixes in execute_query) should trigger explore/refine
    # But we still consider the query "syntactically valid" for fallback purposes
    if state["result_count"] == 0 and not has_technical_error:
        validation_errors.append("Query returned no results (or only a zero count) - will try explore/refine")

    is_valid = len(validation_errors) == 0

    # Calculate confidence
    if is_valid:
        confidence = min(1.0, 0.5 + (state["result_count"] / 100) * 0.5)
    elif syntax_valid:
        confidence = 0.4  # Syntax OK but no results
    else:
        confidence = 0.2  # Technical error

    return {
        "is_valid": is_valid,
        "validation_errors": validation_errors,
        "final_sparql": state["generated_sparql"] if is_valid else "",
        "confidence": confidence,
        "first_valid_sparql": first_valid_sparql,
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
    """Explore ontology to find relevant classes and properties for the query.

    Uses semantic search over ontology definitions to discover appropriate
    vocabulary based on the user's question. This is more effective than
    querying the endpoint for property URIs because it provides:
    - Property/class descriptions (what they mean)
    - Domain/range information (how to use them)
    - SPARQL usage examples
    """
    try:
        retriever = _get_retriever("ontology")

        # Retrieve properties and classes relevant to the question
        # Focus on properties since they're most important for query construction
        properties = retriever.retrieve_properties(state["question"], top_k=8)
        classes = retriever.retrieve_classes(state["question"], top_k=4)

        # Combine and format for the prompt
        all_entries = properties + classes
        schema_context = retriever.format_for_prompt(all_entries, include_examples=True)

        # Also store URIs for backward compatibility
        discovered_uris = [item.entry.uri for item in all_entries]

        return {
            "discovered_properties": discovered_uris,
            "schema_context": schema_context,  # Detailed descriptions for prompt
            "schema_explored": True
        }

    except Exception as e:
        # Fallback if ontology retrieval fails
        return {
            "discovered_properties": [],
            "schema_context": f"Schema exploration failed: {str(e)}",
            "schema_explored": True
        }


def output_result(state: NL2SPARQLState) -> dict[str, Any]:
    """Prepare final output.

    Priority:
    1. Valid query with results → use it
    2. Exhausted attempts → use first syntactically valid query (fallback)
    3. No valid query → return error
    """

    # If we have a valid result with results, use it
    if state["is_valid"] and state["final_sparql"]:
        return {
            "explanation": f"Successfully generated query with {state['result_count']} results after {state['generation_attempts']} attempt(s)."
        }

    # If we exhausted attempts, prefer first syntactically valid query as fallback
    # This is better than later attempts which may have introduced errors
    first_valid = state.get("first_valid_sparql", "")
    if first_valid:
        return {
            "final_sparql": first_valid,
            "confidence": 0.3,
            "explanation": f"Returning first syntactically valid query after {state['generation_attempts']} attempts. Refinement did not improve results."
        }

    # Fallback to last generated query if no first_valid
    if state["generated_sparql"]:
        return {
            "final_sparql": state["generated_sparql"],
            "confidence": 0.2,
            "explanation": f"Best effort after {state['generation_attempts']} attempts. Issues: {'; '.join(state['validation_errors'])}"
        }

    return {
        "explanation": "Failed to generate a valid query."
    }

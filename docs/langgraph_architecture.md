# LangGraph Agentic Architecture for NL2SPARQL

This document outlines a proposed agentic architecture using LangGraph to improve the NL-to-SPARQL translation system.

## Current vs Proposed Architecture

### Current Architecture (Linear)
```
Question → Pattern Detection → Retrieval → LLM Generation → Validation → Output
                                                              ↓
                                                         (retry if invalid)
```

### Proposed Architecture (Agentic)
```
                                    ┌─────────────────────────────────────┐
                                    │                                     │
Question → Analyze → Plan → Generate → Execute → Verify ──→ Output       │
              ↑         ↑       ↑         │         │                     │
              │         │       │         │         │                     │
              │         │       │    ┌────┴────┐    │                     │
              │         │       │    │ Results │    │                     │
              │         │       │    │ Empty?  │    │                     │
              │         │       │    └────┬────┘    │                     │
              │         │       │         │Yes      │No match             │
              │         │       └─────────┘         │                     │
              │         │                           │                     │
              │         └───────────────────────────┘                     │
              │                    Refine                                 │
              └───────────────────────────────────────────────────────────┘
                                 Schema Explore
```

## State Definition

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from operator import add

class NL2SPARQLState(TypedDict):
    # Input
    question: str
    language: str  # "it" or "en"

    # Analysis
    detected_patterns: list[str]
    complexity: Literal["simple", "moderate", "complex"]
    requires_service: bool
    requires_translation: bool
    dialects_needed: list[str]  # ["sicilian", "parmigiano"]

    # Planning
    sub_tasks: list[str]
    current_task_index: int

    # Retrieval
    retrieved_examples: list[dict]
    relevant_constraints: str

    # Generation
    generated_sparql: str
    generation_attempts: int

    # Execution
    execution_result: dict | None
    result_count: int
    execution_error: str | None

    # Verification
    is_valid: bool
    validation_errors: list[str]

    # Refinement
    refinement_history: Annotated[list[dict], add]  # Accumulates

    # Schema exploration
    discovered_properties: list[str]
    discovered_classes: list[str]

    # Output
    final_sparql: str
    confidence: float
    explanation: str
```

## Node Definitions

### 1. Analyze Node
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

def analyze_question(state: NL2SPARQLState) -> dict:
    """Analyze the question to understand what's needed."""

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    analysis_prompt = f"""Analyze this natural language question for SPARQL translation:

Question: {state["question"]}

Determine:
1. What patterns are needed (EMOTION, TRANSLATION, SEMANTIC_RELATION, etc.)
2. Complexity level (simple/moderate/complex)
3. Does it need CompL-it SERVICE block?
4. Does it need dialect translations? Which ones?
5. Are there multiple sub-questions to answer?

Respond in JSON format."""

    response = llm.invoke([
        SystemMessage(content="You are an expert at analyzing linguistic queries."),
        HumanMessage(content=analysis_prompt)
    ])

    analysis = json.loads(response.content)

    return {
        "detected_patterns": analysis["patterns"],
        "complexity": analysis["complexity"],
        "requires_service": analysis["requires_service"],
        "requires_translation": analysis["requires_translation"],
        "dialects_needed": analysis.get("dialects", []),
    }
```

### 2. Plan Node (for complex queries)
```python
def plan_query(state: NL2SPARQLState) -> dict:
    """Break complex queries into sub-tasks."""

    if state["complexity"] == "simple":
        return {"sub_tasks": [state["question"]], "current_task_index": 0}

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    planning_prompt = f"""Break this complex query into simpler sub-tasks:

Question: {state["question"]}
Patterns needed: {state["detected_patterns"]}
Dialects: {state["dialects_needed"]}

Create a step-by-step plan where each step produces a SPARQL fragment.
The steps should be ordered so later steps can reference earlier results.

Example:
1. Find Italian adjectives ending with -oso
2. Get definitions from CompL-it
3. Find Sicilian translations
4. Find Parmigiano translations
5. Combine all results

Return as JSON list of task descriptions."""

    response = llm.invoke([HumanMessage(content=planning_prompt)])
    sub_tasks = json.loads(response.content)

    return {
        "sub_tasks": sub_tasks,
        "current_task_index": 0
    }
```

### 3. Retrieve Node
```python
def retrieve_examples(state: NL2SPARQLState) -> dict:
    """Retrieve relevant examples and constraints."""

    from nl2sparql.retrieval.hybrid_retriever import HybridRetriever
    from nl2sparql.constraints.prompt_builder import build_constraints

    retriever = HybridRetriever()

    # Get current sub-task or full question
    current_query = (
        state["sub_tasks"][state["current_task_index"]]
        if state["sub_tasks"]
        else state["question"]
    )

    # Retrieve examples
    examples = retriever.retrieve(current_query, top_k=5)

    # Build constraints based on detected patterns
    constraints = build_constraints(state["detected_patterns"])

    return {
        "retrieved_examples": [
            {"nl": ex.example.nl, "sparql": ex.example.sparql, "score": ex.score}
            for ex in examples
        ],
        "relevant_constraints": constraints
    }
```

### 4. Generate Node
```python
def generate_sparql(state: NL2SPARQLState) -> dict:
    """Generate SPARQL query using LLM."""

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # Include refinement history if available
    refinement_context = ""
    if state["refinement_history"]:
        refinement_context = "\n\nPrevious attempts and issues:\n"
        for attempt in state["refinement_history"][-3:]:  # Last 3 attempts
            refinement_context += f"- Query: {attempt['sparql'][:200]}...\n"
            refinement_context += f"  Error: {attempt['error']}\n"

    examples_text = "\n\n".join([
        f"Question: {ex['nl']}\nSPARQL:\n{ex['sparql']}"
        for ex in state["retrieved_examples"][:3]
    ])

    generation_prompt = f"""Generate a SPARQL query for the LiITA knowledge base.

## Constraints
{state["relevant_constraints"]}

## Similar Examples
{examples_text}

## Question
{state["question"]}
{refinement_context}

Generate ONLY the SPARQL query, no explanations."""

    response = llm.invoke([HumanMessage(content=generation_prompt)])

    return {
        "generated_sparql": response.content,
        "generation_attempts": state.get("generation_attempts", 0) + 1
    }
```

### 5. Execute Node
```python
def execute_query(state: NL2SPARQLState) -> dict:
    """Execute the SPARQL query against the endpoint."""

    from nl2sparql.validation.endpoint import validate_endpoint

    success, error, count, results = validate_endpoint(
        state["generated_sparql"],
        timeout=30
    )

    return {
        "execution_result": results,
        "result_count": count or 0,
        "execution_error": error
    }
```

### 6. Verify Node
```python
def verify_results(state: NL2SPARQLState) -> dict:
    """Verify that results match the question semantically."""

    # Check for basic validity
    validation_errors = []

    if state["execution_error"]:
        validation_errors.append(f"Execution error: {state['execution_error']}")

    if state["result_count"] == 0:
        validation_errors.append("Query returned no results")

    # Semantic verification using LLM
    if state["result_count"] > 0 and state["execution_result"]:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        sample_results = state["execution_result"][:5]

        verify_prompt = f"""Verify if these SPARQL results answer the question:

Question: {state["question"]}
Results sample: {json.dumps(sample_results, indent=2)}

Do these results make sense? Are they what the user asked for?
Respond with JSON: {{"valid": true/false, "issues": ["list of issues if any"]}}"""

        response = llm.invoke([HumanMessage(content=verify_prompt)])
        verification = json.loads(response.content)

        if not verification["valid"]:
            validation_errors.extend(verification["issues"])

    is_valid = len(validation_errors) == 0

    return {
        "is_valid": is_valid,
        "validation_errors": validation_errors,
        "final_sparql": state["generated_sparql"] if is_valid else "",
        "confidence": 1.0 if is_valid else 0.5
    }
```

### 7. Refine Node
```python
def refine_query(state: NL2SPARQLState) -> dict:
    """Refine the query based on errors."""

    # Add to refinement history
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
        "is_valid": False
    }
```

### 8. Explore Schema Node (optional)
```python
def explore_schema(state: NL2SPARQLState) -> dict:
    """Explore the knowledge base schema for relevant properties."""

    from nl2sparql.validation.endpoint import validate_endpoint

    # Query for available properties related to detected patterns
    schema_query = """
    SELECT DISTINCT ?property (COUNT(?s) as ?usage)
    WHERE {
        ?s ?property ?o .
        FILTER(STRSTARTS(STR(?property), "http://lila-erc.eu/ontologies/lila/"))
    }
    GROUP BY ?property
    ORDER BY DESC(?usage)
    LIMIT 50
    """

    success, error, count, results = validate_endpoint(schema_query)

    properties = [r["property"] for r in (results or [])]

    return {
        "discovered_properties": properties
    }
```

## Graph Definition

```python
from langgraph.graph import StateGraph, END

def should_refine(state: NL2SPARQLState) -> str:
    """Decide whether to refine or finish."""

    # Success - output the result
    if state["is_valid"]:
        return "output"

    # Too many attempts - give up
    if state["generation_attempts"] >= 3:
        return "output"

    # Empty results might need schema exploration
    if state["result_count"] == 0 and not state.get("discovered_properties"):
        return "explore"

    # Otherwise, refine
    return "refine"


def build_graph() -> StateGraph:
    """Build the LangGraph workflow."""

    workflow = StateGraph(NL2SPARQLState)

    # Add nodes
    workflow.add_node("analyze", analyze_question)
    workflow.add_node("plan", plan_query)
    workflow.add_node("retrieve", retrieve_examples)
    workflow.add_node("generate", generate_sparql)
    workflow.add_node("execute", execute_query)
    workflow.add_node("verify", verify_results)
    workflow.add_node("refine", refine_query)
    workflow.add_node("explore", explore_schema)

    # Define edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", "verify")

    # Conditional edges after verification
    workflow.add_conditional_edges(
        "verify",
        should_refine,
        {
            "output": END,
            "refine": "refine",
            "explore": "explore"
        }
    )

    # After refinement, go back to retrieve (might get different examples)
    workflow.add_edge("refine", "retrieve")

    # After exploration, go back to generate with new schema knowledge
    workflow.add_edge("explore", "generate")

    return workflow.compile()
```

## Usage

```python
async def translate(question: str, language: str = "it") -> dict:
    """Translate natural language to SPARQL using the agent."""

    graph = build_graph()

    initial_state = {
        "question": question,
        "language": language,
        "detected_patterns": [],
        "complexity": "simple",
        "requires_service": False,
        "requires_translation": False,
        "dialects_needed": [],
        "sub_tasks": [],
        "current_task_index": 0,
        "retrieved_examples": [],
        "relevant_constraints": "",
        "generated_sparql": "",
        "generation_attempts": 0,
        "execution_result": None,
        "result_count": 0,
        "execution_error": None,
        "is_valid": False,
        "validation_errors": [],
        "refinement_history": [],
        "discovered_properties": [],
        "discovered_classes": [],
        "final_sparql": "",
        "confidence": 0.0,
        "explanation": ""
    }

    # Run the graph
    final_state = await graph.ainvoke(initial_state)

    return {
        "sparql": final_state["final_sparql"],
        "confidence": final_state["confidence"],
        "attempts": final_state["generation_attempts"],
        "result_count": final_state["result_count"],
        "refinement_history": final_state["refinement_history"]
    }
```

## Visualization

```python
from IPython.display import Image, display

graph = build_graph()
display(Image(graph.get_graph().draw_mermaid_png()))
```

## Key Improvements Over Current System

| Aspect | Current | LangGraph Agent |
|--------|---------|-----------------|
| Error handling | Retry with same approach | Analyze error, adapt strategy |
| Empty results | Return empty | Explore schema, try alternatives |
| Complex queries | Single-shot | Decompose into sub-tasks |
| Validation | Syntax only | Semantic verification with LLM |
| Learning | None | Accumulates refinement history |
| Schema awareness | Static constraints | Dynamic schema exploration |

## Future Enhancements

1. **Checkpointing**: Save state for long-running queries
2. **Human-in-the-loop**: Ask user for clarification at decision points
3. **Multi-agent**: Specialized agents for each pattern type
4. **Caching**: Cache successful query patterns for reuse
5. **Feedback learning**: Store successful translations for retrieval

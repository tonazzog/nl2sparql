"""LangGraph workflow definition for the NL2SPARQL agent."""

from typing import Literal

from .state import NL2SPARQLState, create_initial_state
from .nodes import (
    analyze_question,
    plan_query,
    retrieve_examples,
    generate_sparql,
    execute_query,
    verify_results,
    refine_query,
    explore_schema,
    output_result,
)

# Maximum generation attempts before giving up
MAX_ATTEMPTS = 3


def should_refine(state: NL2SPARQLState) -> Literal["output", "refine", "explore"]:
    """Decide whether to refine, explore schema, or output result."""

    # Success - output the result
    if state["is_valid"]:
        return "output"

    # Too many attempts - give up and output best effort
    if state["generation_attempts"] >= MAX_ATTEMPTS:
        return "output"

    # Empty results and haven't explored schema yet - try exploration
    if state["result_count"] == 0 and not state["schema_explored"]:
        return "explore"

    # Otherwise, refine the query
    return "refine"


def build_graph():
    """Build the LangGraph workflow for NL2SPARQL translation."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError(
            "langgraph is required. Install with: pip install langgraph"
        )

    # Create the graph
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
    workflow.add_node("output", output_result)

    # Define the flow
    workflow.set_entry_point("analyze")

    # Linear flow: analyze -> plan -> retrieve -> generate -> execute -> verify
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
            "output": "output",
            "refine": "refine",
            "explore": "explore"
        }
    )

    # After refinement, go back to retrieve (might get different examples with new context)
    workflow.add_edge("refine", "retrieve")

    # After exploration, go back to generate with new schema knowledge
    workflow.add_edge("explore", "generate")

    # Output is the end
    workflow.add_edge("output", END)

    # Compile the graph
    return workflow.compile()


def get_graph_visualization():
    """Get a Mermaid diagram of the graph."""
    graph = build_graph()
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        return None


class NL2SPARQLAgent:
    """Agent for translating natural language to SPARQL using LangGraph."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the agent.

        Args:
            provider: LLM provider ("openai", "anthropic", "mistral", "gemini", "ollama")
            model: Model name (uses provider default if None)
            api_key: API key (uses environment variable if None)
        """
        self.graph = build_graph()
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def translate(
        self,
        question: str,
        language: str = "it",
        verbose: bool = False
    ) -> dict:
        """
        Translate a natural language question to SPARQL.

        Args:
            question: The natural language question
            language: Language code ("it" or "en")
            verbose: If True, print progress

        Returns:
            Dictionary with:
                - sparql: The generated SPARQL query
                - confidence: Confidence score (0-1)
                - attempts: Number of generation attempts
                - result_count: Number of results from execution
                - explanation: Human-readable explanation
                - refinement_history: List of previous attempts
        """
        initial_state = create_initial_state(
            question, language,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )

        if verbose:
            print(f"Translating: {question}")
            print("-" * 50)

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        if verbose:
            print(f"Attempts: {final_state['generation_attempts']}")
            print(f"Valid: {final_state['is_valid']}")
            print(f"Results: {final_state['result_count']}")
            if final_state["validation_errors"]:
                print(f"Errors: {final_state['validation_errors']}")

        return {
            "sparql": final_state["final_sparql"] or final_state["generated_sparql"],
            "confidence": final_state["confidence"],
            "attempts": final_state["generation_attempts"],
            "result_count": final_state["result_count"],
            "explanation": final_state["explanation"],
            "detected_patterns": final_state["detected_patterns"],
            "refinement_history": final_state["refinement_history"],
            "is_valid": final_state["is_valid"],
        }

    async def atranslate(
        self,
        question: str,
        language: str = "it",
        verbose: bool = False
    ) -> dict:
        """Async version of translate."""
        initial_state = create_initial_state(
            question, language,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )

        if verbose:
            print(f"Translating: {question}")
            print("-" * 50)

        # Run the graph asynchronously
        final_state = await self.graph.ainvoke(initial_state)

        if verbose:
            print(f"Attempts: {final_state['generation_attempts']}")
            print(f"Valid: {final_state['is_valid']}")
            print(f"Results: {final_state['result_count']}")

        return {
            "sparql": final_state["final_sparql"] or final_state["generated_sparql"],
            "confidence": final_state["confidence"],
            "attempts": final_state["generation_attempts"],
            "result_count": final_state["result_count"],
            "explanation": final_state["explanation"],
            "detected_patterns": final_state["detected_patterns"],
            "refinement_history": final_state["refinement_history"],
            "is_valid": final_state["is_valid"],
        }

    def stream(
        self,
        question: str,
        language: str = "it"
    ):
        """
        Stream the translation process, yielding state after each node.

        Args:
            question: The natural language question
            language: Language code

        Yields:
            Tuple of (node_name, accumulated_state) after each step.
            The accumulated_state contains all state updates up to that point.
        """
        initial_state = create_initial_state(
            question, language,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )

        # Track accumulated state since stream() yields partial updates
        accumulated_state = dict(initial_state)

        for output in self.graph.stream(initial_state):
            for node_name, node_update in output.items():
                # Merge node update into accumulated state
                accumulated_state.update(node_update)
                yield node_name, accumulated_state

    def get_final_result(self, accumulated_state: dict) -> dict:
        """
        Extract the final result from an accumulated state.

        Args:
            accumulated_state: The accumulated state from stream()

        Returns:
            Dictionary with the final result (same format as translate())
        """
        return {
            "sparql": accumulated_state.get("final_sparql") or accumulated_state.get("generated_sparql"),
            "confidence": accumulated_state.get("confidence", 0.0),
            "attempts": accumulated_state.get("generation_attempts", 0),
            "result_count": accumulated_state.get("result_count", 0),
            "explanation": accumulated_state.get("explanation", ""),
            "detected_patterns": accumulated_state.get("detected_patterns", []),
            "refinement_history": accumulated_state.get("refinement_history", []),
            "is_valid": accumulated_state.get("is_valid", False),
        }

"""Synthetic data generation for fine-tuning LLMs on NL2SPARQL."""

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

from ..config import DATASET_PATH, AVAILABLE_PROVIDERS


@dataclass
class SyntheticPair:
    """A synthetic (NL, SPARQL) training pair."""

    nl_question: str
    sparql_query: str
    patterns: list[str] = field(default_factory=list)
    source_example_id: Optional[str] = None
    result_count: int = 0
    is_valid: bool = False
    language: str = "it"
    generation_method: str = "variation"  # variation, combination, template


@dataclass
class GenerationStats:
    """Statistics from synthetic data generation."""

    total_attempts: int = 0
    successful_pairs: int = 0
    failed_validation: int = 0
    failed_generation: int = 0
    total_time: float = 0.0
    pairs_by_pattern: dict = field(default_factory=dict)


def generate_nl_variations(
    seed_question: str,
    seed_sparql: str,
    num_variations: int,
    llm,
    patterns: list[str] = None,
) -> list[str]:
    """
    Generate NL variations of a seed question using an LLM.

    Strategies applied:
    1. Paraphrase the question (different wording, same meaning)
    2. Change entities (casa → libro, cane → gatto)
    3. Change language style (formal/informal)
    4. Change question structure (interrogative vs imperative)

    Args:
        seed_question: Original NL question
        seed_sparql: Original SPARQL (for context)
        num_variations: Number of variations to generate
        llm: LangChain LLM instance
        patterns: Query patterns for context

    Returns:
        List of NL question variations
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    pattern_context = f"\nQuery patterns: {patterns}" if patterns else ""

    system_msg = """You are an expert at generating natural language variations for SPARQL queries.
Your task is to create diverse Italian questions that would translate to similar SPARQL patterns.
Generate grammatically correct, natural-sounding Italian questions."""

    prompt = f"""Generate {num_variations} variations of this Italian natural language question.

Original question: {seed_question}
Original SPARQL (for context):
```sparql
{seed_sparql[:800]}
```{pattern_context}

Requirements:
1. Each variation should ask a semantically similar question
2. Use different entities where appropriate (e.g., if asking about "casa", try "libro", "acqua", etc.)
3. Vary the phrasing style (formal/informal, different verbs)
4. Keep questions grammatically correct in Italian
5. Ensure questions would require similar SPARQL patterns to answer

Return ONLY a JSON array of {num_variations} question strings, no explanations.
Example format: ["Quali lemmi esprimono gioia?", "Trova le parole che indicano felicità"]"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt)
        ])

        content = response.content.strip()

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        variations = json.loads(content.strip())

        if isinstance(variations, list):
            return [v for v in variations if isinstance(v, str) and len(v) > 10]
        return []

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Warning: Failed to parse variations: {e}")
        return []


def generate_combined_question(
    examples: list[dict],
    llm,
) -> tuple[str, list[str]]:
    """
    Generate a question that combines patterns from multiple examples.

    Args:
        examples: List of seed examples to combine
        llm: LangChain LLM instance

    Returns:
        Tuple of (combined_question, combined_patterns)
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    examples_text = "\n\n".join([
        f"Question: {ex.get('nl', ex.get('question', ''))}\nPatterns: {ex.get('patterns', [])}"
        for ex in examples[:3]
    ])

    prompt = f"""Combine these query patterns into a single, more complex Italian question:

{examples_text}

Create ONE Italian question that requires information from multiple patterns.
For example, combining EMOTION + TRANSLATION might ask:
"Quali sono le traduzioni siciliane delle parole che esprimono tristezza?"

Return JSON: {{"question": "...", "patterns": [...]}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        return result.get("question", ""), result.get("patterns", [])

    except Exception:
        return "", []


class SyntheticDataGenerator:
    """
    Generate synthetic training pairs for fine-tuning LLMs.

    This generator uses the existing NL2SPARQL infrastructure to create
    validated (NL question, SPARQL query) pairs from seed examples.

    Usage:
        generator = SyntheticDataGenerator(provider="openai", model="gpt-4.1-mini")
        pairs = generator.generate_dataset(num_variations_per_seed=5)
        generator.save_dataset(pairs, "synthetic_data.jsonl")
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        use_agent: bool = True,
        seed_dataset_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            provider: LLM provider (openai, anthropic, mistral, etc.)
            model: Model name (uses provider default if None)
            use_agent: Use NL2SPARQLAgent (True) or NL2SPARQL translator (False)
            seed_dataset_path: Path to seed examples (uses default if None)
            api_key: API key (uses environment variable if None)
        """
        self.provider = provider
        self.model = model or AVAILABLE_PROVIDERS.get(provider, {}).get("default_model")
        self.use_agent = use_agent
        self.api_key = api_key

        # Load seed examples
        dataset_path = Path(seed_dataset_path) if seed_dataset_path else DATASET_PATH
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.seed_examples = json.load(f)

        self._llm = None
        self._translator = None

    @property
    def llm(self):
        """Lazy-load LLM for NL generation."""
        if self._llm is None:
            from ..agent.nodes import get_llm
            self._llm = get_llm(
                provider=self.provider,
                model=self.model,
                tier="default",
                api_key=self.api_key,
            )
        return self._llm

    @property
    def translator(self):
        """Lazy-load translator/agent for SPARQL generation."""
        if self._translator is None:
            if self.use_agent:
                from ..agent import NL2SPARQLAgent
                self._translator = NL2SPARQLAgent(
                    provider=self.provider,
                    model=self.model,
                    api_key=self.api_key,
                )
            else:
                from ..generation.synthesizer import NL2SPARQL
                self._translator = NL2SPARQL(
                    provider=self.provider,
                    model=self.model,
                    validate=True,
                    fix_errors=True,
                )
        return self._translator

    def generate_sparql_for_question(
        self,
        question: str,
        language: str = "it",
    ) -> tuple[str, bool, int, list[str]]:
        """
        Generate SPARQL for a question and validate it.

        Args:
            question: Natural language question
            language: Language code

        Returns:
            Tuple of (sparql, is_valid, result_count, detected_patterns)
        """
        try:
            if self.use_agent:
                result = self.translator.translate(question, language=language)
                return (
                    result.get("sparql", ""),
                    result.get("is_valid", False),
                    result.get("result_count", 0),
                    result.get("detected_patterns", []),
                )
            else:
                result = self.translator.translate(question)
                is_valid = (
                    result.validation.syntax_valid
                    and result.validation.execution_success
                )
                return (
                    result.sparql,
                    is_valid,
                    result.validation.result_count or 0,
                    result.detected_patterns,
                )
        except Exception as e:
            print(f"Warning: SPARQL generation failed: {e}")
            return "", False, 0, []

    def generate_variations_for_seed(
        self,
        seed: dict,
        num_variations: int = 5,
        min_results: int = 1,
        language: str = "it",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> list[SyntheticPair]:
        """
        Generate synthetic pairs from a single seed example.

        Args:
            seed: Seed example dictionary
            num_variations: Number of NL variations to generate
            min_results: Minimum results required for valid pair
            language: Language code
            progress_callback: Optional callback for progress updates

        Returns:
            List of validated SyntheticPair
        """
        pairs = []

        # Extract seed data
        seed_nl = seed.get("nl") or seed.get("question", "")
        seed_sparql = seed.get("sparql", "")
        seed_id = seed.get("id")
        patterns = seed.get("patterns", [])

        if not seed_nl or not seed_sparql:
            return pairs

        # Generate NL variations
        if progress_callback:
            progress_callback(f"Generating variations for: {seed_nl[:50]}...")

        variations = generate_nl_variations(
            seed_question=seed_nl,
            seed_sparql=seed_sparql,
            num_variations=num_variations,
            llm=self.llm,
            patterns=patterns,
        )

        # Generate and validate SPARQL for each variation
        for variation in variations:
            if progress_callback:
                progress_callback(f"  Processing: {variation[:40]}...")

            sparql, is_valid, result_count, detected_patterns = (
                self.generate_sparql_for_question(variation, language)
            )

            if is_valid and result_count >= min_results:
                pairs.append(SyntheticPair(
                    nl_question=variation,
                    sparql_query=sparql,
                    patterns=detected_patterns or patterns,
                    source_example_id=seed_id,
                    result_count=result_count,
                    is_valid=True,
                    language=language,
                    generation_method="variation",
                ))

        return pairs

    def generate_combined_pairs(
        self,
        num_combinations: int = 10,
        min_results: int = 1,
        language: str = "it",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> list[SyntheticPair]:
        """
        Generate pairs by combining patterns from multiple seeds.

        Args:
            num_combinations: Number of combined questions to generate
            min_results: Minimum results required
            language: Language code
            progress_callback: Optional callback for progress

        Returns:
            List of validated SyntheticPair
        """
        pairs = []

        for i in range(num_combinations):
            if progress_callback:
                progress_callback(f"Generating combination {i+1}/{num_combinations}...")

            # Select random examples to combine
            selected = random.sample(
                self.seed_examples,
                min(3, len(self.seed_examples))
            )

            question, patterns = generate_combined_question(selected, self.llm)

            if not question:
                continue

            sparql, is_valid, result_count, detected_patterns = (
                self.generate_sparql_for_question(question, language)
            )

            if is_valid and result_count >= min_results:
                pairs.append(SyntheticPair(
                    nl_question=question,
                    sparql_query=sparql,
                    patterns=detected_patterns or patterns,
                    source_example_id=None,
                    result_count=result_count,
                    is_valid=True,
                    language=language,
                    generation_method="combination",
                ))

        return pairs

    def generate_dataset(
        self,
        num_variations_per_seed: int = 5,
        num_combinations: int = 10,
        min_results: int = 1,
        max_pairs: Optional[int] = None,
        language: str = "it",
        include_seeds: bool = True,
        verbose: bool = True,
    ) -> tuple[list[SyntheticPair], GenerationStats]:
        """
        Generate a complete synthetic dataset.

        Args:
            num_variations_per_seed: NL variations per seed example
            num_combinations: Number of pattern combinations to generate
            min_results: Minimum result count to accept a pair
            max_pairs: Maximum total pairs (None = unlimited)
            language: Language code
            include_seeds: Include original seed examples in output
            verbose: Print progress

        Returns:
            Tuple of (pairs, stats)
        """
        start_time = time.time()
        pairs = []
        stats = GenerationStats()

        def progress(msg):
            if verbose:
                print(msg)

        # Optionally include seed examples
        if include_seeds:
            progress("Adding seed examples...")
            for seed in self.seed_examples:
                seed_nl = seed.get("nl") or seed.get("question", "")
                seed_sparql = seed.get("sparql", "")
                patterns = seed.get("patterns", [])

                if seed_nl and seed_sparql:
                    pairs.append(SyntheticPair(
                        nl_question=seed_nl,
                        sparql_query=seed_sparql,
                        patterns=patterns,
                        source_example_id=seed.get("id"),
                        result_count=-1,  # Unknown for seeds
                        is_valid=True,
                        language=language,
                        generation_method="seed",
                    ))

                if max_pairs and len(pairs) >= max_pairs:
                    break

        # Generate variations from seeds
        progress(f"\nGenerating variations from {len(self.seed_examples)} seeds...")
        for i, seed in enumerate(self.seed_examples):
            if max_pairs and len(pairs) >= max_pairs:
                break

            progress(f"\n[{i+1}/{len(self.seed_examples)}] Processing seed...")

            seed_pairs = self.generate_variations_for_seed(
                seed=seed,
                num_variations=num_variations_per_seed,
                min_results=min_results,
                language=language,
                progress_callback=progress if verbose else None,
            )

            stats.total_attempts += num_variations_per_seed
            stats.successful_pairs += len(seed_pairs)
            stats.failed_validation += num_variations_per_seed - len(seed_pairs)

            # Update pattern stats
            for pair in seed_pairs:
                for pattern in pair.patterns:
                    stats.pairs_by_pattern[pattern] = (
                        stats.pairs_by_pattern.get(pattern, 0) + 1
                    )

            pairs.extend(seed_pairs)

            if max_pairs and len(pairs) >= max_pairs:
                break

        # Generate combined patterns
        if num_combinations > 0 and (not max_pairs or len(pairs) < max_pairs):
            progress(f"\nGenerating {num_combinations} pattern combinations...")

            remaining = max_pairs - len(pairs) if max_pairs else num_combinations
            combination_pairs = self.generate_combined_pairs(
                num_combinations=min(num_combinations, remaining),
                min_results=min_results,
                language=language,
                progress_callback=progress if verbose else None,
            )

            stats.total_attempts += num_combinations
            stats.successful_pairs += len(combination_pairs)

            pairs.extend(combination_pairs)

        stats.total_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"Generation complete!")
            print(f"  Total pairs: {len(pairs)}")
            print(f"  Success rate: {stats.successful_pairs}/{stats.total_attempts} "
                  f"({100*stats.successful_pairs/max(1,stats.total_attempts):.1f}%)")
            print(f"  Time: {stats.total_time:.1f}s")
            print(f"  Pairs by pattern: {stats.pairs_by_pattern}")

        return pairs, stats

    def save_dataset(
        self,
        pairs: list[SyntheticPair],
        output_path: str,
        format: str = "jsonl",
        include_metadata: bool = True,
    ) -> None:
        """
        Save the synthetic dataset to a file.

        Args:
            pairs: List of SyntheticPair
            output_path: Output file path
            format: Output format (jsonl, json, alpaca, sharegpt)
            include_metadata: Include patterns and other metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            # JSONL format - one JSON object per line
            with open(output_path, "w", encoding="utf-8") as f:
                for pair in pairs:
                    record = {
                        "instruction": "Traduci questa domanda in linguaggio naturale in una query SPARQL per il knowledge base LiITA.",
                        "input": pair.nl_question,
                        "output": pair.sparql_query,
                    }
                    if include_metadata:
                        record["patterns"] = pair.patterns
                        record["result_count"] = pair.result_count
                        record["source"] = pair.source_example_id
                        record["method"] = pair.generation_method
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        elif format == "json":
            # JSON array format
            records = []
            for pair in pairs:
                record = {
                    "instruction": "Traduci questa domanda in linguaggio naturale in una query SPARQL per il knowledge base LiITA.",
                    "input": pair.nl_question,
                    "output": pair.sparql_query,
                }
                if include_metadata:
                    record["patterns"] = pair.patterns
                    record["result_count"] = pair.result_count
                    record["source"] = pair.source_example_id
                    record["method"] = pair.generation_method
                records.append(record)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        elif format == "alpaca":
            # Alpaca format for fine-tuning
            records = []
            for pair in pairs:
                records.append({
                    "instruction": "Translate this natural language question to SPARQL for the LiITA linguistic knowledge base.",
                    "input": pair.nl_question,
                    "output": pair.sparql_query,
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        elif format == "sharegpt":
            # ShareGPT format for chat fine-tuning
            records = []
            for pair in pairs:
                records.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Translate this question to SPARQL for LiITA:\n\n{pair.nl_question}"
                        },
                        {
                            "from": "gpt",
                            "value": pair.sparql_query
                        }
                    ]
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        elif format == "hf":
            # HuggingFace datasets format (requires datasets library)
            try:
                from datasets import Dataset
            except ImportError:
                raise ImportError(
                    "HuggingFace datasets required for 'hf' format. "
                    "Install with: pip install datasets"
                )

            dataset = Dataset.from_dict({
                "instruction": [
                    "Traduci questa domanda in linguaggio naturale in una query SPARQL per il knowledge base LiITA."
                ] * len(pairs),
                "input": [p.nl_question for p in pairs],
                "output": [p.sparql_query for p in pairs],
                "patterns": [p.patterns for p in pairs],
            })
            dataset.save_to_disk(str(output_path))

        else:
            raise ValueError(f"Unknown format: {format}. Use jsonl, json, alpaca, sharegpt, or hf")

        print(f"Saved {len(pairs)} pairs to {output_path} ({format} format)")

    @staticmethod
    def load_dataset(path: str) -> list[SyntheticPair]:
        """
        Load a previously saved synthetic dataset.

        Args:
            path: Path to dataset file (jsonl or json)

        Returns:
            List of SyntheticPair
        """
        path = Path(path)
        pairs = []

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    pairs.append(SyntheticPair(
                        nl_question=record.get("input", ""),
                        sparql_query=record.get("output", ""),
                        patterns=record.get("patterns", []),
                        source_example_id=record.get("source"),
                        result_count=record.get("result_count", 0),
                        is_valid=True,
                        generation_method=record.get("method", "unknown"),
                    ))
        else:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
                for record in records:
                    pairs.append(SyntheticPair(
                        nl_question=record.get("input", ""),
                        sparql_query=record.get("output", ""),
                        patterns=record.get("patterns", []),
                        source_example_id=record.get("source"),
                        result_count=record.get("result_count", 0),
                        is_valid=True,
                        generation_method=record.get("method", "unknown"),
                    ))

        return pairs

# Persistent Translation Memory

This document explains the methodology and technical choices behind the persistent
translation memory feature: a small learning layer that lets the synthesizer and
agent pipelines remember past translations across process restarts, biasing future
generation toward previously-correct answers and away from previously-made
mistakes.

It was scoped and built to answer a concrete question raised during development of
the MCP server: *can the system learn from experience, so it avoids repeating a
mistake it already made once it has found the right fix, and answers a
already-solved question faster the second time?* The `docs/langgraph_architecture.md`
design notes had flagged exactly this ("Feedback learning: Store successful
translations for retrieval") as a future enhancement; this document describes how
it was actually implemented.

---

## 1. The problem

Before this feature, neither translation pipeline remembered anything across calls:

- `NL2SPARQL` (`nl2sparql/generation/synthesizer.py`) rebuilds its `HybridRetriever`
  from the static ~142-example curated corpus (`nl2sparql/data/sparql_queries_examples.json`)
  on every instantiation.
- `NL2SPARQLAgent` (`nl2sparql/agent/`) keeps a rich per-run `refinement_history` —
  the sequence of failed attempts and their errors — but that history is a
  LangGraph state field, discarded the instant `translate()` returns.
- The MCP server (`nl2sparql/mcp/server.py`) is one process per client session.
  Restart it, and every lesson it "learned" during that session is gone.

Ask the same question twice, even seconds apart in the same MCP session, and the
system regenerates the query from scratch both times — including any mistake it
made and then fixed the first time.

## 2. Scope and design constraints

This is a thesis project with no unit test suite; correctness is judged via F1
scoring against a 100-case gold dataset (`nl2sparql/evaluation/f1_evaluator.py`,
`scripts/run_f1_evaluation.py`). That shaped two decisions made explicitly before
writing any code:

| Decision | Choice | Why |
|---|---|---|
| **Fast path vs. bias-only** | **Bias-only.** A remembered match never skips generation or endpoint validation. | A pure fast path risks silently serving a stale answer if LiITA's live data changes between runs. Bias-only guarantees every returned query is still freshly validated, at the cost of not being a literal zero-LLM-call shortcut. |
| **Feature scope** | **Minimal core.** Just the memory store, its integration into both pipelines, the MCP server, and the CLI. No manual-correction MCP tool, no browsable memory resource, no promotion-to-corpus script. | Keeps the surface area small enough to explain cleanly in a thesis, and those extras can be added later without touching this design. |

The MCP server's `translate` tool goes through `NL2SPARQL` (the synthesizer), not
`NL2SPARQLAgent` — there is no agent-backed MCP tool today. So the synthesizer-side
integration is the half of this feature that actually matters for "the MCP version
of the project," even though the agent pipeline (used by the `nl2sparql agent` CLI
command) received the same treatment for consistency.

## 3. Methodological approach

The guiding idea is **reuse before invention**: every mechanism the two pipelines
already had for biasing generation toward good examples and away from bad ones was
identified first, and the memory feature was built to feed into those existing
mechanisms rather than adding a parallel decision path.

Two existing mechanisms turned out to do most of the work for free:

1. **The synthesizer's adaptation threshold.** `NL2SPARQL.translate()` already
   chooses between two generation strategies based on the top retrieval score:
   above `adaptation_threshold` (0.85) it *adapts* a single best-matching example;
   below it, it *synthesizes* from multiple exemplars. A remembered near-exact
   match naturally scores high enough to cross that threshold — so "jump straight
   to a known-correct answer" required no new branching logic, just feeding a
   remembered example into the same candidate pool the threshold already reads
   from. This was verified directly (see §8).

2. **The agent's `refinement_history` prompt channel.** `generate_sparql()` in
   `nl2sparql/agent/nodes.py` already renders `state["refinement_history"][-3:]`
   into a "## PREVIOUS ATTEMPTS (learn from these errors!)" prompt block. Seeding
   that same field with *remembered* failures (from a previous run, not just the
   current one) makes the agent avoid a documented mistake with zero new
   prompt-plumbing.

The corollary of "reuse before invention" is that the memory store's write-side
data model was designed to mirror the read-side consumers exactly:
`LearnedExample` is a superset of the existing `QueryExample` dataclass so it can
be wrapped as one with no translation step, and `TranslationMemory.retrieve_as_examples()`
returns the same `RetrievalResult` type `HybridRetriever.retrieve()` already
returns, so the two lists merge with a single `sorted(...)` call.

## 4. Architecture overview

```
nl2sparql/memory/
├── __init__.py      # re-exports TranslationMemory, LearnedExample, FailureRecord
├── records.py        # dataclasses + JSONL (de)serialization
├── matching.py        # normalize_question(), near-duplicate threshold
└── store.py           # TranslationMemory — the public API
```

```
                    ┌─────────────────────────┐
                    │   TranslationMemory      │
                    │  (~/.nl2sparql/memory/)  │
                    │  examples.jsonl           │
                    │  failures.jsonl           │
                    └────────────┬─────────────┘
                                 │ lookup() / retrieve_as_examples()
                 ┌───────────────┼────────────────┐
                 │                                 │
                 ▼                                 ▼
      NL2SPARQL.translate()              NL2SPARQLAgent.translate()
      (synthesizer.py)                    (agent/graph.py)
                 │                                 │
      merge into retrieved examples;     seed retrieved_examples /
      known_failures → prompt            refinement_history
                 │                                 │
                 ▼                                 ▼
        record_success() /                record_success() /
        record_failure()                  record_unresolved_run()
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                      TranslationMemory (write)
```

Both pipelines follow the same shape: **look up → merge/seed → generate & validate
(unchanged) → record outcome.** Nothing sits between generation and validation, so
memory can only ever make a run *start* from a better position, never skip the
checks that make a returned query trustworthy.

## 5. Storage design

Records persist as **JSONL** (one JSON object per line, append-only) under a
configurable directory, default `~/.nl2sparql/memory/` (`DEFAULT_MEMORY_PATH` in
`nl2sparql/config.py`, overridable via `NL2SPARQL_MEMORY_PATH`, matching the
project's existing `NL2SPARQL_*` env-var convention):

- `examples.jsonl` — `LearnedExample` records (successful translations)
- `failures.jsonl` — `FailureRecord` records (documented mistakes)

**Why JSONL and not SQLite or a single JSON array:**

- No concurrent-writer requirement across *processes* (an in-process
  `threading.Lock` in `TranslationMemory` covers the MCP server's
  `ThreadPoolExecutor`), so a database's transactional guarantees buy nothing here.
- No relational queries are needed — matching is embedding/BM25-similarity based,
  not `WHERE`-clause based.
- Appends are O(1) and crash-safe, unlike a single JSON array (which the curated
  corpus `sparql_queries_examples.json` uses) that requires a full read-modify-write
  on every change and risks corrupting the whole file on a crash mid-write.
- Plain text stays human-inspectable — useful for a thesis appendix, and for
  debugging what the system has "learned."

**Record schemas** (`nl2sparql/memory/records.py`):

```python
@dataclass
class LearnedExample:
    id: int                    # negative, avoids colliding with curated IDs 1..142
    nl: str
    sparql: str
    nl_normalized: str = ""    # auto-derived in __post_init__
    patterns: dict[str, float] = field(default_factory=dict)
    language: str = "it"
    confidence: float = 0.0
    result_count: int = 0
    source_pipeline: str = "synthesizer"   # "synthesizer" or "agent"
    provider: str = ""
    model: str = ""
    created_at: str = ""       # auto-derived, UTC ISO 8601
    hit_count: int = 0

@dataclass
class FailureRecord:
    id: int
    nl: str
    error: str
    nl_normalized: str = ""
    sparql: str = ""
    detected_patterns: list[str] = field(default_factory=list)
    kind: str = "unresolved"   # "attempt" (one refinement_history entry) or
                                # "unresolved" (a whole run that never converged)
    result_count: int = 0
    language: str = "it"
    source_pipeline: str = "synthesizer"
    provider: str = ""
    model: str = ""
    run_id: str = ""           # groups all "attempt" records from one run
    created_at: str = ""
```

IDs are negative specifically so a `LearnedExample`, if ever manually promoted
into the curated corpus, cannot silently collide with the curated IDs 1–142. They
are also random (62 bits from a UUID) rather than `min(existing_ids) - 1`, because
separate processes — the MCP server and CLI runs — can share one store, and each
only sees the records that existed when it loaded.

Identical successes (same normalized question and whitespace-normalized query) are
stored once: `record_success()` skips a duplicate, and `lookup()` additionally
de-duplicates by content, so repeated questions can't fill the few-shot slots with
copies of one query.

Loading is tolerant of damage: a line that is not valid JSON or doesn't fit the
record schema (e.g. a partial last line from a process killed mid-write) is
skipped with a warning rather than making the whole store — and every translation
using it — fail, and the next append starts on a fresh line.

`kind="attempt"` vs. `kind="unresolved"` on `FailureRecord` exists because not
every failed attempt is equally informative: an agent run's `refinement_history`
can contain several attempts en route to eventual success (not worth remembering
as "a mistake" in isolation) or none at all if the run never converged. The agent
integration (§7) writes one `attempt` record per `refinement_history` entry *plus*
one `unresolved` record summarizing the run, so both granularities are available —
retrieval naturally ranks by embedding similarity across both, and a consumer can
filter by `kind` later if the distinction turns out to matter.

## 6. Matching strategy

`nl2sparql/memory/matching.py` and `TranslationMemory.lookup()` combine two tiers,
cheapest first:

1. **Exact normalized match** — `normalize_question()` lowercases, strips
   punctuation, and collapses whitespace; an O(1) dict-free scan catches "asked
   again verbatim" (`"Quali lemmi esprimono tristezza?"` vs.
   `"quali lemmi esprimono tristezza"`) with no embedding computation and no
   threshold to justify.
2. **Near-duplicate semantic match** — cosine similarity via the *existing*
   `EmbeddingIndex` (`nl2sparql/retrieval/embeddings.py`, the same
   `paraphrase-multilingual-MiniLM-L12-v2` model and FAISS `IndexFlatIP` mechanism
   `HybridRetriever` already uses for the curated corpus). `NEAR_DUPLICATE_THRESHOLD
   = 0.93` is the working default.

`TranslationMemory` owns its own small `EmbeddingIndex` instance, separate from
`HybridRetriever`'s, **rebuilt lazily on the first `lookup()`/`retrieve_as_examples()`
call after any write** (`_index_dirty` flag) rather than incrementally. Both
`EmbeddingIndex.build_index()` and the underlying FAISS index are full-rebuild-only
in this codebase already — there's no incremental-add API to hook into — and at the
scale this feature targets (dozens to low-thousands of records for a thesis-scope
project), a full rebuild of short question strings is sub-second. Building bespoke
incremental-index logic was judged not worth the complexity for that scale; if the
store grows much larger, `TranslationMemory._ensure_index()` is the one place that
would need to change.

Failures are matched the same way but against a looser gate,
`FAILURE_RELEVANCE_THRESHOLD = 0.75`, because they only ever bias a prompt ("avoid
this"), never replace a decision. A loosely-related past mistake is still useful
context below the strict "same question" bar — but without any floor, mistakes
recorded for unrelated questions would be injected into every prompt once the
store is non-empty.

## 7. Integration: synthesizer pipeline (`nl2sparql/generation/synthesizer.py`)

`NL2SPARQL.__init__` gained one new parameter, `memory: TranslationMemory | None =
None`, defaulting to off. In `translate()`, immediately after the existing
`self.retriever.retrieve(...)` call:

```python
if self.memory is not None:
    memory_examples = self.memory.retrieve_as_examples(question, top_k=2)
    if memory_examples:
        retrieved = sorted(retrieved + memory_examples, key=lambda r: r.score, reverse=True)[:top_k]
    memory_hit = self.memory.lookup(question, top_k=3)
    known_failures = [f.error for f, _score in memory_hit.failures]
```

`known_failures` is threaded through `adapt_query()`/`synthesize_query()`
(`nl2sparql/generation/adapters.py`) into `build_adaptation_prompt()`/
`build_synthesis_prompt()` (`nl2sparql/constraints/prompt_builder.py`), which append
a `## KNOWN PAST MISTAKES FOR SIMILAR QUESTIONS (avoid these)` section when
non-empty — the same idea as the agent's existing "PREVIOUS ATTEMPTS" block,
introduced here since the synthesizer had no per-run failure memory of its own
(its `_validate_and_fix()` retry loop only keeps `was_fixed`/`fix_attempts`
counters, not per-attempt error history).

At the end of `translate()`, the outcome is recorded: `record_success()` only if
the query actually executed against the endpoint *and* returned at least one row;
`record_failure(..., kind="unresolved")` if it was invalid or returned 0 rows
(error `"Query returned 0 results"`); and nothing if it was never executed.
`ValidationResult.is_valid` alone is not enough here, because it also accepts
empty results and unexecuted queries — remembering those as "correct" would let
a wrong query reinforce itself through the adaptation path. (The agent needs no
such gate: its `verify_results()` node already treats 0 rows as invalid.)

**Verified behavior** (stubbed LLM client, no network calls, `validate=False` where
noted):

```
no-memory run:                 strategy = synthesize
memory run (nothing stored):   strategy = synthesize  (memory present but empty — no-op)
memory-biased run
  (after seeding an exact
   past success):               strategy = adapt   ← existing 0.85 threshold triggered
```

This confirms the central claim of §3: no new "skip to the answer" branch was
needed. Feeding one high-scoring remembered example into the existing candidate
list was sufficient to flip the generation strategy.

## 8. Integration: agent pipeline (`nl2sparql/agent/`)

`NL2SPARQLAgent.__init__` gained the same `memory` parameter. Two private helpers
do the work:

- `_memory_seed(question)` — calls `self.memory.lookup()` and formats the results
  as `seeded_examples`/`seeded_failures` dicts for `create_initial_state()`.
- `_remember_outcome(question, language, final_state)` — after `graph.invoke()`,
  writes `record_success()` if `final_state["is_valid"]`, else
  `record_unresolved_run(refinement_history=final_state["refinement_history"], ...)`
  — which persists one `attempt` `FailureRecord` per entry of the run's own
  `refinement_history`, plus one `unresolved` record summarizing it, under a shared
  `run_id`. Seeded failures are tagged `from_memory: True` and filtered out
  (`_own_attempts()`) both here and in the returned `refinement_history`, so a
  remembered mistake is never re-persisted as a new one and never reported as an
  attempt of the current run. Because `refine_query()` only appends to
  `refinement_history` before a retry, the run's final failing query
  (`generated_sparql` + `validation_errors`) is added explicitly, and the
  `unresolved` summary record carries that last query and error so it remains
  actionable when shown to the LLM later.

`stream()` (used by `nl2sparql agent --stream` and the Gradio agent app) is seeded
and records its outcome the same way, once the caller has consumed the whole
stream.

`create_initial_state()` (`nl2sparql/agent/state.py`) gained two optional
parameters, `seeded_examples` and `seeded_failures`, both defaulting to `None` so
every existing caller (including the evaluation harness's `AgentAdapter`) is
unaffected. `seeded_failures` populates the initial `refinement_history` —
because that field is `Annotated[list[dict], add]` (a LangGraph reducer), seeding
it at creation time means `generate_sparql()`'s existing prompt-rendering code
picks it up automatically.

`seeded_examples`, however, needed one extra step: `retrieve_examples()`
(`nl2sparql/agent/nodes.py`) *overwrites* `retrieved_examples` on every call
(including the first), so a seed placed directly into that field would be silently
dropped. Instead it's stored in a new state field, `memory_examples`, and
`retrieve_examples()` merges it into the live `HybridRetriever` results before
truncating to the top 5:

```python
if state.get("memory_examples"):
    retrieved_examples = sorted(
        retrieved_examples + state["memory_examples"],
        key=lambda e: e["score"], reverse=True,
    )[:5]
```

This mirrors the synthesizer's merge exactly, just adapted to the agent's
dict-based (rather than `RetrievalResult`-based) example representation.

## 9. MCP server and CLI

`MCPConfig` (`nl2sparql/mcp/server.py`) gained `memory_path: str | None = None` and
`memory_enabled: bool = True` (default **on** — the MCP server is the
cross-restart, interactive use case this feature exists for), plus matching
`NL2SPARQL_MEMORY_PATH` / `NL2SPARQL_MEMORY_ENABLED` entries in `from_env()`.
`NL2SPARQLMCPServer` gained a lazy singleton `memory` property, following the exact
existing pattern of its `translator`/`hybrid_retriever`/`ontology_retriever`
properties, and the `translator` property's `NL2SPARQL(...)` construction now
passes `memory=self.memory`. No changes were needed to `tools.py` or `resources.py`
— `handle_translate` already just calls `translator.translate(question)`, which now
carries memory transparently.

### Client-driven memory (no server-side LLM)

`translate` is the only tool that runs the synthesizer, and therefore the only
one that needs an LLM provider API. An MCP client that is itself an LLM — e.g.
Claude Desktop — typically skips it and writes SPARQL directly with
`retrieve_examples` / `execute_sparql`. Two additions let that workflow use the
memory with no server-side LLM call at all:

- **`retrieve_examples`** additionally returns `remembered_examples` (lookup hits
  above `NEAR_DUPLICATE_THRESHOLD`) and `known_failures` (hits above
  `FAILURE_RELEVANCE_THRESHOLD`). They are separate fields rather than merged into
  `examples`, because their scores are plain cosine similarities and not on the
  scale of the hybrid retrieval scores.
- **`record_translation(question, sparql, outcome, error?)`** — the client
  reports its final answer. A claimed `success` is re-executed against the
  endpoint and stored as a `LearnedExample` only if it runs and returns at least
  one row (the same bar the synthesizer applies); otherwise it is stored as a
  failure with the real error. `failure` stores the client's `error` description.
  Records are tagged `source_pipeline="mcp_client"`.

Judging whether the results actually answer the question is left to the client
— the server can only check that the query runs and is non-empty.

**When the client should record.** The server sends a recording policy to the
client in two places: its MCP `instructions` (sent at initialization; the memory
part is included only when memory is enabled) and, more briefly, in the
`record_translation` tool description, which clients see reliably. The policy:
record only after showing the user results, never mid-attempt; record a success
only once the user confirms the rows answer the question (the server can only
check that the query runs and is non-empty, so the user is the judge of
correctness, and a wrong-but-runnable query stored as a success would reinforce
itself); skip recording when a remembered query was reused unchanged, but record
when it was changed in any way, even just a filter value, since that is a
distinct mapping; record a failure without asking when the user points out a
mistake or an executed query was conceptually wrong, but not for syntax slips
fixed immediately. The server's exact-duplicate check in `record_success()`
remains as a safety net.

To guarantee the server makes no LLM calls at all, start it with
`--no-translate` (or `NL2SPARQL_TRANSLATE_ENABLED=0`): `translate` is removed from
the tool list and rejected if called anyway, and no provider client or API key is
ever needed.

`cli.py`'s `translate`, `agent`, and `mcp serve` commands, and `mcp/__main__.py`'s
argparse `serve` subcommand, all gained matching `--memory-path`/`--no-memory`
options — the two independent server-startup entrypoints (`cli.py`'s click-based
`mcp serve` and `mcp/__main__.py`'s argparse-based `serve`) are kept in sync by
hand, since they duplicate option definitions today (a pre-existing structural
quirk of the codebase, not introduced by this feature).

## 10. Keeping F1 evaluation clean

`scripts/run_f1_evaluation.py` constructs `NL2SPARQL(...)`/`NL2SPARQLAgent(...)`
without a `memory` kwarg, and this feature's constructors default `memory` to
`None`. **No script changes were required** to keep F1 scores memory-free — the
evaluation harness simply never opts in. This was treated as a hard requirement,
not a convenience: contaminating the primary reported metric with a memory store
that might trivially "solve" test-set questions from earlier manual runs would
undermine the evaluation's validity.

## 11. Design decisions summary

| Decision | Choice | Rationale |
|---|---|---|
| Fast path vs. bias-only | Bias-only | Never silently serve a stale answer if live LiITA data changes |
| Storage format | JSONL, two files | Crash-safe appends, no relational needs, human-inspectable |
| Matching | Exact-normalized tier + embedding near-duplicate tier (0.93) | Cheap common case, semantic generalization for paraphrases |
| Index rebuild | Full rebuild, lazy on first read after write | No incremental API exists in the reused `EmbeddingIndex`; cheap at target scale |
| Failure granularity | `attempt` + `unresolved` records | Preserves both fine- and coarse-grained signal without deciding upfront which matters |
| Synthesizer "jump to answer" | Reuse existing `adaptation_threshold`, no new branch | Fewer moving parts; verified to work via merge alone |
| Agent mistake-avoidance | Seed `refinement_history`, no new prompt code | `generate_sparql()` already renders this field |
| Feature scope | Minimal core (no MCP tool/resource/promotion script) | Smallest design that satisfies the stated goal; extensible later |
| Evaluation safety | `memory=None` default everywhere, harness unmodified | F1 numbers must stay uncontaminated by design, not by convention |

## 12. Limitations and future work

- **No fast path.** Every memory-biased run still costs at least one full
  generation call, even for a verbatim repeat. This was a deliberate
  correctness-over-speed tradeoff (§2); a future revalidate-then-serve fast path
  (skip generation, re-run only the remembered query against the live endpoint
  once, fall back to full generation if that fails) was designed but deferred.
- **No cross-run promotion path.** A `LearnedExample` never becomes part of the
  curated `sparql_queries_examples.json` corpus automatically; doing so would
  require a human-reviewed offline script, which was scoped out of the minimal
  core.
- **Limited MCP-level visibility.** `record_translation` lets a client add
  records and `retrieve_examples` shows matching ones, but there is no
  `liita://memory/translations` resource to browse the whole store, and no way to
  delete or correct an existing record from an MCP client — only by editing the
  JSONL files.
- **Global, not provider/model-partitioned retrieval.** Records are tagged with
  `provider`/`model` but retrieval does not currently prefer same-provider/model
  matches; a mistake made by one model is treated as equally relevant guidance for
  another. This is likely fine at thesis scale but is a candidate refinement if the
  store grows and provider behavior diverges noticeably.
- **No F1 ablation tooling yet.** Measuring *whether* memory actually improves
  translation quality (as opposed to just being wired in correctly) would need a
  deliberate, clearly-labeled experiment — e.g., running the F1 harness twice with
  the same fresh `--memory-path`, or against a paraphrased copy of the test
  questions to test generalization — which was intentionally left out of the F1
  script for this iteration to avoid conflating "does it work" with "does it help."

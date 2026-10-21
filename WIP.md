# Work in progress

Experimental features, open issues and planned experiments that are not (yet)
part of the official documentation. Not linked from the README on purpose.

## Experimental flags

### `NL2SPARQL_ALL_CONSTRAINTS`

Set to `1` / `true` / `yes` to inject **every** constraint block into the prompt,
instead of only the blocks matching the detected patterns
(`get_constraints_for_patterns()` in `nl2sparql/constraints/prompt_builder.py`).

- **Why**: pattern detection can miss or misclassify a question, and then the LLM
  never sees the rule it needs. Example: *"Di cosa è composto un edificio?"*
  detects no pattern and *"Cosa contiene una casa?"* detects `MORPHO_REGEX`, so
  neither prompt gets the `partMeronym` direction rule, while *"Quali sono le
  parti di un edificio?"* does.
- **Cost**: longer prompts on every call; rules irrelevant to the question may
  distract the LLM.
- **Not a benefit**: it does not make Anthropic prompt caching hit, because the
  synthesis system prompt also contains per-question retrieved examples.
- **Status**: never evaluated.

## Planned experiments

- [ ] **F1 with vs. without `NL2SPARQL_ALL_CONSTRAINTS`**: measures how much
      pattern-detection mistakes cost.
- [ ] **Translation memory ablation**: does memory improve F1? E.g. run the F1
      harness twice against the same fresh memory directory, or on paraphrased
      test questions to test generalization. The F1 scripts never enable memory
      today, so this needs a deliberate, clearly labelled setup.

## Translation memory: open issues

Known limitations found in review, deliberately not fixed yet:

- [ ] **Explore-path attempts are lost.** Agent attempts after an `explore`
      step bypass `refine_query`, so only the final one is recorded.
- [ ] **`hit_count` is never updated.** Persisting it would require rewriting
      the JSONL file, which can drop records appended by another process.
- [ ] **Full index rebuild after every write.** Each recorded outcome makes the
      next lookup re-embed the whole store. Fine at thesis scale.
- [ ] **Two lookups per synthesizer translation.** `retrieve_as_examples()` and
      `lookup()` each embed the question; one `lookup()` could serve both.
- [ ] **Memory is on by default in the CLI and MCP server**, so running the
      same CLI command twice can produce different SPARQL. Keep in mind for
      CLI-based experiments.
- [ ] **Recording policy is not enforced.** Whether Claude Desktop asks before
      saving, or skips a reused query, depends on it following the server
      instructions.
- [ ] **No way to delete or correct a record from an MCP client**, only by
      editing `examples.jsonl` / `failures.jsonl`.

## Small fixes

- [ ] README "Available MCP Tools" table is missing `fix_service_clause`.
- [ ] ruff import-order warnings in `nl2sparql/agent/graph.py`.

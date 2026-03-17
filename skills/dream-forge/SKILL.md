# dream-forge: Experiment Capture

Capture learnable moments from the current conversation as focused training experiments for a self-improving local model (Qwen3.5-9B).

## When to trigger

Run `/dream-forge` after completing work — bug fixes, features, research, config changes, or at any natural stopping point.

**Proactive suggestion:** If the user just committed code and hasn't captured an experiment yet in this session, suggest running `/dream-forge` before ending the conversation. Every resolved task is training data — don't let it go uncaptured.

## Core principle: capture learnable moments, not session summaries

A "learnable moment" is a specific point where someone had to think — a bug found, an edge case discovered, a non-obvious decision made, an error debugged. Each moment becomes one focused experiment with **actual code** in the context and solution.

**Good training data:** "Domain cap on 3 examples drops 2/3 of data" → the 5 lines of context code + the 5-line fix.

**Bad training data:** "Built 4 modules" → a prose summary of an entire file.

A broad session should produce 3-10 focused experiments, not 1 vague one.

## Instructions

### Step 1: Identify learnable moments

Scan the conversation for individual learnable moments. Look for:

- **Bugs found and fixed** (especially non-obvious ones)
- **Edge cases discovered** (empty inputs, overflow, off-by-one)
- **Design decisions that required thought** (why X instead of Y)
- **Errors debugged** (stack trace → root cause → fix)
- **Non-obvious patterns** (workarounds, compatibility fixes)
- **Corrections from review** (external feedback that changed the code)

For each moment, assign a difficulty:
- **hard**: Required debugging, research, or domain knowledge. A 9B model would likely get this wrong. These are the highest-value training examples.
- **easy**: Straightforward application of known patterns. A 9B model might get this right. These serve as anchor data to prevent regression.

Skip routine boilerplate (argparse setup, dataclass definitions, file I/O) — the model already knows those.

### Step 2: For each moment, extract focused data

For each learnable moment, build a focused experiment:

**problem** — The specific issue. Include the actual error message, the specific symptom, or the exact question. Be concrete:
- Good: "CrossEntropyLoss returns NaN when all labels are -100 because mean reduction divides 0/0. This happens when prompt_len exceeds input_ids length after truncation."
- Bad: "Fixed a bug in tune.py"

**pre_solution_context** — The actual code that was there BEFORE the fix. Read the relevant function/block and include it as real code (20-100 lines). This is what the model sees as input during training.
- For bugs: the buggy code
- For edge cases: the code missing the guard
- For design decisions: the code that needed to be written

**reference_solution** — The actual code AFTER the fix. The real function/block, not a description of it. This is what the model learns to produce.
- For code_change: the actual fixed function or code block
- For answer: the specific answer with evidence
- For config_change: the exact config/command
- For research_finding: the key finding with evidence

**error_output** — The actual error message, stack trace, or symptom that prompted the fix. Include it verbatim when available.

### Step 3: Determine git provenance (once per session)

Run these commands silently:

```bash
git rev-parse HEAD          # → repo_hash
git diff --stat             # → check for uncommitted changes
git status --porcelain      # → check dirty state
```

All experiments from this session share the same `repo_hash` and `git_start_hash`.

### Step 4: Assign task_group_id

All moments from the same logical task share a `task_group_id` (kebab-case, e.g., `tune-nan-loss-fix`). Moments from different tasks get different group IDs. This controls train/test split integrity.

### Step 5: Build the JSON for each moment

For each learnable moment, generate a complete experiment JSON:

```json
{
  "id": "<UUID v4>",
  "source": "claude",
  "timestamp": "<ISO 8601>",
  "project": "dream-forge",
  "problem": "<specific problem — include actual error messages>",
  "breakdown": ["<sub-step 1>", "<sub-step 2>"],
  "proposed_solutions": ["<what was tried>"],
  "review_issues": [],
  "final_plan": "<brief — what was decided and why>",
  "status": "resolved",
  "task_group_id": "<from step 4>",
  "superseded": false,
  "reference_solution": "<ACTUAL CODE — the function/block after the fix>",
  "resolution_type": "code_change",
  "pre_solution_context": [
    {
      "path": "src/example.py",
      "content": "<ACTUAL CODE — the function/block BEFORE the fix, 20-100 lines>",
      "revision": "<git hash>",
      "provenance": "retrieved_pre"
    }
  ],
  "post_solution_artifacts": [
    {
      "path": "src/example.py",
      "content": "<the modified code>",
      "revision": null,
      "provenance": "diff"
    }
  ],
  "repo_hash": "<from step 3>",
  "repo_dirty": false,
  "git_diff": null,
  "git_start_hash": "<from step 3>",
  "test_results": null,
  "build_results": null,
  "lint_results": null,
  "error_logs": null,
  "commands_run": null,
  "error_output": "<actual error message or null>",
  "constraints": null,
  "resolves_experiment_id": null,
  "synthetic": false,
  "generator": null,
  "parent_experiment_id": null,
  "generation_depth": 0,
  "tags": ["python"],
  "confidence": "inferred",
  "difficulty": "hard",
  "retrieval_count": 0,
  "positive_outcome_count": 0,
  "last_retrieved": null
}
```

### Step 6: Save and validate each experiment

For each moment:

1. Filename: `experiments/<YYYYMMDD>-<HHMMSS>-<slug>.json` (increment seconds for multiple)
2. Write the JSON file
3. Validate:

```bash
uv run python -m src.capture.cli validate experiments/<filename>.json
```

4. If validation fails, fix and retry (max 2 retries)

### Step 7: Report summary to user

After all moments are captured, show a summary table:

```
Captured N learnable moments:
  [hard] tune-nan-loss — NaN from all-masked labels (tune.py)
  [easy] best-of-n-truncation — CETT missing max_length (best_of_n.py)
  [hard] calibrate-query-mismatch — CETT query didn't match inference prompt
  ...
Saved to experiments/
```

## Difficulty guidelines

**hard** — assign when:
- The fix required understanding WHY something failed (not just WHAT failed)
- Multiple approaches were considered before finding the right one
- The bug was subtle (wrong results, not a crash)
- Domain knowledge was needed (VRAM budgets, tokenizer behavior, probe training format)
- External review was needed to find the issue

**easy** — assign when:
- The fix was mechanical (add a missing parameter, fix a typo)
- The pattern is well-known (add truncation, add a None check)
- Anyone reading the error message would know what to do

## What NOT to do

- Do NOT write prose summaries as reference_solution — use actual code
- Do NOT capture entire files — capture the specific function/block (20-100 lines)
- Do NOT fabricate code not present in the conversation
- Do NOT combine multiple unrelated fixes into one experiment
- Do NOT set `reference_solution` for unresolved experiments
- Do NOT generate synthetic data (synthetic=true) — this skill captures real experiments only
- Do NOT skip the `difficulty` field — the training pipeline uses it for curriculum ordering

# dream-forge: Experiment Capture

Capture a resolved or unresolved experiment from the current conversation into the dream-forge training pipeline.

## When to trigger

Run `/dream-forge` after completing a plan execution, fixing a bug, answering a research question, or at any natural stopping point where the conversation produced useful problem-solving data.

**Proactive suggestion:** If the user just committed code and hasn't captured an experiment yet in this session, suggest running `/dream-forge` before ending the conversation. Every resolved task is training data — don't let it go uncaptured.

## Instructions

You are capturing this conversation as a structured experiment for a self-improving local model (Qwen3.5-9B). Your output trains a LoRA adapter. Quality matters — bad captures produce bad training data.

### Step 1: Determine status

- **resolved**: The task has an actual implemented/verified outcome (code written, answer confirmed, config applied).
- **unresolved**: Still in progress, planning-only, or no concrete resolution yet.

### Step 2: Determine git provenance

Run these commands silently (do not show output to user unless errors occur):

```bash
git rev-parse HEAD          # → repo_hash
git diff --stat             # → if non-empty, check if changes are from THIS task
git status --porcelain      # → if dirty before task started, set repo_dirty=true
```

- `repo_hash`: Current HEAD commit hash
- `git_start_hash`: The commit hash at the START of the task (before any edits). If unknown, use `repo_hash`.
- `repo_dirty`: Set to `true` if the working tree had uncommitted changes BEFORE you started solving. If unsure, set `false`.
- `git_diff`: Run `git diff` to capture the net changes from this task. Store the full diff string. This is for provenance only — NOT the training target.

### Step 3: Extract the 5 stages

From the conversation history, extract:

1. **problem**: What the user needed. Be specific — include error messages, file paths, constraints. This is the primary embedding field for retrieval.
2. **breakdown**: How the problem was decomposed into sub-tasks. List of strings.
3. **proposed_solutions**: What approaches were considered. List of strings.
4. **review_issues**: Problems found during implementation. Each entry is a ReviewRound with `round_number`, `issues_found` (list), `corrections_made` (list). Empty list if no review cycles.
5. **final_plan**: The planning prose — what was decided and why. This is NOT the solution itself.

### Step 4: Extract the reference solution (resolved only)

For **resolved** experiments, `reference_solution` is MANDATORY. This is the **actual final correct output** that solved the problem:

- **code_change**: The final correct code (function, class, or block AFTER the fix). NOT a diff — the actual code. Store diffs in `git_diff` instead.
- **answer**: A structured summary of the answer.
- **config_change**: The exact config/command that resolved the issue.
- **research_finding**: The key finding, with evidence.

Set `resolution_type` to match: `code_change`, `answer`, `config_change`, or `research_finding`.

For **unresolved** experiments, set `reference_solution` to `null` and `resolution_type` to `null`.

### Step 5: Classify pre-solution vs post-solution context

**pre_solution_context** — files/info available BEFORE solving:
- `error_trace`: Files referenced in error messages/stack traces shown before you started solving.
- `user_provided`: Files the user explicitly shared or referenced.
- `retrieved_pre`: Files you read BEFORE your first Edit/Write/modifying-Bash call.

**post_solution_artifacts** — files created/modified AFTER solving:
- `diff`: Files appearing in `git diff` (modified by your edits).
- `test_output`: Test results from verification.
- `retrieved_post`: Files read after the first modification.

Use git provenance as the decision boundary: files present in `git diff` = post-solution.

### Step 6: Assign task_group_id

Generate a short, descriptive kebab-case ID for the underlying task/bug/feature (e.g., `compat-check-vram-fix`, `lance-db-crud-setup`). If this conversation continues work from a previous experiment, reuse the same `task_group_id`. This field controls train/test split integrity — experiments with the same group ID stay in the same split.

### Step 7: Build the JSON

Generate a complete JSON object with ALL required fields. Use this template:

```json
{
  "id": "<generate a UUID v4>",
  "source": "claude",
  "timestamp": "<current ISO 8601 datetime>",
  "project": "dream-forge",
  "problem": "<from step 3>",
  "breakdown": ["<step 1>", "<step 2>", "..."],
  "proposed_solutions": ["<approach 1>", "<approach 2>", "..."],
  "review_issues": [
    {
      "round_number": 1,
      "issues_found": ["<issue>"],
      "corrections_made": ["<fix>"]
    }
  ],
  "final_plan": "<from step 3>",
  "status": "resolved or unresolved",
  "task_group_id": "<from step 6>",
  "superseded": false,
  "reference_solution": "<from step 4, or null>",
  "resolution_type": "<code_change|answer|config_change|research_finding, or null>",
  "pre_solution_context": [
    {
      "path": "src/example.py",
      "content": "<relevant excerpt>",
      "revision": "<git hash or null>",
      "provenance": "error_trace|user_provided|retrieved_pre"
    }
  ],
  "post_solution_artifacts": [
    {
      "path": "src/example.py",
      "content": "<modified excerpt>",
      "revision": null,
      "provenance": "diff|test_output|retrieved_post"
    }
  ],
  "repo_hash": "<from step 2>",
  "repo_dirty": false,
  "git_diff": "<from step 2, or null>",
  "git_start_hash": "<from step 2, or null>",
  "test_results": null,
  "build_results": null,
  "lint_results": null,
  "error_logs": null,
  "commands_run": ["<commands executed during resolution>"],
  "error_output": "<original error that prompted the task, or null>",
  "constraints": ["<any constraints mentioned>"],
  "resolves_experiment_id": null,
  "synthetic": false,
  "generator": null,
  "parent_experiment_id": null,
  "generation_depth": 0,
  "tags": ["<domain tags: python, android, web, config, research, etc.>"],
  "confidence": "inferred",
  "retrieval_count": 0,
  "positive_outcome_count": 0,
  "last_retrieved": null
}
```

**Critical rules:**
- If `status` is `resolved`, both `reference_solution` and `resolution_type` MUST be non-null.
- `problem` and `task_group_id` MUST be non-empty strings.
- All enum values must be lowercase: `resolved`, `unresolved`, `code_change`, `answer`, `config_change`, `research_finding`, `claude`, `inferred`.
- UUIDs must be valid v4 format.
- Timestamps must be ISO 8601.

### Step 8: Save and validate

1. Generate a filename: `experiments/<YYYYMMDD>-<HHMMSS>-<slug>.json` where `<slug>` is a short kebab-case description (3-5 words).
2. Write the JSON file.
3. Run validation:

```bash
uv run python -m src.capture.cli validate experiments/<filename>.json
```

4. If validation fails, read the error, fix the JSON, and retry (max 2 retries).
5. Report the result to the user: experiment ID, status, and file path.

## What NOT to do

- Do NOT fabricate information not present in the conversation.
- Do NOT include post-solution artifacts (test results, diffs) in the training prompt context — they go in `post_solution_artifacts`, not `pre_solution_context`.
- Do NOT set `reference_solution` for unresolved experiments.
- Do NOT generate synthetic data (synthetic=true) — this skill captures real experiments only.
- Do NOT skip fields — the schema has ~35 fields and Pydantic validation will catch missing ones.

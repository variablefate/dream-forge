# dream-forge: Raw Session Export

Capture learnable moments from the current conversation as a raw session export. Qwen 9B processes this later into focused training experiments.

## When to trigger

Run `/dream-forge` after completing work — bug fixes, features, research, config changes, or at any natural stopping point.

**Proactive suggestion:** If the user just committed code and hasn't captured yet in this session, suggest running `/dream-forge` before ending the conversation.

## Your job: identify moments and write context. NOT code extraction.

You identify WHAT was learned and WHY it was hard. Qwen extracts the actual code from git diffs later. Your notes must be specific enough that a 9B model reading them alongside the git diff can understand the moment.

## Instructions

### Step 1: Gather git data

Run silently:

```bash
git log --oneline -20
```

Identify which commits are from this session. Then get the full log for those:

```bash
git log --patch <first_session_commit>^..HEAD
```

Also gather:
```bash
git rev-parse HEAD              # repo_hash
git log --oneline -1 <first>^   # git_start_hash (commit before session)
```

### Step 2: Identify learnable moments

Scan the conversation for moments where someone had to think. There are two kinds:

**Code moments** (produce before/after code diffs):
- **Bugs found and fixed** — especially non-obvious ones
- **Edge cases discovered** — empty inputs, overflow, cold-start issues
- **Errors debugged** — symptom → root cause → fix
- **Corrections from review** — external feedback that changed code
- **Non-obvious patterns** — workarounds, compatibility fixes

**Reasoning moments** (the investigation/judgment IS the value, not a code diff):
- **Design decisions** — why X instead of Y, tradeoffs evaluated, alternatives rejected with reasons
- **Investigations** — empirical tests that confirmed or debunked a hypothesis
- **External feedback evaluation** — analyzing suggestions and deciding what to adopt vs reject, with reasoning
- **Architecture choices** — why a module was structured a certain way, constraints that drove the design

Skip: boilerplate, routine argparse/dataclass/IO code, trivial renames.

### Step 3: For each moment, write a detailed note

Each moment needs enough context for a 9B model to understand it.

**Common fields (all moments):**
- **moment_type**: `code` or `reasoning`
- **problem**: 2-4 sentences. What was broken or needed? Be concrete — "CrossEntropyLoss returns NaN when all labels are -100" not "fixed a loss bug."
- **difficulty**: `hard` or `easy`
- **why_hard** or **why_easy**: 1-2 sentences explaining the rating.
- **quality**: 1-5 integer rating of the training data quality (not problem difficulty):
  - **5**: Focused code fix with real before/after code, specific error message, clear root cause
  - **4**: Good code or reasoning, minor gaps in context
  - **3**: Decent but broad — multiple changes or partly prose solution
  - **2**: Weak context or vague problem statement
  - **1**: Barely useful — keep as filler only
- **tags**: Domain tags (python, machine-learning, config, etc.)
- **task_group_id**: Kebab-case group ID. Moments from the same logical task share a group.

**Code moment fields** (moment_type=code):
- **symptom**: The observable behavior that revealed the issue.
- **root_cause**: Why it happened. The underlying mechanism.
- **files**: Which files are involved (list of paths)
- **function**: The specific function/class/block name if applicable
- **commit_before**: Commit hash where the buggy code exists (BEFORE the fix).
- **commit_after**: Commit hash where the fix exists (AFTER the fix).
- **extractable**: `true` if both before and after versions exist in git. `false` if the file was created in the same commit as the fix. **Non-extractable code moments are dropped by the splitter.**

**Reasoning moment fields** (moment_type=reasoning):
- **context_notes**: The full reasoning that makes this moment valuable. This is the actual training content — write as much as needed. Include:
  - What was investigated and why
  - What evidence was gathered (test results, empirical observations, measurements)
  - What alternatives were considered and why they were rejected
  - What was decided and the reasoning behind it
  - Any constraints that drove the decision
  This field replaces the code extraction pipeline — the splitter passes it through directly as the `reference_solution`. Write it as a clear, self-contained explanation that a 9B model could learn from.
- **resolution_type**: `answer` (for investigations/evaluations) or `research_finding` (for empirical results)
- **files**: Optional. Files referenced in the reasoning.

**How to find commit_before vs commit_after (code moments only):**
Run `git log --oneline -- <file>` to see the commit history. If the fix spans two commits (e.g., file created in commit A, bug fixed in commit B), use A as commit_before and B as commit_after. If the buggy code and fix are in the SAME commit (file was created with the fix already applied), mark `extractable: false`.

**Difficulty guidelines:**
- **hard**: Required debugging, research, or domain knowledge. Subtle bugs (wrong results, not crashes). Multiple approaches considered. External review needed. A 9B model would likely get this wrong.
- **easy**: Mechanical fix. Well-known pattern. Anyone reading the error would know the fix. Serves as anchor data.

### Step 4: Build the raw session export

Write a single JSON file with all moments:

```json
{
  "session_id": "<YYYYMMDD-HHMMSS>",
  "repo_hash": "<HEAD commit>",
  "git_start_hash": "<commit before session>",
  "commits": ["<hash1>", "<hash2>", "..."],
  "moments": [
    {
      "id": "<session_id>-<NNN>",
      "moment_type": "code",
      "problem": "<2-4 sentences, specific>",
      "symptom": "<what was observed>",
      "root_cause": "<why it happened>",
      "difficulty": "hard",
      "why_hard": "<explanation>",
      "quality": 5,
      "files": ["src/engine/data_prep.py"],
      "function": "apply_domain_cap",
      "commit_before": "<hash where buggy code exists>",
      "commit_after": "<hash where fix exists>",
      "extractable": true,
      "tags": ["python", "data-pipeline"],
      "task_group_id": "domain-cap-cold-start"
    },
    {
      "id": "<session_id>-<NNN>",
      "moment_type": "reasoning",
      "problem": "<what question/decision needed to be resolved>",
      "difficulty": "hard",
      "why_hard": "<explanation>",
      "quality": 4,
      "context_notes": "<full reasoning: what was investigated, evidence gathered, alternatives considered and rejected with reasons, final decision and rationale. Write as much as needed — this IS the training content.>",
      "resolution_type": "research_finding",
      "files": [],
      "tags": ["machine-learning", "evaluation"],
      "task_group_id": "template-mismatch-investigation"
    }
  ]
}
```

### Step 5: Save

1. Filename: `raw_sessions/<YYYYMMDD>-<HHMMSS>.json`
2. Write the file (create `raw_sessions/` if needed)
3. Report summary to user:

```
Captured N learnable moments:
  [hard] domain-cap-cold-start — Domain cap drops 2/3 of data at small sizes
  [easy] cett-truncation — CETT missing max_length on tokenizer call
  [hard] calibrate-query-mismatch — CETT query didn't match inference prompt
  ...
Saved to raw_sessions/<filename>
Run: uv run python -m src.engine.experiment_splitter raw_sessions/<filename>
```

## What NOT to do

- Do NOT extract code — that's Qwen's job using the git diffs
- Do NOT write vague one-liners — Qwen needs specific context to understand each moment
- Do NOT fabricate moments not present in the conversation
- Do NOT combine unrelated fixes into one moment
- Do NOT include routine boilerplate as moments

# dream-forge Project Instructions

## Post-Commit Learnable Moment Capture

A hook fires after every `git commit` and logs the commit to `.claude/hooks/pending_moments.jsonl`. After each commit you make, do the following:

1. Check if `.claude/hooks/pending_moments.jsonl` exists and has new entries
2. For each pending commit, write a learnable moment annotation:
   - Was this commit trivial (typo, formatting, boilerplate)? If so, skip it.
   - If non-trivial: write the moment annotation (problem, symptom, root_cause, difficulty, quality, files, function, commit_before, commit_after, extractable) and append it to the running session file at `raw_sessions/<today's date>.json`
   - Use `git log --oneline -2` to get the commit_before (parent) and commit_after (this commit)
   - Use `git log --oneline -- <file>` to verify the file existed before this commit (if not, mark extractable=false)
3. Clear the pending entry after annotating

This takes ~30 seconds per commit and captures moments while context is fresh.

At the end of the session, or when the user runs `/dream-forge`, run the experiment splitter:
```bash
uv run python -m src.engine.experiment_splitter raw_sessions/<filename>.json
```

## Session File Format

The running session file at `raw_sessions/<YYYYMMDD>-<HHMMSS>.json` accumulates moments throughout the conversation:

```json
{
  "session_id": "<YYYYMMDD-HHMMSS>",
  "repo_hash": "<updated after each commit>",
  "git_start_hash": "<first commit's parent>",
  "commits": ["<accumulates>"],
  "moments": [{"...each annotated commit..."}]
}
```

Create this file on the first commit of the session. Append moments as commits happen.

## Experiment Capture Reminder

After completing a task and committing code, if you haven't annotated the commit yet, do so immediately. Every resolved task is potential training data for the self-improving model. Don't skip this — the training pipeline needs real experiments to function.
# First training cycle completed 2026-03-18

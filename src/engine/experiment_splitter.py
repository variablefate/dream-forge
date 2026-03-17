"""Experiment splitter — Qwen processes raw session exports into focused experiments.

Takes a raw session export (Claude's moment annotations + commit hashes) and uses
the local model to extract actual code context and solutions from git diffs,
producing focused experiment JSONs ready for the training pipeline.

Pipeline:
  /dream-forge (Claude) → raw_sessions/<timestamp>.json (moments + notes)
  experiment_splitter.py (Qwen) → experiments/<timestamp>-<slug>.json (focused, with code)

Usage:
    uv run python -m src.engine.experiment_splitter raw_sessions/20260316-200000.json
    uv run python -m src.engine.experiment_splitter raw_sessions/20260316-200000.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

SYSTEM_PROMPT = (
    "You are a code analysis assistant. You read git diffs and extract "
    "specific code blocks as instructed. Output valid JSON only."
)

MAX_DIFF_TOKENS = 6000  # leave room for prompt + output in 8K context


# ── Git helpers ────────────────────────────────────────────────────────────────

def get_commit_diff(commit_hash: str) -> str:
    """Get the full diff for a single commit."""
    result = subprocess.run(
        ["git", "show", "--patch", "--no-color", commit_hash],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout if result.returncode == 0 else ""


def get_file_at_commit(path: str, commit_hash: str) -> str:
    """Get file content at a specific commit."""
    result = subprocess.run(
        ["git", "show", f"{commit_hash}:{path}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout if result.returncode == 0 else ""


def get_file_before_commit(path: str, commit_hash: str) -> str:
    """Get file content from the commit BEFORE this one."""
    return get_file_at_commit(path, f"{commit_hash}~1")


# ── Verification checks ───────────────────────────────────────────────────────

def verify_code_experiment(
    before_code: str, after_code: str,
    primary_file: str, commit_after: str,
) -> tuple[float, list[str]]:
    """Run lightweight verification checks on a code experiment.

    Returns (bonus, checks_passed) where bonus is 0.0 or 0.5.
    Checks:
      1. Diff sanity: before and after are actually different
      2. Syntax check: after-code compiles as valid Python
      3. Import check: the file imports cleanly at commit_after
    """
    checks_passed = []
    checks_failed = []

    # 1. Diff sanity — before and after must differ
    if before_code.strip() != after_code.strip():
        checks_passed.append("diff_sanity")
    else:
        checks_failed.append("diff_sanity: before and after are identical")

    # 2. Syntax check — after-code must be valid Python
    try:
        compile(after_code, "<after>", "exec")
        checks_passed.append("syntax")
    except SyntaxError:
        # Function-level extract may not compile standalone (missing imports)
        # Try wrapping in a dummy context
        try:
            compile(f"class _:\n" + "\n".join(
                f"  {line}" for line in after_code.split("\n")
            ), "<after_wrapped>", "exec")
            checks_passed.append("syntax_wrapped")
        except SyntaxError:
            checks_failed.append("syntax: after-code has syntax errors")

    # 3. Import check — the file imports at commit_after
    if primary_file.endswith(".py") and commit_after:
        module_path = primary_file.replace("/", ".").replace("\\", ".").removesuffix(".py")
        result = subprocess.run(
            ["git", "stash"],
            capture_output=True, text=True,
        )
        stashed = "No local changes" not in result.stdout

        try:
            result = subprocess.run(
                ["git", "checkout", commit_after, "--", primary_file],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                import_result = subprocess.run(
                    ["uv", "run", "python", "-c", f"import {module_path}"],
                    capture_output=True, text=True, timeout=30,
                )
                if import_result.returncode == 0:
                    checks_passed.append("import")
                else:
                    checks_failed.append(f"import: {import_result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            checks_failed.append("import: timeout")
        finally:
            # Restore working tree
            subprocess.run(["git", "checkout", "HEAD", "--", primary_file],
                          capture_output=True)
            if stashed:
                subprocess.run(["git", "stash", "pop"], capture_output=True)

    # Bonus: +0.5 if all checks pass
    bonus = 0.5 if len(checks_failed) == 0 and len(checks_passed) >= 2 else 0.0

    return bonus, checks_passed + [f"FAIL:{f}" for f in checks_failed]


def extract_function_from_file(content: str, function_name: str) -> str:
    """Extract a function/class block from file content by name.

    Simple heuristic: find 'def function_name(' or 'class function_name',
    then take lines until the next def/class at the same indent level.
    Returns up to 100 lines.
    """
    lines = content.split("\n")
    start = None
    start_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if (stripped.startswith(f"def {function_name}(") or
                stripped.startswith(f"def {function_name} (") or
                stripped.startswith(f"class {function_name}(") or
                stripped.startswith(f"class {function_name}:")):
            start = i
            start_indent = len(line) - len(stripped)
            break

    if start is None:
        return ""

    # Find the end: next def/class at same or lower indent, or end of file
    end = min(start + 100, len(lines))
    for i in range(start + 1, min(start + 150, len(lines))):
        stripped = lines[i].lstrip()
        current_indent = len(lines[i]) - len(stripped)
        if (stripped and current_indent <= start_indent and
                (stripped.startswith("def ") or stripped.startswith("class ") or
                 stripped.startswith("# ──"))):
            end = i
            break

    return "\n".join(lines[start:end]).rstrip()


# ── Qwen-based extraction (for complex cases) ────────────────────────────────

def qwen_extract(
    model, tokenizer, moment: dict, diff_text: str,
) -> dict | None:
    """Use Qwen to extract code context and solution from a diff.

    Returns {before_code, after_code, problem_refined} or None on failure.
    """
    import torch

    prompt = (
        f"Given this learnable moment and its git diff, extract:\n"
        f"1. The code BEFORE the fix (the buggy/missing version, 20-100 lines)\n"
        f"2. The code AFTER the fix (the corrected version, 20-100 lines)\n"
        f"3. A refined problem statement suitable as a coding task prompt\n\n"
        f"Moment:\n"
        f"  Problem: {moment['problem']}\n"
        f"  Symptom: {moment.get('symptom', 'N/A')}\n"
        f"  Root cause: {moment.get('root_cause', 'N/A')}\n"
        f"  Files: {', '.join(moment.get('files', []))}\n"
        f"  Function: {moment.get('function', 'N/A')}\n\n"
        f"Git diff:\n```\n{diff_text[:8000]}\n```\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"before_code": "...", "after_code": "...", "problem_refined": "..."}}'
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, enable_thinking=False,
        tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=8192,
    ).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=2048,
            temperature=0.1, do_sample=True,
        )

    response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    # Parse JSON from response (handle markdown code blocks)
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


# ── Deterministic extraction (preferred — no model needed) ────────────────────

def deterministic_extract(moment: dict) -> dict | None:
    """Extract code using git + function name — no model needed.

    Uses commit_before (buggy version) and commit_after (fixed version).
    Returns {before_code, after_code} or None if extraction fails.
    """
    # Drop non-extractable moments immediately
    if not moment.get("extractable", True):
        return None

    commit_after = moment.get("commit_after", moment.get("commit", ""))
    commit_before = moment.get("commit_before", "")
    files = moment.get("files", [])
    function_name = moment.get("function", "")

    if not commit_after or not files:
        return None

    primary_file = files[0]

    # Get the after (fixed) version
    after_content = get_file_at_commit(primary_file, commit_after)
    if not after_content:
        return None

    # Get the before (buggy) version
    if commit_before:
        before_content = get_file_at_commit(primary_file, commit_before)
    else:
        # Fall back to parent of commit_after
        before_content = get_file_before_commit(primary_file, commit_after)

    # If before doesn't exist, the file was created in commit_after — not extractable
    if not before_content:
        return None

    if function_name:
        before_code = extract_function_from_file(before_content, function_name)
        after_code = extract_function_from_file(after_content, function_name)
    else:
        # No function name — can't cleanly separate
        return None

    # Both before and after must have real code
    if not before_code or not after_code:
        return None

    return {
        "before_code": before_code,
        "after_code": after_code,
    }


# ── Build experiment JSON ──────────────────────────────────────────────────────

def build_experiment(
    moment: dict,
    session: dict,
    before_code: str,
    after_code: str,
    problem_refined: str | None = None,
    quality_bonus: float = 0.0,
    verification_checks: list[str] | None = None,
) -> dict:
    """Build a complete experiment JSON from extracted code."""
    primary_file = moment.get("files", ["unknown"])[0]
    base_quality = moment.get("quality") or 3
    final_quality = min(5.0, base_quality + quality_bonus)

    return {
        "id": str(uuid.uuid4()),
        "source": "claude",
        "timestamp": datetime.now().isoformat(),
        "project": "dream-forge",
        "problem": problem_refined or moment["problem"],
        "breakdown": [],
        "proposed_solutions": [],
        "review_issues": [],
        "final_plan": moment.get("root_cause", ""),
        "status": "resolved",
        "task_group_id": moment.get("task_group_id", "unknown"),
        "superseded": False,
        "reference_solution": after_code,
        "resolution_type": "code_change",
        "pre_solution_context": [
            {
                "path": primary_file,
                "content": before_code,
                "revision": moment.get("commit_before", moment.get("commit", "")),
                "provenance": "retrieved_pre",
            }
        ] if before_code else None,
        "post_solution_artifacts": [
            {
                "path": primary_file,
                "content": after_code[:500],
                "revision": None,
                "provenance": "diff",
            }
        ],
        "repo_hash": session.get("repo_hash", ""),
        "repo_dirty": False,
        "git_diff": None,
        "git_start_hash": session.get("git_start_hash", ""),
        "test_results": None,
        "build_results": None,
        "lint_results": None,
        "error_logs": None,
        "commands_run": None,
        "error_output": moment.get("symptom") or moment.get("problem", ""),
        "constraints": None,
        "resolves_experiment_id": None,
        "synthetic": False,
        "generator": None,
        "parent_experiment_id": None,
        "generation_depth": 0,
        "tags": moment.get("tags", ["python"]),
        "confidence": "inferred",
        "difficulty": moment.get("difficulty", None),
        "quality": final_quality,
        "retrieval_count": 0,
        "positive_outcome_count": 0,
        "last_retrieved": None,
    }


def build_reasoning_experiment(moment: dict, session: dict) -> dict:
    """Build an experiment from a reasoning moment — no git extraction.

    The context_notes field IS the reference_solution. The problem field
    is the question/decision that was investigated.
    """
    return {
        "id": str(uuid.uuid4()),
        "source": "claude",
        "timestamp": datetime.now().isoformat(),
        "project": "dream-forge",
        "problem": moment["problem"],
        "breakdown": [],
        "proposed_solutions": [],
        "review_issues": [],
        "final_plan": "",
        "status": "resolved",
        "task_group_id": moment.get("task_group_id", "unknown"),
        "superseded": False,
        "reference_solution": moment["context_notes"],
        "resolution_type": moment.get("resolution_type", "research_finding"),
        "pre_solution_context": None,
        "post_solution_artifacts": None,
        "repo_hash": session.get("repo_hash", ""),
        "repo_dirty": False,
        "git_diff": None,
        "git_start_hash": session.get("git_start_hash", ""),
        "test_results": None,
        "build_results": None,
        "lint_results": None,
        "error_logs": None,
        "commands_run": None,
        "error_output": None,
        "constraints": None,
        "resolves_experiment_id": None,
        "synthetic": False,
        "generator": None,
        "parent_experiment_id": None,
        "generation_depth": 0,
        "tags": moment.get("tags", ["research"]),
        "confidence": "inferred",
        "difficulty": moment.get("difficulty", None),
        "quality": moment.get("quality", None),
        "retrieval_count": 0,
        "positive_outcome_count": 0,
        "last_retrieved": None,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_session(
    session_path: Path,
    use_model: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Process a raw session export into focused experiments.

    Args:
        session_path: Path to the raw session JSON.
        use_model: If True, load Qwen for complex extractions.
                   If False, use deterministic git extraction only.
        dry_run: If True, print what would be created but don't write files.

    Returns:
        List of generated experiment dicts.
    """
    session = json.loads(session_path.read_text(encoding="utf-8"))
    moments = session.get("moments", [])

    if not moments:
        console.print("[yellow]No moments found in session export.[/yellow]")
        return []

    if not session.get("repo_hash"):
        console.print("[yellow]Warning: session missing repo_hash — code_change "
                       "experiments will fail data_prep completeness gates[/yellow]")

    console.print(f"[bold]Processing {len(moments)} moments from {session_path.name}[/bold]")

    # Separate code moments from reasoning moments
    model = None
    tokenizer = None
    experiments = []
    needs_model = []

    for moment in moments:
        moment_type = moment.get("moment_type", "code")

        # Reasoning moments: pass through directly (no git extraction needed)
        if moment_type == "reasoning":
            context_notes = moment.get("context_notes", "")
            if not context_notes:
                console.print(
                    f"  [red]DROPPED[/red] [{moment.get('difficulty', '?')}] "
                    f"{moment.get('task_group_id', '?')} — reasoning moment with empty context_notes")
                continue
            experiments.append(build_reasoning_experiment(moment, session))
            console.print(
                f"  [green]OK[/green] [{moment.get('difficulty', '?')}] "
                f"{moment.get('task_group_id', '?')} — reasoning moment")
            continue

        # Code moments: deterministic git extraction
        extracted = deterministic_extract(moment)
        if extracted and extracted.get("after_code"):
            # Run verification checks
            primary_file = moment.get("files", ["unknown"])[0]
            commit_after = moment.get("commit_after", moment.get("commit", ""))
            bonus, checks = verify_code_experiment(
                extracted["before_code"], extracted["after_code"],
                primary_file, commit_after,
            )
            experiments.append(build_experiment(
                moment, session,
                before_code=extracted["before_code"],
                after_code=extracted["after_code"],
                quality_bonus=bonus,
            ))
            check_str = ", ".join(c for c in checks if not c.startswith("FAIL:"))
            fail_str = ", ".join(c.replace("FAIL:", "") for c in checks if c.startswith("FAIL:"))
            quality_display = f"q={moment.get('quality', 3)}"
            if bonus > 0:
                quality_display += f"+{bonus}"
            status_parts = [f"deterministic", quality_display]
            if check_str:
                status_parts.append(f"verified:{check_str}")
            if fail_str:
                status_parts.append(f"[yellow]failed:{fail_str}[/yellow]")
            console.print(
                f"  [green]OK[/green] [{moment.get('difficulty', '?')}] "
                f"{moment.get('task_group_id', '?')} — {' | '.join(status_parts)}")
        else:
            # Check if explicitly marked non-extractable — drop it
            if not moment.get("extractable", True):
                console.print(
                    f"  [red]DROPPED[/red] [{moment.get('difficulty', '?')}] "
                    f"{moment.get('task_group_id', '?')} — marked non-extractable "
                    f"(no before version in git)")
            else:
                needs_model.append(moment)
                console.print(
                    f"  [yellow]PENDING[/yellow] [{moment.get('difficulty', '?')}] "
                    f"{moment.get('task_group_id', '?')} — needs model extraction")

    # Use Qwen for remaining moments (if requested and needed)
    if needs_model and use_model:
        console.print(f"\n  Loading model for {len(needs_model)} remaining moments...")
        from src.engine.model_loader import load_model
        model, tokenizer = load_model()

        for moment in needs_model:
            commit = moment.get("commit", "")
            diff_text = get_commit_diff(commit) if commit else ""

            if not diff_text:
                console.print(
                    f"  [red]SKIP[/red] {moment.get('task_group_id', '?')} — no diff available")
                continue

            result = qwen_extract(model, tokenizer, moment, diff_text)
            if result and result.get("after_code"):
                experiments.append(build_experiment(
                    moment, session,
                    before_code=result.get("before_code", ""),
                    after_code=result["after_code"],
                    problem_refined=result.get("problem_refined"),
                ))
                console.print(
                    f"  [green]OK[/green] [{moment.get('difficulty', '?')}] "
                    f"{moment.get('task_group_id', '?')} — model extraction")
            else:
                console.print(
                    f"  [red]FAIL[/red] {moment.get('task_group_id', '?')} — model couldn't extract")

        del model, tokenizer
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    elif needs_model:
        console.print(
            f"\n  [yellow]{len(needs_model)} moments need model extraction. "
            f"Re-run with --use-model to process them.[/yellow]")

    # Save experiments
    if not dry_run:
        output_dir = Path("experiments")
        output_dir.mkdir(parents=True, exist_ok=True)

        session_id = session.get("session_id", "unknown")
        for i, exp in enumerate(experiments):
            slug = exp.get("task_group_id", "moment")[:30]
            filename = f"{session_id}-{i:03d}-{slug}.json"
            path = output_dir / filename
            path.write_text(json.dumps(exp, indent=2, default=str), encoding="utf-8")

        console.print(
            f"\n[bold green]Saved {len(experiments)} experiments to {output_dir}/[/bold green]")

        # Validate all
        for i, exp in enumerate(experiments):
            slug = exp.get("task_group_id", "moment")[:30]
            filename = f"{session_id}-{i:03d}-{slug}.json"
            result = subprocess.run(
                ["uv", "run", "python", "-m", "src.capture.cli", "validate",
                 str(output_dir / filename)],
                capture_output=True, text=True,
            )
            status = "[green]valid[/green]" if result.returncode == 0 else "[red]invalid[/red]"
            console.print(f"  {filename}: {status}")
            if result.returncode != 0:
                console.print(f"    {result.stdout.strip()}", style="dim")
    else:
        console.print(f"\n[yellow]Dry run — {len(experiments)} experiments would be created[/yellow]")
        for exp in experiments:
            console.print(
                f"  [{exp.get('difficulty', '?')}] {exp.get('task_group_id', '?')} — "
                f"{len(exp.get('reference_solution', ''))} chars solution")

    return experiments


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process raw session exports into focused experiments")
    parser.add_argument("session", type=Path,
                        help="Path to raw session JSON")
    parser.add_argument("--use-model", action="store_true",
                        help="Use Qwen for complex extractions (requires GPU)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing")
    args = parser.parse_args()

    if not args.session.exists():
        console.print(f"[red]File not found: {args.session}[/red]")
        sys.exit(1)

    experiments = process_session(args.session, use_model=args.use_model, dry_run=args.dry_run)

    console.print(f"\nTotal: {len(experiments)} experiments")
    hard = sum(1 for e in experiments if e.get("difficulty") == "hard")
    easy = sum(1 for e in experiments if e.get("difficulty") == "easy")
    console.print(f"  hard: {hard}, easy: {easy}")


if __name__ == "__main__":
    main()

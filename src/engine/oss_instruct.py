"""OSS-Instruct: generate training pairs from real code snippets.

Takes extracted code snippets from repo_miner and generates
instruction→solution pairs by reverse-engineering the problem
each function solves.

Pipeline:
  1. Filter snippets (meaningful, clear purpose, right size)
  2. Generate instruction: "What problem would this code solve?"
  3. Verify round-trip: give instruction to model, does it produce similar code?
  4. Save as experiment JSON for the training pipeline

Inspired by Magicoder's OSS-Instruct approach.

Usage:
    uv run python -m src.engine.oss_instruct --generate --limit 50
    uv run python -m src.engine.oss_instruct --filter-only --stats
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

SNIPPETS_DIR = Path("data/oss_snippets")
OUTPUT_DIR = Path("experiments")
MIN_LINES = 8
MAX_LINES = 80
MAX_TOKENS_ESTIMATE = 1500  # rough char limit for 2048 token budget


# ── Snippet filtering ──────────────────────────────────────────────────────────

def load_snippets(snippets_dir: Path = SNIPPETS_DIR) -> list[dict]:
    """Load all extracted snippets from JSONL files."""
    snippets = []
    for f in snippets_dir.glob("*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
            try:
                snippets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return snippets


def filter_snippets(snippets: list[dict]) -> list[dict]:
    """Filter for snippets suitable for OSS-Instruct.

    Good candidates:
    - 8-80 lines (not too trivial, not too complex)
    - Have a docstring or clear function name
    - Not test files, not boilerplate
    - Code fits in training context (~1500 chars)
    """
    filtered = []
    seen_names = set()

    for s in snippets:
        # Size filter
        if s["line_count"] < MIN_LINES or s["line_count"] > MAX_LINES:
            continue

        # Code length filter
        if len(s["code"]) > MAX_TOKENS_ESTIMATE:
            continue

        # Skip common boilerplate names
        name = s["name"].lower()
        if name in ("main", "init", "setup", "teardown", "tostring",
                     "hashcode", "equals", "oncreate", "ondestroy",
                     "onresume", "onpause", "build", "dispose"):
            continue

        # Skip exact duplicates (same name + language + repo)
        # but allow same function name from different repos (legitimate diversity)
        dedup_key = f"{name}_{s['language']}_{s['repo']}"
        if dedup_key in seen_names:
            continue
        seen_names.add(dedup_key)

        # Prefer snippets with docstrings (clearer purpose)
        # But don't require them
        filtered.append(s)

    return filtered


def prioritize_snippets(snippets: list[dict]) -> list[dict]:
    """Sort by quality: docstring > clear name > language diversity."""
    def score(s):
        sc = 0
        if s["has_docstring"]:
            sc += 10
        # Prefer medium-length functions (not too short, not too long)
        if 15 <= s["line_count"] <= 50:
            sc += 5
        # Boost underrepresented languages
        if s["language"] in ("rust", "swift", "go", "dart"):
            sc += 3
        return sc

    return sorted(snippets, key=score, reverse=True)


# ── Instruction generation ────────────────────────────────────────────────────

INSTRUCT_PROMPT = """You are analyzing a code snippet. Generate a clear, specific coding problem description that would lead someone to write this code as the solution.

Rules:
- Write the problem as if asking a developer to implement this
- Be specific about inputs, outputs, and behavior
- Don't mention the function name or give away the solution structure
- The problem should be self-contained (no external context needed)
- Keep it to 2-4 sentences

Code:
```{language}
{code}
```

Problem description:"""


def generate_instruction(model, tokenizer, snippet: dict) -> str | None:
    """Use the model to generate an instruction for a code snippet."""
    import torch

    prompt = INSTRUCT_PROMPT.format(
        language=snippet["language"],
        code=snippet["code"],
    )

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, enable_thinking=False,
        tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048,
    ).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200,
            temperature=0.3, do_sample=True,
        )

    response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    # Basic validation
    if len(response) < 20:
        return None
    if "```" in response:
        # Model leaked code — bad instruction
        return None

    return response


# ── Round-trip verification ───────────────────────────────────────────────────

def verify_round_trip(
    model, tokenizer, instruction: str, original_code: str, language: str,
) -> tuple[bool, str]:
    """Give the instruction to the model (blind) and check if it produces similar code.

    Returns (passed, generated_code).
    """
    import torch

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Solve the problem accurately."},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages, enable_thinking=False,
        tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048,
    ).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=500,
            temperature=0.1, do_sample=True,
        )

    response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    # Check if the response contains code
    has_code = ("def " in response or "fun " in response or
                "func " in response or "function " in response or
                "fn " in response or "class " in response or
                "```" in response)

    # Basic similarity: does the response address the same problem?
    # (Not checking for exact match — different approaches are fine)
    from src.store.embeddings import embed_text, cosine_similarity
    sim = cosine_similarity(embed_text(original_code[:500]), embed_text(response[:500]))

    # Threshold: if similarity > 0.4 and has code, the instruction is clear enough
    passed = has_code and sim > 0.4

    return passed, response


# ── Build experiment ──────────────────────────────────────────────────────────

def build_oss_experiment(
    snippet: dict,
    instruction: str,
    verified: bool,
) -> dict:
    """Build an experiment JSON from an OSS-Instruct pair."""
    return {
        "id": str(uuid.uuid4()),
        "source": "claude",
        "timestamp": datetime.now().isoformat(),
        "project": "dream-forge",
        "problem": instruction,
        "breakdown": [],
        "proposed_solutions": [],
        "review_issues": [],
        "final_plan": "",
        "status": "resolved",
        "task_group_id": f"oss-{snippet['repo'].replace('/', '-')}-{snippet['name'][:20]}",
        "superseded": False,
        "reference_solution": snippet["code"],
        "resolution_type": "code_change",
        "pre_solution_context": [
            {
                "path": snippet["file_path"],
                "content": snippet.get("context", ""),
                "revision": None,
                "provenance": "retrieved_pre",
            }
        ] if snippet.get("context") else None,
        "post_solution_artifacts": None,
        "repo_hash": "",
        "repo_dirty": False,
        "git_diff": None,
        "git_start_hash": None,
        "test_results": None,
        "build_results": None,
        "lint_results": None,
        "error_logs": None,
        "commands_run": None,
        "error_output": None,
        "constraints": [f"Language: {snippet['language']}",
                        f"From: {snippet['repo']}"],
        "resolves_experiment_id": None,
        "synthetic": True,
        "generator": "oss_instruct",
        "parent_experiment_id": None,
        "generation_depth": 1,
        "tags": [snippet["language"], "oss-instruct"],
        "confidence": "inferred",
        "difficulty": "easy",
        "quality": 4.0 if verified else 3.0,
        "retrieval_count": 0,
        "positive_outcome_count": 0,
        "last_retrieved": None,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_oss_instruct(
    limit: int = 50,
    verify: bool = True,
    snippets_dir: Path = SNIPPETS_DIR,
    output_dir: Path = OUTPUT_DIR,
):
    """Generate OSS-Instruct training pairs."""
    print(f"Loading snippets from {snippets_dir}...")
    all_snippets = load_snippets(snippets_dir)
    print(f"  Total: {len(all_snippets)}")

    filtered = filter_snippets(all_snippets)
    print(f"  After filtering: {len(filtered)}")

    prioritized = prioritize_snippets(filtered)
    candidates = prioritized[:limit]
    print(f"  Selected top {len(candidates)} candidates")

    # Load model
    print("\nLoading model...")
    from src.engine.model_loader import load_model
    model, tokenizer = load_model()

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    verified_count = 0
    failed = 0

    print(f"\nGenerating instructions for {len(candidates)} snippets...")

    for i, snippet in enumerate(candidates):
        # Generate instruction
        instruction = generate_instruction(model, tokenizer, snippet)
        if not instruction:
            failed += 1
            print(f"  [{i+1}/{len(candidates)}] SKIP {snippet['name']} — bad instruction")
            continue

        # Verify round-trip
        is_verified = False
        if verify:
            is_verified, _ = verify_round_trip(
                model, tokenizer, instruction, snippet["code"], snippet["language"])

        # Build and validate experiment
        exp = build_oss_experiment(snippet, instruction, is_verified)

        # Validate against schema before saving
        try:
            from src.capture.schema import Experiment
            Experiment(**exp)
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{len(candidates)}] INVALID {snippet['name']} — {str(e)[:80]}")
            continue

        slug = f"oss-{snippet['language']}-{snippet['name'][:20]}"
        filename = f"oss-{datetime.now().strftime('%Y%m%d%H%M%S')}-{i:03d}-{slug}.json"
        (output_dir / filename).write_text(
            json.dumps(exp, indent=2, default=str), encoding="utf-8")

        generated += 1
        if is_verified:
            verified_count += 1

        status = "VERIFIED" if is_verified else "unverified"
        print(f"  [{i+1}/{len(candidates)}] {status} | {snippet['language']:<10} "
              f"{snippet['repo']}/{snippet['name']}")

    print(f"\nDone: {generated} generated, {verified_count} verified, {failed} failed")

    # Cleanup
    del model, tokenizer
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return generated, verified_count


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate training pairs from code snippets (OSS-Instruct)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate instruction-solution pairs")
    parser.add_argument("--filter-only", action="store_true",
                        help="Show filtering stats without generating")
    parser.add_argument("--stats", action="store_true",
                        help="Show snippet statistics")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max snippets to process (default: 50)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip round-trip verification (faster)")
    args = parser.parse_args()

    if args.filter_only or args.stats:
        snippets = load_snippets()
        print(f"Total snippets: {len(snippets)}")
        filtered = filter_snippets(snippets)
        print(f"After filtering: {len(filtered)}")

        # Language breakdown
        by_lang = {}
        for s in filtered:
            by_lang[s["language"]] = by_lang.get(s["language"], 0) + 1
        print("\nFiltered by language:")
        for lang, count in sorted(by_lang.items(), key=lambda x: -x[1]):
            print(f"  {lang:<15} {count:>5}")

        # Docstring stats
        with_doc = sum(1 for s in filtered if s["has_docstring"])
        print(f"\nWith docstring: {with_doc}/{len(filtered)} ({with_doc/len(filtered):.0%})")
        return

    if args.generate:
        run_oss_instruct(
            limit=args.limit,
            verify=not args.no_verify,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

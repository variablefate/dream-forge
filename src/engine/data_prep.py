"""C.5 SFT data construction — experiment JSON to tokenized training pairs.

Reads resolved experiments from disk, applies completeness gates, builds
prompt→completion pairs using pre-solution context only, handles split
management, curriculum ordering, and domain balancing.

This is the bridge between captured experiments (Lane A) and LoRA training
(tune.py). The prompt builder reads ONLY pre_solution_context — it never
touches post_solution_artifacts. This makes leakage a code bug, not a
policy violation.

Usage:
    from src.engine.data_prep import prepare_training_data, DataPrepConfig
    config = DataPrepConfig(data_dir=Path("experiments"))
    dataset = prepare_training_data(config, tokenizer)

CLI:
    uv run python -m src.engine.data_prep --data experiments/ --stats
    uv run python -m src.engine.data_prep --data experiments/ --output data/training_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = "You are a helpful coding assistant. Solve the problem accurately."
CONTEXT_CHAR_LIMIT = 2000       # per context file
ERROR_CHAR_LIMIT = 800          # error_output truncation
MAX_CONTEXT_FILES = 3
SINGLE_DOMAIN_CAP = 0.40        # no domain exceeds 40% of training data
SPLIT_ASSIGNMENTS_PATH = Path("data/split_assignments.json")

# Priority scoring defaults (from C.7.1 + Grok feedback)
PRIORITY_WEIGHTS = {
    "detector_risk": 0.4,
    "disagreement": 0.3,
    "tier1a_fail": 0.2,
    "trap_hit": 0.1,
}


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class DataPrepConfig:
    data_dir: Path = Path("experiments")
    split: str = "train"                    # which split to prepare
    max_seq_length: int = MAX_SEQ_LENGTH
    domain_cap: float = SINGLE_DOMAIN_CAP
    curriculum_sort: bool = True            # sort by difficulty (hard-first)
    include_synthetic: bool = False         # include depth=1 synthetics
    output_path: Path | None = None        # if set, write JSONL


# ── Completeness gates ────────────────────────────────────────────────────────

def _check_completeness(exp: dict) -> str | None:
    """Return rejection reason, or None if experiment is SFT-eligible.

    Per C.5 data construction policy:
    - code_change: requires repo_hash + pre_solution_context + one of error/test/commands
    - answer: requires reference_solution (non-empty)
    - config_change: requires reference_solution + error_output
    - research_finding: requires reference_solution (non-empty)
    """
    res_type = exp.get("resolution_type", "answer")
    ref = exp.get("reference_solution", "")

    if not ref or not ref.strip():
        return "empty reference_solution"

    if res_type == "code_change":
        if not exp.get("repo_hash"):
            return "code_change missing repo_hash"
        pre_ctx = exp.get("pre_solution_context") or []
        if not pre_ctx:
            return "code_change missing pre_solution_context"
        has_signal = (
            exp.get("error_output")
            or exp.get("test_results")
            or exp.get("commands_run")
        )
        if not has_signal:
            return "code_change missing error_output/test_results/commands_run"

    elif res_type == "config_change":
        if not exp.get("error_output"):
            return "config_change missing error_output"

    return None


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(exp: dict) -> tuple[str, str, str]:
    """Build (system_msg, user_msg, reference_solution) from an experiment.

    Reads ONLY pre_solution_context. Never touches post_solution_artifacts.
    Format matches tune.py / wake.py / dream.py exactly.
    """
    problem = exp.get("problem", "")
    error_output = exp.get("error_output", "") or ""
    # pre_solution_context only — mechanically enforced
    context_files = exp.get("pre_solution_context") or []

    context_text = ""
    for cf in context_files[:MAX_CONTEXT_FILES]:
        path = cf.get("path", "file") if isinstance(cf, dict) else "file"
        content = cf.get("content", "") if isinstance(cf, dict) else ""
        context_text += f"\n### {path}\n{content[:CONTEXT_CHAR_LIMIT]}\n"

    system_msg = SYSTEM_PROMPT
    user_msg = problem
    if error_output:
        user_msg += f"\n\nError:\n{error_output[:ERROR_CHAR_LIMIT]}"
    if context_text:
        user_msg += f"\n\nRelevant code:{context_text}"

    return system_msg, user_msg, exp["reference_solution"]


# ── Priority scoring (C.7.1 active replay) ───────────────────────────────────

def compute_priority(exp: dict) -> float:
    """Compute curriculum priority score for an experiment.

    Higher = harder / more valuable for training. Used for curriculum
    ordering (hard examples first in early training).
    """
    score = 0.0

    # Claude's difficulty estimate from capture time
    difficulty = exp.get("difficulty", "")
    if difficulty == "hard":
        score += 0.3
    # "easy" gets no boost — serves as anchor data

    # Claude's data quality rating (1-5, higher = better training data)
    quality = exp.get("quality")
    if quality is not None:
        # Scale 1-5 to 0.0-0.2 boost (quality 3 = neutral, 5 = +0.2, 1 = -0.1)
        score += (quality - 3) * 0.05

    # Detector risk (if available from calibrate.py)
    detector_risk = exp.get("_detector_risk", 0.0)
    score += PRIORITY_WEIGHTS["detector_risk"] * detector_risk

    # Wake/teacher disagreement (embedding distance, if available)
    disagreement = exp.get("_disagreement", 0.0)
    score += PRIORITY_WEIGHTS["disagreement"] * disagreement

    # Tier 1a failure (binary)
    test_results = exp.get("test_results")
    if test_results and not test_results.get("passed", True):
        score += PRIORITY_WEIGHTS["tier1a_fail"]

    # Trap library hit (binary, if tagged)
    if exp.get("_trap_hit", False):
        score += PRIORITY_WEIGHTS["trap_hit"]

    return score


# ── Split management ──────────────────────────────────────────────────────────

def load_split_assignments(path: Path = SPLIT_ASSIGNMENTS_PATH) -> dict[str, str]:
    """Load task_group_id → split assignments from disk."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            k: v.get("split", "train") if isinstance(v, dict) else v
            for k, v in data.get("assignments", data).items()
        }
    except (json.JSONDecodeError, KeyError):
        return {}


def get_split_for_experiment(exp: dict, assignments: dict[str, str]) -> str:
    """Get split assignment for an experiment. New groups default to 'train'."""
    group_id = exp.get("task_group_id", "")
    return assignments.get(group_id, "train")


# ── Domain balancing ──────────────────────────────────────────────────────────

def apply_domain_cap(
    examples: list[dict], cap: float = SINGLE_DOMAIN_CAP,
) -> list[dict]:
    """Cap any single domain tag at `cap` fraction of total examples.

    Removes excess examples from over-represented domains (keeps highest
    priority first). Returns the capped list.
    """
    if not examples or cap >= 1.0:
        return examples

    # Don't cap tiny datasets — the cap prevents single-domain dominance
    # at scale, but at <20 examples it just throws away scarce data
    if len(examples) < 20:
        return examples

    # Count primary domain per example (first tag)
    domain_counts: Counter = Counter()
    for ex in examples:
        domain = ex.get("_primary_domain", "unknown")
        domain_counts[domain] += 1

    total = len(examples)
    max_per_domain = max(1, int(total * cap))

    # Check if any domain exceeds cap
    over_cap = {d for d, c in domain_counts.items() if c > max_per_domain}
    if not over_cap:
        return examples

    # Keep all examples from capped domains up to max, drop lowest priority
    kept = []
    domain_kept: Counter = Counter()
    for ex in examples:  # already sorted by priority
        domain = ex.get("_primary_domain", "unknown")
        if domain in over_cap and domain_kept[domain] >= max_per_domain:
            continue
        domain_kept[domain] += 1
        kept.append(ex)

    return kept


# ── Diversity diagnostic ──────────────────────────────────────────────────────

def compute_batch_diversity(texts: list[str]) -> float:
    """Compute embedding spread (cosine std-dev) for a batch of texts.

    Returns standard deviation of pairwise cosine similarities.
    Higher = more diverse batch. Used as a diagnostic metric.
    """
    if len(texts) < 2:
        return 0.0

    from src.store.embeddings import embed_texts, cosine_similarity

    vecs = embed_texts(texts)
    similarities = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            similarities.append(cosine_similarity(vecs[i], vecs[j]))

    return float(np.std(similarities)) if similarities else 0.0


# ── Main pipeline ─────────────────────────────────────────────────────────────

@dataclass
class PreparedExample:
    """A single prepared training example (not yet tokenized)."""
    system_msg: str
    user_msg: str
    reference_solution: str
    source_file: str
    priority: float
    domain: str
    task_group_id: str
    split: str
    resolution_type: str


@dataclass
class PreparedDataset:
    """The full prepared dataset with metadata."""
    examples: list[PreparedExample]
    stats: dict = field(default_factory=dict)


def prepare_training_data(
    config: DataPrepConfig,
    tokenizer: "PreTrainedTokenizer | None" = None,
) -> PreparedDataset:
    """Load experiments, apply gates, build prompts, sort by curriculum.

    Args:
        config: Data preparation config.
        tokenizer: If provided, validates token counts and writes tokenized
                   output. If None, returns untokenized PreparedExamples.

    Returns:
        PreparedDataset with examples and statistics.
    """
    split_assignments = load_split_assignments()

    raw_experiments = []
    rejected = Counter()

    for json_file in sorted(config.data_dir.glob("*.json")):
        try:
            exp = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            rejected["parse_error"] += 1
            continue

        # Filter: resolved only
        if exp.get("status") != "resolved":
            rejected["not_resolved"] += 1
            continue

        # Filter: not superseded
        if exp.get("superseded", False):
            rejected["superseded"] += 1
            continue

        # Filter: synthetics (unless explicitly included)
        if exp.get("synthetic", False) and not config.include_synthetic:
            rejected["synthetic"] += 1
            continue

        # Completeness gate
        reason = _check_completeness(exp)
        if reason:
            rejected[f"incomplete:{reason}"] += 1
            continue

        # Split assignment
        split = get_split_for_experiment(exp, split_assignments)
        if split != config.split:
            continue  # not in the requested split

        # Annotate with metadata for sorting/filtering
        exp["_source_file"] = json_file.name
        exp["_priority"] = compute_priority(exp)
        tags = exp.get("tags", [])
        exp["_primary_domain"] = tags[0] if tags else "unknown"
        raw_experiments.append(exp)

    # Curriculum sort: highest priority first
    if config.curriculum_sort:
        raw_experiments.sort(key=lambda e: e["_priority"], reverse=True)

    # Domain cap
    raw_experiments = apply_domain_cap(raw_experiments, config.domain_cap)

    # Build prompts
    examples = []
    skipped_token_budget = 0

    for exp in raw_experiments:
        system_msg, user_msg, ref_solution = build_prompt(exp)

        # Optional: validate token count if tokenizer available
        if tokenizer is not None:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ref_solution},
            ]
            text = tokenizer.apply_chat_template(
                messages, enable_thinking=False, tokenize=False)
            token_count = len(tokenizer.encode(text))

            # Check that response tokens exist after prompt masking
            prompt_messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, enable_thinking=False, tokenize=False,
                add_generation_prompt=True)
            prompt_tokens = len(tokenizer.encode(prompt_text))

            if prompt_tokens >= min(token_count, config.max_seq_length):
                skipped_token_budget += 1
                continue

        examples.append(PreparedExample(
            system_msg=system_msg,
            user_msg=user_msg,
            reference_solution=ref_solution,
            source_file=exp["_source_file"],
            priority=exp["_priority"],
            domain=exp["_primary_domain"],
            task_group_id=exp.get("task_group_id", ""),
            split=config.split,
            resolution_type=exp.get("resolution_type", "answer"),
        ))

    # Diversity diagnostic
    diversity = 0.0
    if len(examples) >= 2:
        sample_texts = [e.user_msg[:200] for e in examples[:100]]
        diversity = compute_batch_diversity(sample_texts)

    # Stats
    domain_counts = Counter(e.domain for e in examples)
    type_counts = Counter(e.resolution_type for e in examples)
    stats = {
        "total_eligible": len(examples),
        "rejected": dict(rejected),
        "skipped_token_budget": skipped_token_budget,
        "domain_distribution": dict(domain_counts),
        "resolution_type_distribution": dict(type_counts),
        "embedding_diversity_std": diversity,
        "curriculum_sorted": config.curriculum_sort,
        "split": config.split,
    }

    return PreparedDataset(examples=examples, stats=stats)


# ── JSONL export ──────────────────────────────────────────────────────────────

def export_jsonl(dataset: PreparedDataset, output_path: Path) -> int:
    """Export prepared examples as JSONL (one conversation per line).

    Format matches tune.py's expected input: each line is a JSON object
    with system_msg, user_msg, reference_solution, and metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset.examples:
            record = {
                "system_msg": ex.system_msg,
                "user_msg": ex.user_msg,
                "reference_solution": ex.reference_solution,
                "source_file": ex.source_file,
                "priority": ex.priority,
                "domain": ex.domain,
                "task_group_id": ex.task_group_id,
                "resolution_type": ex.resolution_type,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from experiment captures")
    parser.add_argument("--data", type=Path, default=Path("experiments"),
                        help="Directory with experiment JSON files")
    parser.add_argument("--split", default="train",
                        help="Which split to prepare (default: train)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path")
    parser.add_argument("--stats", action="store_true",
                        help="Print statistics only (no export)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum sorting")
    parser.add_argument("--include-synthetic", action="store_true",
                        help="Include depth=1 synthetic experiments")
    args = parser.parse_args()

    config = DataPrepConfig(
        data_dir=args.data,
        split=args.split,
        curriculum_sort=not args.no_curriculum,
        include_synthetic=args.include_synthetic,
        output_path=args.output,
    )

    console.print(f"[bold]SFT Data Prep[/bold] | dir: {config.data_dir} | split: {config.split}")

    dataset = prepare_training_data(config)

    # Print stats
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Eligible examples: {dataset.stats['total_eligible']}")
    console.print(f"  Skipped (token budget): {dataset.stats['skipped_token_budget']}")
    console.print(f"  Embedding diversity (std): {dataset.stats['embedding_diversity_std']:.4f}")
    console.print(f"  Curriculum sorted: {dataset.stats['curriculum_sorted']}")

    if dataset.stats["rejected"]:
        console.print(f"\n  [dim]Rejected:[/dim]")
        for reason, count in sorted(dataset.stats["rejected"].items()):
            console.print(f"    {reason}: {count}", style="dim")

    if dataset.stats["domain_distribution"]:
        console.print(f"\n  [dim]Domains:[/dim]")
        for domain, count in sorted(dataset.stats["domain_distribution"].items(),
                                     key=lambda x: -x[1]):
            pct = count / max(dataset.stats["total_eligible"], 1) * 100
            console.print(f"    {domain}: {count} ({pct:.0f}%)", style="dim")

    if dataset.stats["resolution_type_distribution"]:
        console.print(f"\n  [dim]Resolution types:[/dim]")
        for rtype, count in sorted(dataset.stats["resolution_type_distribution"].items(),
                                    key=lambda x: -x[1]):
            console.print(f"    {rtype}: {count}", style="dim")

    # Export if requested
    if args.output and not args.stats:
        count = export_jsonl(dataset, args.output)
        console.print(f"\n  Exported {count} examples to [bold]{args.output}[/bold]")
    elif not args.stats:
        console.print(f"\n  [dim]Use --output to export, or --stats for stats only[/dim]")


if __name__ == "__main__":
    main()

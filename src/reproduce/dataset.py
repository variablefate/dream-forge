"""B.1 Full TriviaQA dataset for H-neuron detection.

Downloads TriviaQA rc.nocontext, generates 10 samples per question via
Qwen3.5-9B 8-bit, judges answers via normalized alias matching, filters
for consistency, and extracts CETT activations.

Fully resumable — caches samples as JSONL and activations as .npy.

Run:
    uv run python -m src.reproduce.dataset                      # full 5000 questions
    uv run python -m src.reproduce.dataset --num-questions 500  # quick subset
    uv run python -m src.reproduce.dataset --skip-activations   # sampling only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from src.engine.model_loader import load_model

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_QUESTIONS = 5000
SAMPLES_PER_QUESTION = 10
MAX_NEW_TOKENS = 50
CONSISTENCY_THRESHOLD_CORRECT = 10   # all 10 must be correct
CONSISTENCY_THRESHOLD_INCORRECT = 7  # at least 7/10 incorrect (asymmetric)
TARGET_PER_CLASS = 500               # goal: 500 correct + 500 incorrect

SAMPLES_DIR = Path("data/trivia_samples")
ACTIVATIONS_DIR = Path("data/activations")
RESULTS_DIR = Path("models")

console = Console()


# ── Answer judging ─────────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Normalize for matching: lowercase, strip articles/punctuation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split()).strip()


def judge_answer(response: str, aliases: list[str]) -> bool:
    """Check if response contains any gold answer alias."""
    norm_response = normalize_answer(response)
    for alias in aliases:
        norm_alias = normalize_answer(alias)
        if norm_alias and norm_alias in norm_response:
            return True
    return False


# ── Dataset download ───────────────────────────────────────────────────────────

def download_trivia_questions(num_questions: int) -> list[dict]:
    """Download TriviaQA rc.nocontext validation split."""
    from datasets import load_dataset

    console.print(f"  Downloading TriviaQA rc.nocontext ({num_questions} questions)...", style="dim")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation",
                       trust_remote_code=True)

    questions = []
    for i, item in enumerate(ds):
        if i >= num_questions:
            break
        questions.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "aliases": item["answer"]["aliases"],
            "normalized_aliases": item["answer"]["normalized_aliases"],
        })

    console.print(f"  Got {len(questions)} questions", style="dim")
    return questions


# ── Sampling ───────────────────────────────────────────────────────────────────

def generate_samples(
    model, tokenizer, questions: list[dict],
    cache_dir: Path, batch_name: str = "full",
) -> list[dict]:
    """Generate SAMPLES_PER_QUESTION samples per question. Fully resumable."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{batch_name}_samples.jsonl"

    # Load cache
    cached: dict[str, dict] = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                cached[entry["question_id"]] = entry
        console.print(f"  Resuming: {len(cached)}/{len(questions)} already sampled", style="dim")

    remaining = [q for q in questions if q["question_id"] not in cached]
    if not remaining:
        console.print("  All questions already sampled", style="dim")
        return list(cached.values())

    console.print(f"  Sampling {SAMPLES_PER_QUESTION}x for {len(remaining)} questions "
                   f"(num_return_sequences={SAMPLES_PER_QUESTION})...")

    t_start = time.monotonic()
    with open(cache_file, "a", encoding="utf-8") as f:
        for i, q in enumerate(remaining):
            messages = [{"role": "user", "content": q["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, enable_thinking=False,
                tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
            prompt_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=SAMPLES_PER_QUESTION,
                )

            samples = []
            for seq in outputs:
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                correct = judge_answer(text, q["aliases"])
                samples.append({"text": text, "correct": correct})

            entry = {
                "question_id": q["question_id"],
                "question": q["question"],
                "aliases": q["aliases"],
                "samples": samples,
                "num_correct": sum(1 for s in samples if s["correct"]),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            cached[q["question_id"]] = entry

            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t_start
                per_q = elapsed / (i + 1)
                eta = per_q * (len(remaining) - i - 1)
                n_correct = sum(1 for e in list(cached.values())[-50:]
                                if e["num_correct"] == SAMPLES_PER_QUESTION)
                n_incorrect = sum(1 for e in list(cached.values())[-50:]
                                  if (SAMPLES_PER_QUESTION - e["num_correct"]) >= CONSISTENCY_THRESHOLD_INCORRECT)
                console.print(
                    f"    [{i+1}/{len(remaining)}] "
                    f"{per_q:.1f}s/q | ETA {eta/60:.0f}m | "
                    f"last 50: {n_correct} all-correct, {n_incorrect} mostly-incorrect",
                    style="dim")

    return list(cached.values())


# ── Filtering ──────────────────────────────────────────────────────────────────

@dataclass
class FilteredData:
    correct: list[dict] = field(default_factory=list)
    incorrect: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def filter_for_consistency(samples: list[dict]) -> FilteredData:
    """Filter for consistently correct/incorrect, balance classes."""
    correct = []
    incorrect = []

    for entry in samples:
        n = entry["num_correct"]
        if n == CONSISTENCY_THRESHOLD_CORRECT:
            correct.append(entry)
        elif (SAMPLES_PER_QUESTION - n) >= CONSISTENCY_THRESHOLD_INCORRECT:
            incorrect.append(entry)

    n_balanced = min(len(correct), len(incorrect), TARGET_PER_CLASS)

    return FilteredData(
        correct=correct[:n_balanced],
        incorrect=incorrect[:n_balanced],
        stats={
            "total_questions": len(samples),
            "all_correct": len(correct),
            "mostly_incorrect": len(incorrect),
            "balanced_per_class": n_balanced,
            "thresholds": {
                "correct": f"{CONSISTENCY_THRESHOLD_CORRECT}/{SAMPLES_PER_QUESTION}",
                "incorrect": f">={CONSISTENCY_THRESHOLD_INCORRECT}/{SAMPLES_PER_QUESTION} incorrect",
            },
        },
    )


# ── CETT Activation extraction ────────────────────────────────────────────────

def extract_and_save_activations(
    model, tokenizer, entries: list[dict], label: str, cache_dir: Path,
) -> np.ndarray:
    """Extract CETT activations, cache per-sample .npy files. Returns [N, features]."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Precompute weight norms (constant across samples)
    layer_names = sorted([
        name for name, _ in model.named_modules()
        if name.endswith("down_proj") and "mlp" in name
    ])
    weight_norms: dict[str, torch.Tensor] = {}
    for name in layer_names:
        mod = dict(model.named_modules())[name]
        weight_norms[name] = mod.weight.float().norm(dim=0).cpu()

    num_neurons = weight_norms[layer_names[0]].shape[0]
    expected_features = len(layer_names) * num_neurons
    console.print(f"  {len(layer_names)} layers x {num_neurons} neurons = {expected_features} features",
                   style="dim")

    all_cett = []
    t_start = time.monotonic()

    for idx, entry in enumerate(entries):
        cache_file = cache_dir / f"{label}_{entry['question_id']}.npy"

        # Resume from cache
        if cache_file.exists():
            cett = np.load(cache_file)
            if cett.shape[0] == expected_features:
                all_cett.append(cett)
                continue

        sample_text = entry["samples"][0]["text"]
        messages = [{"role": "user", "content": entry["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False,
            tokenize=False, add_generation_prompt=True)
        full_text = prompt + sample_text
        inputs = tokenizer(full_text, return_tensors="pt").to(next(model.parameters()).device)

        layer_io: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        handles = []
        for name in layer_names:
            mod = dict(model.named_modules())[name]
            def hook(m, inp, out, n=name):
                layer_io[n] = (inp[0].detach(), out.detach())
            handles.append(mod.register_forward_hook(hook))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        layer_cetts = []
        for name in layer_names:
            if name not in layer_io:
                layer_cetts.append(np.zeros(num_neurons, dtype=np.float32))
                continue
            act_in, act_out = layer_io[name]
            act_in = act_in.squeeze(0).float()
            act_out = act_out.squeeze(0).float()
            wn = weight_norms[name].to(act_in.device)
            act_abs = act_in.abs()
            out_norms = act_out.norm(dim=-1, keepdim=True)
            cett = (act_abs * wn) / (out_norms + 1e-8)
            layer_cetts.append(cett.mean(dim=0).cpu().numpy())

        layer_io.clear()

        sample_cett = np.concatenate(layer_cetts)
        np.save(cache_file, sample_cett)
        all_cett.append(sample_cett)

        if (idx + 1) % 50 == 0:
            elapsed = time.monotonic() - t_start
            per_s = elapsed / (idx + 1)
            eta = per_s * (len(entries) - idx - 1)
            console.print(f"    [{idx+1}/{len(entries)}] {label} | "
                           f"{per_s:.1f}s/sample | ETA {eta/60:.0f}m", style="dim")

    return np.stack(all_cett)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="B.1 Full TriviaQA dataset for H-neuron detection")
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS,
                        help=f"Number of questions (default: {NUM_QUESTIONS})")
    parser.add_argument("--skip-activations", action="store_true",
                        help="Only do sampling + filtering, skip CETT extraction")
    args = parser.parse_args()

    console.print(f"[bold]B.1 Full Dataset[/bold] | {args.num_questions} questions | "
                   f"{SAMPLES_PER_QUESTION} samples each")

    # Phase 1: Load model
    console.print("\n[bold cyan]Phase 1: Model Loading[/bold cyan]")
    model, tokenizer = load_model()

    # Phase 2: Download + sample
    console.print("\n[bold cyan]Phase 2: TriviaQA Sampling[/bold cyan]")
    questions = download_trivia_questions(args.num_questions)
    samples = generate_samples(model, tokenizer, questions, SAMPLES_DIR, batch_name="full")

    # Phase 3: Filter
    console.print("\n[bold cyan]Phase 3: Consistency Filtering[/bold cyan]")
    filtered = filter_for_consistency(samples)

    table = Table(title="Filtering Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for k, v in filtered.stats.items():
        if k != "thresholds":
            table.add_row(k, str(v))
    table.add_row("correct threshold", filtered.stats["thresholds"]["correct"])
    table.add_row("incorrect threshold", filtered.stats["thresholds"]["incorrect"])
    console.print(table)

    if filtered.stats["balanced_per_class"] < 100:
        console.print("[yellow]Warning: fewer than 100 per class. "
                       "Consider running with more questions.[/yellow]")

    # Phase 4: CETT extraction (optional)
    if not args.skip_activations:
        console.print("\n[bold cyan]Phase 4: CETT Activation Extraction[/bold cyan]")
        X_correct = extract_and_save_activations(
            model, tokenizer, filtered.correct, "correct", ACTIVATIONS_DIR)
        X_incorrect = extract_and_save_activations(
            model, tokenizer, filtered.incorrect, "incorrect", ACTIVATIONS_DIR)

        X = np.vstack([X_correct, X_incorrect])
        y = np.concatenate([np.zeros(len(X_correct)), np.ones(len(X_incorrect))])

        # Save combined dataset
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(RESULTS_DIR / "cett_features.npy", X)
        np.save(RESULTS_DIR / "cett_labels.npy", y)
        console.print(f"  Saved: {X.shape} features, {y.shape} labels")

    # Save filtered question IDs for reproducibility
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filtered_ids = {
        "correct": [e["question_id"] for e in filtered.correct],
        "incorrect": [e["question_id"] for e in filtered.incorrect],
        "stats": filtered.stats,
    }
    (RESULTS_DIR / "filtered_dataset.json").write_text(json.dumps(filtered_ids, indent=2))

    console.print(f"\n[bold green]Done.[/bold green] "
                   f"{filtered.stats['balanced_per_class']} samples per class.")


if __name__ == "__main__":
    main()

"""B.0b Fast sanity gate — 200-question TriviaQA pilot.

Tests both H-neuron probe roles (detector + intervention) on a small sample
before committing to the full Lane B build.

Decision matrix:
  Both signals   → proceed with full Lane B
  Detector only  → Lane C uses detection for training/eval aids only
  Neither        → pivot to LoRA-only (no H-neuron)

Run:
    uv run python -m src.reproduce.sanity_gate
    uv run python -m src.reproduce.sanity_gate --num-questions 50  # quick test
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-9B"
NUM_QUESTIONS = 200
SAMPLES_PER_QUESTION = 10
MAX_NEW_TOKENS = 50
CONSISTENCY_THRESHOLD_CORRECT = 10   # all 10 must be correct
CONSISTENCY_THRESHOLD_INCORRECT = 7  # at least 7/10 incorrect (asymmetric)
MIN_PER_CLASS = 25  # minimum samples per class for sanity gate (50 ideal)

SAMPLES_DIR = Path("data/trivia_samples")
ACTIVATIONS_DIR = Path("data/activations")
MODELS_DIR = Path("models")

console = Console()


# ── Answer judging ─────────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Normalize answer for matching: lowercase, strip articles/punctuation."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def judge_answer(response: str, aliases: list[str]) -> bool:
    """Check if response contains any of the gold answer aliases."""
    norm_response = normalize_answer(response)
    for alias in aliases:
        norm_alias = normalize_answer(alias)
        if norm_alias and norm_alias in norm_response:
            return True
    return False


# ── CETT computation ───────────────────────────────────────────────────────────

def compute_cett(
    activations: torch.Tensor,  # [seq, neurons] — input to down_proj
    weight: torch.Tensor,       # [out, neurons] — down_proj weight
    output: torch.Tensor,       # [seq, out] — down_proj output
) -> np.ndarray:
    """Compute CETT for a single layer.

    CETT = (abs(activations) * weight_norms) / (output_norms + eps)
    Aggregated by mean across token dimension.
    Returns: [neurons] array.
    """
    # Per-column norms of weight matrix (one per neuron)
    weight_norms = weight.float().norm(dim=0)  # [neurons]

    # Per-token output norms
    output_norms = output.float().norm(dim=-1, keepdim=True)  # [seq, 1]

    # CETT per token per neuron
    act_abs = activations.float().abs()  # [seq, neurons]
    cett = (act_abs * weight_norms.unsqueeze(0)) / (output_norms + 1e-8)  # [seq, neurons]

    # Mean across tokens
    return cett.mean(dim=0).cpu().numpy()  # [neurons]


# ── Model loading (shared with compat_check) ──────────────────────────────────

def load_model():
    """Load Qwen3.5-9B 8-bit with Unsloth (preferred) or PEFT fallback.

    Returns (model, tokenizer). Handles vision encoder swap and fused CE patch.
    """
    # Try Unsloth
    try:
        from unsloth import FastLanguageModel
        console.print("  Loading via Unsloth...", style="dim")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_8bit=True,
            dtype=None,
        )

        # Swap multimodal processor for text-only tokenizer
        has_vision = any("visual" in n or "vision" in n
                         for n, _ in model.named_modules())
        if has_vision:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            if hasattr(model, "visual"):
                del model.visual
            elif hasattr(model, "model") and hasattr(model.model, "visual"):
                del model.model.visual
            torch.cuda.empty_cache()
            gc.collect()

        # Patch fused CE VRAM check
        try:
            import functools
            from unsloth_zoo.fused_losses import cross_entropy_loss

            @functools.cache
            def _patched(vocab_size, target_gb=None):
                if target_gb is None:
                    target_gb = 1.0
                return (vocab_size * 4 / 1024 / 1024 / 1024) / target_gb / 4

            cross_entropy_loss._get_chunk_multiplier = _patched
        except Exception:
            pass

        console.print(f"  Loaded via Unsloth | VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB",
                       style="dim")
        return model, tokenizer

    except ImportError:
        console.print("  Unsloth not available, trying PEFT...", style="dim")

    # PEFT fallback
    from transformers import AutoTokenizer, BitsAndBytesConfig
    try:
        from transformers import Qwen3_5ForCausalLM as ModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelClass

    model = ModelClass.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    console.print(f"  Loaded via PEFT | VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB",
                   style="dim")
    return model, tokenizer


# ── Sampling ───────────────────────────────────────────────────────────────────

def download_trivia_questions(num_questions: int) -> list[dict]:
    """Download TriviaQA rc.nocontext and return question dicts."""
    from datasets import load_dataset

    console.print(f"  Downloading TriviaQA (rc.nocontext, {num_questions} questions)...", style="dim")
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


def generate_samples(
    model, tokenizer, questions: list[dict], cache_dir: Path,
) -> list[dict]:
    """Generate SAMPLES_PER_QUESTION samples per question. Resume from cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "sanity_gate_samples.jsonl"

    # Load existing cache
    cached: dict[str, dict] = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cached[entry["question_id"]] = entry
        console.print(f"  Resuming: {len(cached)} questions already sampled", style="dim")

    remaining = [q for q in questions if q["question_id"] not in cached]
    if not remaining:
        console.print("  All questions already sampled", style="dim")
        return list(cached.values())

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"  Generating {SAMPLES_PER_QUESTION} samples each for {len(remaining)} questions...")

    with open(cache_file, "a", encoding="utf-8") as f:
        for i, q in enumerate(remaining):
            messages = [{"role": "user", "content": q["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, enable_thinking=False,
                tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            samples = []
            for _ in range(SAMPLES_PER_QUESTION):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=1.0,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True,
                    )
                text = tokenizer.decode(
                    out[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True).strip()
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

            if (i + 1) % 10 == 0:
                console.print(
                    f"    [{i+1}/{len(remaining)}] "
                    f"Last: {entry['num_correct']}/{SAMPLES_PER_QUESTION} correct",
                    style="dim")

    return list(cached.values())


# ── Consistency filtering ──────────────────────────────────────────────────────

@dataclass
class FilteredData:
    correct: list[dict] = field(default_factory=list)    # all 10 correct
    incorrect: list[dict] = field(default_factory=list)  # >= threshold incorrect
    stats: dict = field(default_factory=dict)


def filter_for_consistency(samples: list[dict]) -> FilteredData:
    """Filter for consistently correct/incorrect questions."""
    correct = []
    incorrect = []

    for entry in samples:
        n_correct = entry["num_correct"]
        if n_correct == CONSISTENCY_THRESHOLD_CORRECT:
            correct.append(entry)
        elif (SAMPLES_PER_QUESTION - n_correct) >= CONSISTENCY_THRESHOLD_INCORRECT:
            incorrect.append(entry)

    # Balance: take min of both classes
    n = min(len(correct), len(incorrect))

    stats = {
        "total_questions": len(samples),
        "all_correct": len(correct),
        "mostly_incorrect": len(incorrect),
        "balanced_per_class": n,
    }

    return FilteredData(
        correct=correct[:n],
        incorrect=incorrect[:n],
        stats=stats,
    )


# ── Activation extraction ─────────────────────────────────────────────────────

def extract_activations(
    model, tokenizer, entries: list[dict], label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract CETT activations for a set of entries.

    Uses one representative sample per question (first sample).
    Returns (cett_array [N, 32*neurons], labels [N]).
    """
    all_cett = []

    # Register hooks to capture down_proj inputs and outputs
    for idx, entry in enumerate(entries):
        sample_text = entry["samples"][0]["text"]  # use first sample

        messages = [{"role": "user", "content": entry["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False,
            tokenize=False, add_generation_prompt=True)
        # Append the model's response for activation extraction
        full_text = prompt + sample_text

        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

        # Collect activations from all down_proj layers
        layer_activations: dict[str, dict] = {}
        handles = []

        for name, mod in model.named_modules():
            if name.endswith("down_proj") and "mlp" in name:
                def hook(m, inp, out, n=name):
                    layer_activations[n] = {
                        "input": inp[0].detach(),
                        "output": out.detach(),
                        "weight": m.weight.detach(),
                    }
                handles.append(mod.register_forward_hook(hook))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Compute CETT per layer, concatenate
        layer_cetts = []
        for name in sorted(layer_activations.keys()):
            data = layer_activations[name]
            cett = compute_cett(data["input"], data["weight"], data["output"])
            layer_cetts.append(cett)

        if layer_cetts:
            sample_cett = np.concatenate(layer_cetts)  # [32 * neurons]
            all_cett.append(sample_cett)

        layer_activations.clear()

        if (idx + 1) % 10 == 0:
            console.print(f"    [{idx+1}/{len(entries)}] {label} activations extracted", style="dim")

    cett_array = np.stack(all_cett)  # [N, features]
    labels = np.ones(len(entries)) if label == "incorrect" else np.zeros(len(entries))
    return cett_array, labels


# ── Probe training ─────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    accuracy: float
    num_features: int
    num_positive_weights: int  # for L1: H-neuron candidates
    sparsity: float  # fraction of neurons identified as H-neurons
    probe: object  # sklearn model


def train_detector_probe(X: np.ndarray, y: np.ndarray) -> ProbeResult:
    """L2 1-vs-1 detector probe. Best for classification accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(
        penalty="l2", C=1.0,
        solver="lbfgs", max_iter=1000,
        random_state=42,
    )

    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    clf.fit(X, y)

    return ProbeResult(
        accuracy=scores.mean(),
        num_features=X.shape[1],
        num_positive_weights=int((clf.coef_[0] > 0).sum()),
        sparsity=(clf.coef_[0] > 0).sum() / X.shape[1],
        probe=clf,
    )


def train_intervention_probe(X: np.ndarray, y: np.ndarray) -> ProbeResult:
    """L1 3-vs-1 intervention probe. Sparse — identifies H-neuron candidates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Grid search over C for sparsity/accuracy tradeoff
    best_score = 0
    best_clf = None
    best_c = None

    for c_val in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(
            penalty="l1", C=c_val,
            solver="liblinear", max_iter=1000,
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_clf = clf
            best_c = c_val

    best_clf.fit(X, y)
    n_positive = int((best_clf.coef_[0] > 0).sum())

    return ProbeResult(
        accuracy=best_score,
        num_features=X.shape[1],
        num_positive_weights=n_positive,
        sparsity=n_positive / X.shape[1],
        probe=best_clf,
    )


# ── Intervention test (alpha sweep) ───────────────────────────────────────────

def alpha_sweep(
    model, tokenizer,
    incorrect_entries: list[dict],
    h_neuron_indices: list[tuple[int, int]],  # (layer_idx, neuron_idx)
) -> dict[float, dict]:
    """Quick alpha sweep: scale H-neuron activations and measure impact."""
    results = {}

    for alpha in [0.0, 0.3, 0.5]:
        # Register intervention hooks
        handles = []
        layer_names = sorted([
            name for name, _ in model.named_modules()
            if name.endswith("down_proj") and "mlp" in name
        ])

        # Build per-layer neuron sets
        layer_neurons: dict[int, list[int]] = {}
        for layer_idx, neuron_idx in h_neuron_indices:
            layer_neurons.setdefault(layer_idx, []).append(neuron_idx)

        for layer_idx, neurons in layer_neurons.items():
            if layer_idx < len(layer_names):
                mod = dict(model.named_modules())[layer_names[layer_idx]]
                neuron_tensor = torch.tensor(neurons, device=model.device)

                def hook(m, inp, out, nt=neuron_tensor, a=alpha):
                    inp[0][:, :, nt] *= a
                    return None

                handles.append(mod.register_forward_hook(hook))

        # Evaluate on incorrect entries
        correct_after = 0
        total = min(len(incorrect_entries), 20)  # quick sweep on subset

        for entry in incorrect_entries[:total]:
            messages = [{"role": "user", "content": entry["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, enable_thinking=False,
                tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.1, do_sample=False,
                )
            text = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True).strip()

            if judge_answer(text, entry["aliases"]):
                correct_after += 1

        for h in handles:
            h.remove()

        results[alpha] = {
            "correct_after": correct_after,
            "total": total,
            "accuracy": correct_after / total if total > 0 else 0,
        }

    return results


# ── Main gate ──────────────────────────────────────────────────────────────────

class SanityGate:
    def __init__(self, num_questions: int = NUM_QUESTIONS):
        self.num_questions = num_questions
        self.model = None
        self.tokenizer = None
        self.t0 = time.monotonic()

    def run(self) -> bool:
        console.print(Panel.fit(
            f"[bold]B.0b Sanity Gate[/bold]\n"
            f"Model: {MODEL_ID} | Questions: {self.num_questions}\n"
            f"Samples per question: {SAMPLES_PER_QUESTION}",
            title="dream-forge"))

        # Phase 1: Load model
        console.print("\n[bold cyan]Phase 1: Model Loading[/bold cyan]")
        self.model, self.tokenizer = load_model()

        # Phase 2: Download + sample
        console.print("\n[bold cyan]Phase 2: TriviaQA Sampling[/bold cyan]")
        questions = download_trivia_questions(self.num_questions)
        samples = generate_samples(
            self.model, self.tokenizer, questions, SAMPLES_DIR)

        # Phase 3: Filter
        console.print("\n[bold cyan]Phase 3: Consistency Filtering[/bold cyan]")
        filtered = filter_for_consistency(samples)
        console.print(f"  Total: {filtered.stats['total_questions']} | "
                       f"All correct: {filtered.stats['all_correct']} | "
                       f"Mostly incorrect: {filtered.stats['mostly_incorrect']} | "
                       f"Balanced per class: {filtered.stats['balanced_per_class']}")

        if filtered.stats["balanced_per_class"] < MIN_PER_CLASS:
            console.print(
                f"[red]Insufficient data: {filtered.stats['balanced_per_class']} per class "
                f"(need {MIN_PER_CLASS}). Try more questions (--num-questions).[/red]")
            return self._decide("insufficient_data", filtered.stats, None, None, None)

        # Phase 4: Extract activations
        console.print("\n[bold cyan]Phase 4: Activation Extraction (CETT)[/bold cyan]")
        X_correct, y_correct = extract_activations(
            self.model, self.tokenizer, filtered.correct, "correct")
        X_incorrect, y_incorrect = extract_activations(
            self.model, self.tokenizer, filtered.incorrect, "incorrect")

        X = np.vstack([X_correct, X_incorrect])
        y = np.concatenate([y_correct, y_incorrect])
        console.print(f"  Feature matrix: {X.shape} | Labels: {y.shape}")

        # Phase 5: Detector probe (L2 1-vs-1)
        console.print("\n[bold cyan]Phase 5: Detector Probe (L2 1-vs-1)[/bold cyan]")
        detector = train_detector_probe(X, y)
        console.print(f"  Accuracy: {detector.accuracy:.1%} | "
                       f"Positive weights: {detector.num_positive_weights}")

        # Phase 6: Intervention probe (L1 3-vs-1)
        console.print("\n[bold cyan]Phase 6: Intervention Probe (L1 3-vs-1)[/bold cyan]")
        intervention = train_intervention_probe(X, y)
        console.print(f"  Accuracy: {intervention.accuracy:.1%} | "
                       f"H-neuron candidates: {intervention.num_positive_weights} | "
                       f"Sparsity: {intervention.sparsity:.4%}")

        # Phase 7: Alpha sweep (if intervention looks promising)
        sweep_results = None
        if intervention.sparsity < 0.001 and intervention.num_positive_weights > 0:
            console.print("\n[bold cyan]Phase 7: Alpha Sweep[/bold cyan]")

            # Extract H-neuron indices from intervention probe
            coef = intervention.probe.coef_[0]
            h_indices = []
            total_neurons_per_layer = X.shape[1] // 32  # assuming 32 layers
            for flat_idx in np.where(coef > 0)[0]:
                layer_idx = flat_idx // total_neurons_per_layer
                neuron_idx = flat_idx % total_neurons_per_layer
                h_indices.append((int(layer_idx), int(neuron_idx)))

            console.print(f"  Testing {len(h_indices)} H-neurons across {len(set(l for l, _ in h_indices))} layers")
            sweep_results = alpha_sweep(
                self.model, self.tokenizer, filtered.incorrect, h_indices)

            for alpha, res in sorted(sweep_results.items()):
                console.print(f"  alpha={alpha:.1f}: {res['correct_after']}/{res['total']} correct "
                               f"({res['accuracy']:.1%})")
        else:
            console.print("\n[dim]Skipping alpha sweep (sparsity too high or no H-neurons found)[/dim]")

        return self._decide("complete", filtered.stats, detector, intervention, sweep_results)

    def _decide(
        self,
        status: str,
        filter_stats: dict,
        detector: ProbeResult | None,
        intervention: ProbeResult | None,
        sweep: dict | None,
    ) -> bool:
        elapsed = time.monotonic() - self.t0

        # Decision logic
        detector_ok = detector is not None and detector.accuracy > 0.55
        detector_good = detector is not None and detector.accuracy > 0.60
        intervention_ok = (
            intervention is not None
            and intervention.sparsity < 0.001
            and sweep is not None
            and any(r["accuracy"] > 0 for r in sweep.values() if r is not None)
        )

        if detector_good and intervention_ok:
            decision = "FULL_H_NEURON"
            color = "green"
            msg = "Both signals present. Proceed with full Lane B."
        elif detector_ok:
            decision = "DETECTOR_ONLY"
            color = "yellow"
            msg = "Detector works but intervention weak. Use detection for training/eval aids only."
        else:
            decision = "NO_H_NEURON"
            color = "red"
            msg = "Neither signal reliable. Pivot to LoRA-only (no H-neuron)."

        # Summary table
        table = Table(title="\nSanity Gate Results", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_column("Threshold")

        if detector:
            table.add_row("Detector accuracy", f"{detector.accuracy:.1%}",
                         "> 55% (some signal) / > 60% (useful)")
        if intervention:
            table.add_row("Intervention accuracy", f"{intervention.accuracy:.1%}", "> 55%")
            table.add_row("H-neuron sparsity", f"{intervention.sparsity:.4%}", "< 0.1%")
            table.add_row("H-neuron count", str(intervention.num_positive_weights), "sparse set")
        table.add_row("Samples per class", str(filter_stats.get("balanced_per_class", 0)),
                      f">= {MIN_PER_CLASS}")

        console.print(table)
        console.print(f"\n[bold {color}]Decision: {decision}[/bold {color}] — {msg}")
        console.print(f"[dim]Elapsed: {elapsed:.0f}s[/dim]")

        # Save results
        result = {
            "decision": decision,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_questions": self.num_questions,
            "filter_stats": filter_stats,
            "detector": {
                "accuracy": detector.accuracy,
                "num_positive_weights": detector.num_positive_weights,
            } if detector else None,
            "intervention": {
                "accuracy": intervention.accuracy,
                "sparsity": intervention.sparsity,
                "num_positive_weights": intervention.num_positive_weights,
            } if intervention else None,
            "alpha_sweep": {str(k): v for k, v in sweep.items()} if sweep else None,
        }

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = MODELS_DIR / "sanity_gate_result.json"
        result_path.write_text(json.dumps(result, indent=2))
        console.print(f"Results saved to [bold]{result_path}[/bold]")

        # Save probes if useful
        if detector and detector.accuracy > 0.55:
            import joblib
            joblib.dump(detector.probe, MODELS_DIR / "detector_probe_pilot.pkl")
            console.print(f"Detector probe saved to [bold]{MODELS_DIR / 'detector_probe_pilot.pkl'}[/bold]")

        return decision != "NO_H_NEURON"


def main():
    parser = argparse.ArgumentParser(
        description="B.0b Sanity gate — quick TriviaQA pilot for H-neuron detection")
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS,
                        help=f"Number of TriviaQA questions (default: {NUM_QUESTIONS})")
    args = parser.parse_args()

    gate = SanityGate(num_questions=args.num_questions)
    try:
        ok = gate.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

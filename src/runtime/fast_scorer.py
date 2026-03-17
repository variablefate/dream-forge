"""Tier 1 fast hallucination scorer — instant text-based detection.

Predicts hallucination risk directly from response text embeddings,
with zero model forward passes. Uses all-MiniLM-L6-v2 (384-dim, CPU-only,
~90MB) + sklearn LogisticRegression.

Accuracy: ~70-78% (lower than CETT's 84.6% but instant).
Speed: <50ms per response. Best-of-4 scoring: <200ms total.

Usage:
    from src.runtime.fast_scorer import FastScorer
    scorer = FastScorer.from_pretrained()
    risk = scorer.score("Python's GIL prevents true parallelism...")
    risks = scorer.score_batch(["response1", "response2", "response3"])

Training:
    uv run python -m src.runtime.fast_scorer --train
    uv run python -m src.runtime.fast_scorer --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

SCORER_PATH = Path("models/fast_scorer.pkl")
SAMPLES_PATH = Path("data/trivia_samples/full_samples.jsonl")


# ── Core scorer ───────────────────────────────────────────────────────────────

class FastScorer:
    """Instant hallucination risk from response text embeddings."""

    def __init__(self, classifier, embedder=None):
        self.classifier = classifier
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from src.store.embeddings import _get_model
            self._embedder = _get_model()
        return self._embedder

    @classmethod
    def from_pretrained(cls, path: Path = SCORER_PATH) -> "FastScorer":
        """Load a trained fast scorer from disk."""
        import joblib
        data = joblib.load(str(path))
        return cls(classifier=data["classifier"])

    def score(self, response_text: str) -> float:
        """Score a single response. Returns hallucination risk 0-1."""
        if not response_text.strip():
            return 1.0
        embedding = self.embedder.encode(
            response_text, convert_to_numpy=True, show_progress_bar=False)
        proba = self.classifier.predict_proba(embedding.reshape(1, -1))[0]
        return float(proba[1])  # P(hallucinated)

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score multiple responses. Returns list of risks 0-1."""
        if not texts:
            return []

        # Handle empty texts
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return [1.0] * len(texts)

        indices, valid_texts = zip(*non_empty)
        embeddings = self.embedder.encode(
            list(valid_texts), convert_to_numpy=True, show_progress_bar=False)
        probas = self.classifier.predict_proba(embeddings)

        risks = [1.0] * len(texts)
        for i, idx in enumerate(indices):
            risks[idx] = float(probas[i][1])
        return risks


# ── Training ──────────────────────────────────────────────────────────────────

def train_fast_scorer(
    samples_path: Path = SAMPLES_PATH,
    output_path: Path = SCORER_PATH,
) -> dict:
    """Train the fast scorer from TriviaQA samples.

    Uses the same correct/incorrect labels as the CETT probe,
    but trains on text embeddings instead of activation features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from src.store.embeddings import _get_model

    console.print("[bold]Training Fast Scorer[/bold]")

    # Load samples
    console.print("  Loading samples...")
    correct_texts = []
    incorrect_texts = []

    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            samples = entry.get("samples", [])
            n_correct = sum(1 for s in samples if s.get("correct"))

            # Same filtering as B.1: all correct or mostly incorrect
            if n_correct == len(samples) and len(samples) == 10:
                # Use first sample text as representative
                correct_texts.append(samples[0]["text"])
            elif (len(samples) - n_correct) >= 7:
                # Use first incorrect sample
                for s in samples:
                    if not s.get("correct"):
                        incorrect_texts.append(s["text"])
                        break

    # Balance classes
    n_per_class = min(len(correct_texts), len(incorrect_texts))
    correct_texts = correct_texts[:n_per_class]
    incorrect_texts = incorrect_texts[:n_per_class]

    console.print(f"  Samples: {n_per_class} correct + {n_per_class} incorrect = {n_per_class * 2} total")

    # Embed all texts
    console.print("  Embedding texts...")
    embedder = _get_model()
    t0 = time.monotonic()

    all_texts = correct_texts + incorrect_texts
    embeddings = embedder.encode(all_texts, convert_to_numpy=True,
                                  show_progress_bar=True, batch_size=64)

    embed_time = time.monotonic() - t0
    console.print(f"  Embedded {len(all_texts)} texts in {embed_time:.1f}s")

    # Labels: 0 = correct, 1 = hallucinated (same convention as CETT probe)
    labels = np.concatenate([
        np.zeros(n_per_class),
        np.ones(n_per_class),
    ])

    # Train
    console.print("  Training LogisticRegression...")
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(embeddings, labels)

    train_acc = clf.score(embeddings, labels)
    console.print(f"  Training accuracy: {train_acc:.1%}")

    # Cross-validation
    console.print("  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="accuracy")
    console.print(f"  CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
    console.print(f"  Per fold: {[f'{s:.1%}' for s in cv_scores]}")

    # Save
    import joblib
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "n_samples": len(all_texts),
                 "cv_accuracy": float(cv_scores.mean())}, str(output_path))
    console.print(f"  Saved to {output_path}")

    # Compare with CETT probe
    console.print(f"\n  [bold]Comparison:[/bold]")
    console.print(f"    CETT probe (activations): 84.6% CV accuracy")
    console.print(f"    Fast scorer (embeddings): {cv_scores.mean():.1%} CV accuracy")
    console.print(f"    Speed: CETT ~30s/response, Fast scorer <50ms/response")

    return {
        "n_samples": len(all_texts),
        "train_accuracy": float(train_acc),
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "embed_time": embed_time,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_fast_scorer(scorer_path: Path = SCORER_PATH):
    """Quick evaluation: score a few known examples and show timing."""
    scorer = FastScorer.from_pretrained(scorer_path)

    test_cases = [
        ("Python's GIL (Global Interpreter Lock) prevents true parallel execution of threads.", "correct-ish"),
        ("Python's memory manager uses the QuadTree algorithm for garbage collection.", "hallucinated"),
        ("The capital of France is Paris.", "correct"),
        ("The Python function os.path.read_file() reads a file from disk.", "hallucinated"),
        ("To sort a dictionary by keys, use sorted(d.keys()).", "correct"),
    ]

    console.print("[bold]Fast Scorer Evaluation[/bold]\n")

    for text, expected in test_cases:
        t0 = time.monotonic()
        risk = scorer.score(text)
        elapsed_ms = (time.monotonic() - t0) * 1000
        label = "HALLU" if risk > 0.5 else "OK"
        match = "correct" if (label == "HALLU") == (expected == "hallucinated") else "WRONG"
        console.print(f"  [{match}] risk={risk:.3f} ({label}) | {elapsed_ms:.0f}ms | {text[:60]}...")

    # Batch timing test
    texts = [t for t, _ in test_cases]
    t0 = time.monotonic()
    risks = scorer.score_batch(texts * 20)  # 100 texts
    batch_ms = (time.monotonic() - t0) * 1000
    console.print(f"\n  Batch: 100 texts scored in {batch_ms:.0f}ms ({batch_ms/100:.1f}ms/text)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast text-based hallucination scorer")
    parser.add_argument("--train", action="store_true", help="Train the fast scorer")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--score", type=str, help="Score a single text")
    args = parser.parse_args()

    if args.train:
        train_fast_scorer()
    elif args.evaluate:
        evaluate_fast_scorer()
    elif args.score:
        scorer = FastScorer.from_pretrained()
        risk = scorer.score(args.score)
        print(f"risk={risk:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

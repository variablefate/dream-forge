"""Evaluation metrics for dream-forge sleep/wake cycles.

Tracks across ablation variants and promotion cycles:
  - Accuracy rate
  - Hallucination rate
  - Calibration (ECE, Brier) when calibrator available
  - Per-domain performance
  - Response type distribution (normal / hedged / hard abstain)
  - Selective prediction (coverage, risk-at-coverage)
  - BPB (bits-per-byte, vocab-invariant loss)
  - Pass@N (for best-of-N evaluation)

Usage:
    from src.eval.metrics import EvalSuite
    suite = EvalSuite()
    suite.add("What is X?", prediction="X is Y", gold="X is Y", correct=True, domain="python")
    suite.add("What is Z?", prediction="I'm not sure...", gold="Z is W", correct=False, hedged=True)
    report = suite.compute()
    suite.print_report()
    suite.save("models/eval_results.json")
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


# ── Response type classification ───────────────────────────────────────────────

HEDGE_MARKERS = [
    "i'm not sure", "i'm not confident", "not fully confident",
    "i think", "i believe", "you should verify", "please verify",
    "may not be accurate", "take this with", "double-check",
    "not certain", "might be wrong", "could be incorrect",
]

ABSTAIN_MARKERS = [
    "i can't answer", "i cannot answer", "i don't know",
    "unable to answer", "not confident enough to answer",
    "i should not", "i refuse", "beyond my ability",
]


def classify_response(text: str) -> str:
    """Classify response as normal, hedged, or abstain."""
    lower = text.lower()
    for marker in ABSTAIN_MARKERS:
        if marker in lower:
            return "abstain"
    for marker in HEDGE_MARKERS:
        if marker in lower:
            return "hedged"
    return "normal"


# ── Core data types ────────────────────────────────────────────────────────────

@dataclass
class EvalExample:
    query: str
    prediction: str
    gold: str
    correct: bool
    domain: str = "general"
    response_type: str = ""        # auto-classified if empty
    confidence: float | None = None  # from detector probe or calibrator
    hallucinated: bool = False     # contains provably false statements
    hedged: bool = False           # override: force hedged classification
    abstained: bool = False        # override: force abstain classification


@dataclass
class EvalReport:
    # Core metrics
    accuracy: float = 0.0
    hallucination_rate: float = 0.0
    total: int = 0

    # Response type distribution
    normal_count: int = 0
    hedged_count: int = 0
    abstain_count: int = 0
    normal_accuracy: float = 0.0
    hedged_accuracy: float = 0.0

    # Selective prediction
    coverage: float = 0.0          # % answered (normal + hedged)
    risk_at_coverage: float = 0.0  # error rate on answered queries

    # Per-domain
    domain_accuracy: dict = field(default_factory=dict)

    # Calibration (only if confidence scores provided)
    ece: float | None = None
    brier: float | None = None

    # BPB (only if loss data provided)
    bpb: float | None = None

    # Pass@N (only if multi-sample data provided)
    pass_at: dict = field(default_factory=dict)  # {1: 0.71, 4: 0.86}


# ── BPB computation ────────────────────────────────────────────────────────────

def compute_bpb(ce_nats_sum: float, utf8_bytes_sum: int) -> float:
    """Compute bits-per-byte from accumulated CE (nats) and UTF-8 byte counts.

    BPB = sum(CE_nats) / (ln(2) * sum(utf8_bytes))

    This is the autoresearch methodology — vocab-invariant, comparable across
    models with different tokenizers. 0.7 BPB = "70% compression."
    Needed for the corrections sharing pipeline where contributors may use
    different model sizes (9B vs 4B) with different tokenizers.
    """
    if utf8_bytes_sum <= 0:
        return float("inf")
    return ce_nats_sum / (math.log(2) * utf8_bytes_sum)


# ── Pass@N computation ─────────────────────────────────────────────────────────

def compute_pass_at_n(
    num_correct_per_query: list[int],
    num_samples_per_query: list[int],
    n_values: list[int] = (1, 2, 4),
) -> dict[int, float]:
    """Compute pass@N using the unbiased estimator.

    pass@N = 1 - C(total-correct, N) / C(total, N)

    For each query with k correct out of total samples:
    P(at least 1 correct in N draws) = 1 - prod((total-k-i)/(total-i) for i in range(N))
    """
    results = {}
    for n in n_values:
        pass_rates = []
        for k, total in zip(num_correct_per_query, num_samples_per_query):
            if total < n:
                continue
            if k >= total:
                pass_rates.append(1.0)
                continue
            if k == 0:
                pass_rates.append(0.0)
                continue
            # Unbiased estimator: 1 - C(total-k, n) / C(total, n)
            fail_prob = 1.0
            for i in range(n):
                fail_prob *= (total - k - i) / (total - i)
            pass_rates.append(1.0 - max(0.0, fail_prob))

        results[n] = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
    return results


# ── Calibration metrics ────────────────────────────────────────────────────────

def compute_ece(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error — measures how well confidence matches accuracy.

    ECE = sum(|accuracy_bin - confidence_bin| * n_bin / N) over bins.
    Lower is better. 0.0 = perfectly calibrated.
    """
    if not confidences:
        return 0.0

    bins = defaultdict(list)
    for conf, correct in zip(confidences, correctness):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, float(correct)))

    ece = 0.0
    total = len(confidences)
    for bin_idx, entries in bins.items():
        if not entries:
            continue
        avg_conf = sum(c for c, _ in entries) / len(entries)
        avg_acc = sum(a for _, a in entries) / len(entries)
        ece += abs(avg_acc - avg_conf) * len(entries) / total

    return ece


def compute_brier(confidences: list[float], correctness: list[bool]) -> float:
    """Brier score — mean squared error of confidence vs outcome.

    Brier = mean((confidence - correct)^2). Lower is better. Range [0, 1].
    """
    if not confidences:
        return 0.0
    return sum((c - float(o)) ** 2 for c, o in zip(confidences, correctness)) / len(confidences)


# ── Main evaluation suite ──────────────────────────────────────────────────────

class EvalSuite:
    """Accumulates evaluation examples and computes all metrics."""

    def __init__(self):
        self.examples: list[EvalExample] = []

    def add(
        self,
        query: str,
        prediction: str,
        gold: str,
        correct: bool,
        domain: str = "general",
        confidence: float | None = None,
        hallucinated: bool = False,
        hedged: bool = False,
        abstained: bool = False,
    ):
        """Add a single evaluation example."""
        ex = EvalExample(
            query=query, prediction=prediction, gold=gold,
            correct=correct, domain=domain, confidence=confidence,
            hallucinated=hallucinated, hedged=hedged, abstained=abstained,
        )
        # Auto-classify response type (overrides take precedence)
        if abstained:
            ex.response_type = "abstain"
        elif hedged:
            ex.response_type = "hedged"
        else:
            ex.response_type = classify_response(prediction)
        self.examples.append(ex)

    def compute(self) -> EvalReport:
        """Compute all metrics from accumulated examples."""
        if not self.examples:
            return EvalReport()

        report = EvalReport(total=len(self.examples))

        # Accuracy + hallucination rate
        correct_count = sum(1 for e in self.examples if e.correct)
        halluc_count = sum(1 for e in self.examples if e.hallucinated)
        report.accuracy = correct_count / len(self.examples)
        report.hallucination_rate = halluc_count / len(self.examples)

        # Response type distribution
        by_type = defaultdict(list)
        for e in self.examples:
            by_type[e.response_type].append(e)

        report.normal_count = len(by_type["normal"])
        report.hedged_count = len(by_type["hedged"])
        report.abstain_count = len(by_type["abstain"])

        if by_type["normal"]:
            report.normal_accuracy = sum(1 for e in by_type["normal"] if e.correct) / len(by_type["normal"])
        if by_type["hedged"]:
            report.hedged_accuracy = sum(1 for e in by_type["hedged"] if e.correct) / len(by_type["hedged"])

        # Selective prediction (coverage = answered / total)
        answered = [e for e in self.examples if e.response_type != "abstain"]
        report.coverage = len(answered) / len(self.examples) if self.examples else 0.0
        if answered:
            report.risk_at_coverage = sum(1 for e in answered if not e.correct) / len(answered)

        # Per-domain accuracy
        by_domain = defaultdict(list)
        for e in self.examples:
            by_domain[e.domain].append(e)
        report.domain_accuracy = {
            domain: sum(1 for e in exs if e.correct) / len(exs)
            for domain, exs in by_domain.items()
        }

        # Calibration (only if confidence scores available)
        with_conf = [(e.confidence, e.correct) for e in self.examples if e.confidence is not None]
        if len(with_conf) >= 10:
            confs = [c for c, _ in with_conf]
            cors = [o for _, o in with_conf]
            report.ece = compute_ece(confs, cors)
            report.brier = compute_brier(confs, cors)

        return report

    def print_report(self, report: EvalReport | None = None):
        """Print a formatted report to console."""
        if report is None:
            report = self.compute()

        table = Table(title="Evaluation Report", show_lines=True)
        table.add_column("Metric", style="bold", min_width=25)
        table.add_column("Value", min_width=15)

        table.add_row("Total examples", str(report.total))
        table.add_row("Accuracy", f"{report.accuracy:.1%}")
        table.add_row("Hallucination rate", f"{report.hallucination_rate:.1%}")
        table.add_row("", "")
        table.add_row("Normal responses", f"{report.normal_count} ({report.normal_accuracy:.1%} accurate)")
        table.add_row("Hedged responses", f"{report.hedged_count} ({report.hedged_accuracy:.1%} accurate)")
        table.add_row("Abstained", str(report.abstain_count))
        table.add_row("", "")
        table.add_row("Coverage", f"{report.coverage:.1%}")
        table.add_row("Risk at coverage", f"{report.risk_at_coverage:.1%}")

        if report.ece is not None:
            table.add_row("", "")
            table.add_row("ECE (calibration)", f"{report.ece:.4f}")
            table.add_row("Brier score", f"{report.brier:.4f}")

        if report.bpb is not None:
            table.add_row("", "")
            table.add_row("BPB", f"{report.bpb:.4f}")

        if report.pass_at:
            table.add_row("", "")
            for n, rate in sorted(report.pass_at.items()):
                table.add_row(f"pass@{n}", f"{rate:.1%}")

        if report.domain_accuracy:
            table.add_row("", "")
            for domain, acc in sorted(report.domain_accuracy.items()):
                count = sum(1 for e in self.examples if e.domain == domain)
                table.add_row(f"  {domain}", f"{acc:.1%} (n={count})")

        console.print(table)

    def save(self, path: str | Path):
        """Save report as JSON."""
        report = self.compute()
        data = {
            "total": report.total,
            "accuracy": report.accuracy,
            "hallucination_rate": report.hallucination_rate,
            "response_types": {
                "normal": {"count": report.normal_count, "accuracy": report.normal_accuracy},
                "hedged": {"count": report.hedged_count, "accuracy": report.hedged_accuracy},
                "abstain": {"count": report.abstain_count},
            },
            "selective_prediction": {
                "coverage": report.coverage,
                "risk_at_coverage": report.risk_at_coverage,
            },
            "domain_accuracy": report.domain_accuracy,
        }
        if report.ece is not None:
            data["calibration"] = {"ece": report.ece, "brier": report.brier}
        if report.bpb is not None:
            data["bpb"] = report.bpb
        if report.pass_at:
            data["pass_at"] = {str(k): v for k, v in report.pass_at.items()}

        Path(path).write_text(json.dumps(data, indent=2))
        console.print(f"Report saved to [bold]{path}[/bold]")

"""C.5 Abstain policy — generate hedged training examples.

Teaches the model WHEN to hedge vs answer confidently. Labels come ONLY
from Tier 1a failures where ALL dream samples AND wake also fail.

Hedged examples use the same prompt templates as data_prep.py but with
hedged reference_solution text. They are baked INTO the capability LoRA
(not a separate adapter) so the behavior survives GGUF export.

Mixing ratio: ~10% of total SFT data (keep small to avoid over-hedging).
Non-regression guard: if abstain rate jumps >15% without proportional
accuracy gain, roll back.

Usage:
    from src.engine.abstain import generate_abstain_examples, AbstainConfig
    config = AbstainConfig(data_dir=Path("experiments"))
    examples = generate_abstain_examples(config, calibration_results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from src.engine.calibrate import CalibrationResult
    from src.engine.data_prep import PreparedExample

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

ABSTAIN_MIX_RATIO = 0.10    # 10% of total SFT data
MAX_ABSTAIN_RATE = 0.15      # roll back if abstain rate exceeds this

# Hedging templates — natural language, no special tokens.
# Must survive GGUF export and work in LM Studio.
HEDGE_TEMPLATES = [
    (
        "I'm not confident enough to answer this accurately. "
        "Here's my best attempt, but please verify:\n\n{attempt}"
    ),
    (
        "I'm uncertain about this answer and recommend verifying it "
        "independently. My best understanding is:\n\n{attempt}"
    ),
    (
        "This is outside my area of confidence. "
        "Here's what I think, but I could be wrong:\n\n{attempt}"
    ),
]

HARD_ABSTAIN_TEMPLATES = [
    (
        "I don't have enough confidence to provide a reliable answer "
        "to this question. I'd recommend consulting documentation or "
        "a more authoritative source."
    ),
    (
        "I'm not able to answer this reliably. The problem involves "
        "specifics I'm uncertain about, and guessing could be harmful."
    ),
]


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class AbstainConfig:
    data_dir: Path = Path("experiments")
    mix_ratio: float = ABSTAIN_MIX_RATIO
    max_rate: float = MAX_ABSTAIN_RATE
    # Minimum dream samples that must also fail for an "should abstain" label
    min_dream_failures: int = 3


# ── Label generation ──────────────────────────────────────────────────────────

def identify_abstain_candidates(
    experiments: list[dict],
    calibration_results: dict[str, "CalibrationResult"],
    min_dream_failures: int = 3,
) -> list[dict]:
    """Identify experiments where the model should abstain.

    Criteria:
    - Wake output failed (test_results.passed == False, or high detector risk)
    - At least min_dream_failures dream samples also scored high risk
    - The experiment has a reference_solution (so we know the right answer exists)

    Returns experiments that qualify as "should abstain" examples.
    """
    candidates = []

    for exp in experiments:
        exp_id = str(exp.get("id", ""))

        # Must have reference_solution (we need to know there IS a right answer)
        if not exp.get("reference_solution"):
            continue

        # Tier 1a: test_results must show failure
        test_results = exp.get("test_results")
        has_tier1a_fail = test_results and not test_results.get("passed", True)

        # Detector risk as supplementary signal
        cal = calibration_results.get(exp_id)
        if cal is None:
            continue

        wake_high_risk = cal.wake_risk > 0.7

        # Either Tier 1a failure OR consistently high detector risk
        if not has_tier1a_fail and not wake_high_risk:
            continue

        # Require enough dream samples to confirm model is truly stuck
        if len(cal.dream_scores) < min_dream_failures:
            continue

        all_dreams_risky = all(
            s.hallucination_risk > 0.6 for s in cal.dream_scores
        )

        if not all_dreams_risky:
            continue  # some dream samples are OK → model CAN answer, just inconsistent

        candidates.append(exp)

    return candidates


# ── Example generation ────────────────────────────────────────────────────────

def generate_abstain_examples(
    config: AbstainConfig,
    candidates: list[dict],
    total_training_examples: int,
    calibration_results: dict[str, "CalibrationResult"] | None = None,
) -> list["PreparedExample"]:
    """Generate hedged/abstain training examples from candidates.

    Uses the same prompt template as data_prep.py but with hedged
    reference_solution text. Respects the mix ratio cap.

    Args:
        config: Abstain policy config.
        candidates: Experiments identified as "should abstain" by
                    identify_abstain_candidates().
        total_training_examples: Current count of normal SFT examples.
                                 Abstain examples are capped at mix_ratio * total.
        calibration_results: Optional calibration data for selecting
                            "best wrong attempt" to include in hedged responses.

    Returns:
        List of PreparedExample with hedged reference_solution.
    """
    from src.engine.data_prep import PreparedExample, build_prompt

    max_abstain = max(1, int(total_training_examples * config.mix_ratio))
    examples = []

    for i, exp in enumerate(candidates):
        if len(examples) >= max_abstain:
            break

        system_msg, user_msg, _original_ref = build_prompt(exp)

        # Get the best wrong attempt from calibration data
        best_attempt = ""
        exp_id = str(exp.get("id", ""))
        if calibration_results and exp_id in calibration_results:
            cal = calibration_results[exp_id]
            # Pick the dream sample with lowest risk (least-bad attempt)
            if cal.dream_scores:
                best_dream = min(cal.dream_scores,
                                 key=lambda s: s.hallucination_risk)
                best_attempt = best_dream.text
            elif cal.wake_score:
                best_attempt = cal.wake_score.text

        # Alternate between hedge templates (with attempt) and hard abstain
        if best_attempt:
            template = HEDGE_TEMPLATES[i % len(HEDGE_TEMPLATES)]
            hedged_solution = template.format(attempt=best_attempt[:500])
        else:
            template = HARD_ABSTAIN_TEMPLATES[i % len(HARD_ABSTAIN_TEMPLATES)]
            hedged_solution = template

        examples.append(PreparedExample(
            system_msg=system_msg,
            user_msg=user_msg,
            reference_solution=hedged_solution,
            source_file=f"abstain_{exp.get('_source_file', 'generated')}",
            priority=1.0,  # high priority — abstain examples are rare and valuable
            domain=exp.get("_primary_domain", (exp.get("tags") or ["unknown"])[0]),
            task_group_id=exp.get("task_group_id", ""),
            split="train",
            resolution_type="abstain",
        ))

    console.print(
        f"  Abstain examples: {len(examples)} / {max_abstain} max "
        f"({len(candidates)} candidates)")

    return examples


# ── Non-regression guard ──────────────────────────────────────────────────────

@dataclass
class AbstainMetrics:
    """Track abstain rates across cycles for regression detection."""
    cycle: int
    total_responses: int
    abstain_count: int
    accuracy_on_answered: float

    @property
    def abstain_rate(self) -> float:
        if self.total_responses == 0:
            return 0.0
        return self.abstain_count / self.total_responses


def check_abstain_regression(
    current: AbstainMetrics,
    previous: AbstainMetrics | None,
    max_rate: float = MAX_ABSTAIN_RATE,
) -> tuple[bool, str]:
    """Check if abstain rate exceeds cap without proportional accuracy gain.

    Uses absolute rate ceiling (not just per-cycle increase) to prevent
    unbounded accumulation of abstain behavior across cycles.

    Returns (should_rollback, reason).
    """
    if previous is None:
        return False, "no previous cycle to compare"

    rate_increase = current.abstain_rate - previous.abstain_rate

    if rate_increase <= 0:
        return False, "abstain rate did not increase"

    # Absolute ceiling: abstain rate must not exceed max_rate
    # unless accuracy gain justifies it
    if current.abstain_rate > max_rate:
        accuracy_gain = current.accuracy_on_answered - previous.accuracy_on_answered
        if accuracy_gain < rate_increase * 0.5:
            return True, (
                f"abstain rate {current.abstain_rate:.1%} exceeds "
                f"{max_rate:.0%} cap and accuracy only improved "
                f"{accuracy_gain:.1%} — rollback recommended"
            )

    return False, "abstain rate within acceptable bounds"


def save_abstain_metrics(metrics: AbstainMetrics, path: Path):
    """Append metrics to a JSONL log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cycle": metrics.cycle,
        "total_responses": metrics.total_responses,
        "abstain_count": metrics.abstain_count,
        "abstain_rate": metrics.abstain_rate,
        "accuracy_on_answered": metrics.accuracy_on_answered,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_previous_metrics(path: Path) -> AbstainMetrics | None:
    """Load the most recent abstain metrics from the log."""
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        if not lines:
            return None
        last = json.loads(lines[-1])
        return AbstainMetrics(
            cycle=last["cycle"],
            total_responses=last["total_responses"],
            abstain_count=last["abstain_count"],
            accuracy_on_answered=last["accuracy_on_answered"],
        )
    except (json.JSONDecodeError, KeyError):
        return None

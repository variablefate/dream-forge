"""C.3 Teacher reference comparison — fully local, no API calls.

Compares Qwen's wake/dream outputs against the experiment's reference_solution.
Labels each output with a quality tier:
  Tier 1a: Executable verification (tests pass)
  Tier 1b: Build/lint artifacts
  Tier 2:  Embedding similarity
  Tier 3:  Qwen self-judge

Classification: correct / partially_correct / incorrect, tagged with tier.

Usage:
    from src.engine.compare import compare_outputs, CompareResult
    result = compare_outputs(model, tokenizer, wake_text, reference, experiment)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

SIMILARITY_CORRECT_THRESHOLD = 0.85
SIMILARITY_PARTIAL_THRESHOLD = 0.60
JUDGE_SYSTEM_PROMPT = "You are a careful code reviewer comparing two solutions for equivalence."


@dataclass
class CompareResult:
    classification: str  # "correct" | "partially_correct" | "incorrect"
    tier: str            # "tier_1a" | "tier_1b" | "tier_2" | "tier_3"
    confidence: float    # 0-1
    details: dict


# ── Tier 2: Embedding similarity ───────────────────────────────────────────────

def _embedding_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity via all-MiniLM-L6-v2."""
    from src.store.embeddings import embed_text, cosine_similarity
    vec_a = embed_text(text_a)
    vec_b = embed_text(text_b)
    return float(cosine_similarity(vec_a, vec_b))


# ── Tier 1a: Executable verification ──────────────────────────────────────────

def _tier_1a_verify(
    experiment: dict, prediction: str, detector_risk: float = 0.5,
) -> CompareResult | None:
    """Run the prediction in a git worktree and verify with tests + detector.

    Uses verify.py to: create worktree at commit_before, apply prediction,
    check syntax/imports/tests, combine with detector confidence.

    Returns CompareResult if verification was possible, None if the experiment
    lacks the fields needed for execution (no target file, no commit).
    """
    from src.engine.verify import verify_wake_output, VERIFICATION_THRESHOLD

    # Need pre_solution_context with a file path and a commit to check out
    pre_ctx = experiment.get("pre_solution_context") or []
    if not pre_ctx:
        return None
    first_ctx = pre_ctx[0] if isinstance(pre_ctx[0], dict) else {}
    target_file = first_ctx.get("path", "")
    commit = experiment.get("git_start_hash", experiment.get("repo_hash", ""))

    if not target_file or not commit:
        return None

    # Only attempt for code_change experiments
    if experiment.get("resolution_type") not in ("code_change", None):
        return None

    try:
        result = verify_wake_output(
            wake_text=prediction,
            experiment=experiment,
            detector_risk=detector_risk,
            run_tests=True,
        )
    except Exception:
        return None  # verification failed, fall through to Tier 2

    # Map composite score to classification
    if result.passed and result.tests_pass is True:
        classification = "correct"
    elif result.passed and result.tests_pass is None:
        classification = "partially_correct"  # passed on syntax+detector but no tests
    elif result.syntax_ok and result.imports_ok:
        classification = "partially_correct"  # builds but tests fail
    else:
        classification = "incorrect"

    return CompareResult(
        classification=classification,
        tier="tier_1a",
        confidence=result.score,
        details={
            "verification_score": result.score,
            "syntax_ok": result.syntax_ok,
            "imports_ok": result.imports_ok,
            "tests_pass": result.tests_pass,
            "detector_confidence": result.detector_confidence,
            "test_output": result.test_output[:300],
            "checks_run": result.checks_run,
            "error": result.error,
        },
    )


# ── Tier 1b: Build/lint verification ─────────────────────────────────────────

def _tier_1b_verify(
    experiment: dict, prediction: str,
) -> CompareResult | None:
    """Quick syntax + import check without full test execution.

    Lighter than Tier 1a — no git worktree, no test run. Just checks
    if the prediction compiles and the module imports cleanly.
    Falls through to Tier 2 if the experiment lacks needed fields.
    """
    from src.engine.verify import _check_syntax, _detect_language

    pre_ctx = experiment.get("pre_solution_context") or []
    if not pre_ctx:
        return None
    first_ctx = pre_ctx[0] if isinstance(pre_ctx[0], dict) else {}
    target_file = first_ctx.get("path", "")

    if not target_file:
        return None

    language = _detect_language(target_file)
    syntax_ok = _check_syntax(prediction, language)

    if not syntax_ok:
        return CompareResult(
            classification="incorrect",
            tier="tier_1b",
            confidence=0.1,
            details={"syntax_ok": False, "language": language},
        )

    # Syntax passes but we can't run full tests — partial signal
    return None  # fall through to Tier 2 for more info


# ── Tier 3: Qwen self-judge ───────────────────────────────────────────────────

def _tier_3_self_judge(
    model, tokenizer, prediction: str, reference: str,
) -> CompareResult:
    """Ask Qwen to compare its own output against the reference. Weakest signal."""
    import torch

    judge_prompt = (
        f"Compare these two solutions to the same problem. "
        f"Are they equivalent in meaning and correctness?\n\n"
        f"Solution A (to evaluate):\n{prediction[:1000]}\n\n"
        f"Solution B (reference):\n{reference[:1000]}\n\n"
        f"Answer with one word: CORRECT, PARTIAL, or INCORRECT."
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, enable_thinking=False,
        tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20,
            do_sample=False,  # greedy for deterministic judging
        )

    response = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

    if not response:
        # Empty judge response — can't classify, return None-like result
        return CompareResult(
            classification="partially_correct",
            tier="tier_3",
            confidence=0.2,
            details={"judge_response": "(empty)"},
        )

    if "incorrect" in response:
        classification = "incorrect"
        confidence = 0.3
    elif "partial" in response:
        classification = "partially_correct"
        confidence = 0.3
    elif "correct" in response:
        classification = "correct"
        confidence = 0.4
    else:
        classification = "incorrect"
        confidence = 0.3

    return CompareResult(
        classification=classification,
        tier="tier_3",
        confidence=confidence,
        details={"judge_response": response[:100]},
    )


# ── Main comparison pipeline ──────────────────────────────────────────────────

def compare_outputs(
    model,
    tokenizer: "PreTrainedTokenizer",
    prediction: str,
    reference: str,
    experiment: dict,
    skip_self_judge: bool = False,
    detector_risk: float = 0.5,
    skip_execution: bool = False,
) -> CompareResult:
    """Compare a prediction against the reference solution.

    Tries tiers in order (strongest first), returns the first available result.
    Tier 1a (execution + detector) is tried first if not skipped.
    Tier 2 (embedding similarity) is always computed as a baseline.

    Args:
        detector_risk: Hallucination risk from calibrate.py (0-1). Used by
                      Tier 1a's composite score. Default 0.5 (neutral).
        skip_execution: If True, skip Tier 1a/1b (no git worktree, no tests).
                       Useful for dream samples where execution is too slow.
    """
    # Guard: empty prediction or reference means we can't compare meaningfully
    if not prediction or not prediction.strip():
        return CompareResult(
            classification="incorrect", tier="tier_2", confidence=0.0,
            details={"error": "empty prediction"})
    if not reference or not reference.strip():
        return CompareResult(
            classification="partially_correct", tier="tier_2", confidence=0.0,
            details={"error": "empty reference"})

    if not skip_execution:
        # Tier 1a: executable verification (worktree + tests + detector)
        t1a = _tier_1a_verify(experiment, prediction, detector_risk)
        if t1a is not None:
            return t1a

        # Tier 1b: syntax check (lightweight, no worktree)
        t1b = _tier_1b_verify(experiment, prediction)
        if t1b is not None:
            return t1b

    # Tier 2: embedding similarity (always available)
    similarity = _embedding_similarity(prediction, reference)

    if similarity >= SIMILARITY_CORRECT_THRESHOLD:
        classification = "correct"
    elif similarity >= SIMILARITY_PARTIAL_THRESHOLD:
        classification = "partially_correct"
    else:
        classification = "incorrect"

    tier2_result = CompareResult(
        classification=classification,
        tier="tier_2",
        confidence=similarity,
        details={"embedding_similarity": similarity},
    )

    # Tier 3: self-judge (optional, expensive — adds a forward pass)
    if not skip_self_judge and model is not None:
        t3 = _tier_3_self_judge(model, tokenizer, prediction, reference)

        # If Tier 2 and Tier 3 agree, boost confidence
        if t3.classification == tier2_result.classification:
            tier2_result.confidence = min(tier2_result.confidence + 0.1, 1.0)
            tier2_result.details["self_judge_agrees"] = True
            tier2_result.details["self_judge_response"] = t3.details.get("judge_response", "")
        else:
            # Disagreement — note it but trust Tier 2 (embedding is more reliable)
            tier2_result.details["self_judge_disagrees"] = True
            tier2_result.details["self_judge_classification"] = t3.classification

    return tier2_result


def compare_dream_cloud(
    model, tokenizer, dream_cloud, reference: str, experiment: dict,
    skip_self_judge: bool = True,
    skip_execution: bool = True,
) -> list[CompareResult]:
    """Compare all samples in a dream cloud against reference.

    Defaults to skip_execution=True because running Tier 1a verification
    on every dream sample (N=4 × worktree + tests) would be too slow.
    Dream comparisons use Tier 2 (embedding) + Tier 3 (self-judge) only.
    """
    return [
        compare_outputs(
            model, tokenizer, sample.text, reference, experiment,
            skip_self_judge=skip_self_judge,
            skip_execution=skip_execution)
        for sample in dream_cloud.samples
    ]

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

import numpy as np

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

def _tier_1a_verify(experiment: dict, prediction: str) -> CompareResult | None:
    """Run tests on the prediction if possible. Returns result or None.

    NOTE: Tier 1a requires actually executing the prediction against tests.
    The original experiment's test_results tell us about the REFERENCE, not
    the model's prediction. Until code execution is implemented, this always
    returns None (falling through to Tier 2).

    TODO: Implement code execution to verify predictions against tests.
    """
    # Cannot verify the prediction without code execution.
    # The experiment's test_results are for the reference solution, not
    # for whatever the model just generated. Returning those would give
    # every experiment with passing tests a free "correct" regardless of
    # what the model actually said.
    return None


# ── Tier 1b: Build/lint artifacts ─────────────────────────────────────────────

def _tier_1b_verify(experiment: dict) -> CompareResult | None:
    """Check build/lint results if available."""
    build = experiment.get("build_results")
    lint = experiment.get("lint_results")

    if build is None and lint is None:
        return None

    build_ok = build.get("passed", True) if build else True
    lint_ok = lint.get("passed", True) if lint else True

    if build_ok and lint_ok:
        return CompareResult(
            classification="partially_correct",  # build passes but doesn't prove correctness
            tier="tier_1b",
            confidence=0.6,
            details={"build_passed": build_ok, "lint_passed": lint_ok},
        )
    else:
        return CompareResult(
            classification="incorrect",
            tier="tier_1b",
            confidence=0.7,
            details={"build_passed": build_ok, "lint_passed": lint_ok},
        )


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

    if "correct" in response and "incorrect" not in response:
        classification = "correct"
        confidence = 0.4
    elif "partial" in response:
        classification = "partially_correct"
        confidence = 0.3
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
) -> CompareResult:
    """Compare a prediction against the reference solution.

    Tries tiers in order (strongest first), returns the first available result.
    Tier 2 (embedding similarity) is always computed as a baseline.
    """
    # Tier 1a: executable verification
    t1a = _tier_1a_verify(experiment, prediction)
    if t1a is not None:
        return t1a

    # Tier 1b: build/lint
    t1b = _tier_1b_verify(experiment)
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
) -> list[CompareResult]:
    """Compare all samples in a dream cloud against reference."""
    return [
        compare_outputs(
            model, tokenizer, sample.text, reference, experiment,
            skip_self_judge=skip_self_judge)
        for sample in dream_cloud.samples
    ]

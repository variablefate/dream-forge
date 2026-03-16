"""C.2 Dream phase — high-temp multi-sampling + structured dream generation.

Generates multiple diverse responses to explore where the model is uncertain.
Supports structured dream variants:
  - replay: re-present known failures
  - false_premise: corrupt the premise to test over-compliance
  - counterfactual: change one constraint, see if model adapts
  - high_temp: standard diverse sampling

Usage:
    from src.engine.dream import dream_sample, structured_dream
    results = dream_sample(model, tokenizer, problem, n=5)
    variants = structured_dream(model, tokenizer, experiment, types=["false_premise", "replay"])
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = "You are a helpful coding assistant. Solve the problem accurately."


@dataclass
class DreamResult:
    text: str
    query: str
    temperature: float
    sample_index: int
    generator: str  # "high_temp" | "false_premise" | "counterfactual" | "replay"


@dataclass
class DreamCloud:
    """Collection of dream samples for a single problem."""
    samples: list[DreamResult] = field(default_factory=list)
    parent_experiment_id: str | None = None
    generator: str = "high_temp"

    @property
    def texts(self) -> list[str]:
        return [s.text for s in self.samples]

    @property
    def n(self) -> int:
        return len(self.samples)


def dream_sample(
    model,
    tokenizer: "PreTrainedTokenizer",
    problem: str,
    error_output: str | None = None,
    context: str | None = None,
    n: int = 5,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
) -> DreamCloud:
    """Generate N high-temp diverse responses to a problem."""
    user_msg = problem
    if error_output:
        user_msg += f"\n\nError:\n{error_output[:800]}"  # match tune.py truncation
    if context:
        user_msg += f"\n\nRelevant code:{context}"  # no extra \n — match tune.py format

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, enable_thinking=False,
        tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    samples = []
    for i in range(n):
        # Clone inputs to avoid any in-place mutation from generate()
        step_inputs = {k: v.clone() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **step_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True,
            )
        text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        if not text:
            continue  # skip empty generations
        samples.append(DreamResult(
            text=text, query=user_msg, temperature=temperature,
            sample_index=i, generator="high_temp",
        ))

    return DreamCloud(samples=samples, generator="high_temp")


def dream_from_experiment(model, tokenizer, experiment: dict, **kwargs) -> DreamCloud:
    """Convenience wrapper that extracts fields from an experiment dict."""
    context_text = ""
    for cf in (experiment.get("pre_solution_context") or [])[:3]:
        context_text += f"\n### {cf.get('path', 'file')}\n{cf.get('content', '')[:2000]}\n"

    cloud = dream_sample(
        model, tokenizer,
        problem=experiment.get("problem", ""),
        error_output=experiment.get("error_output"),
        context=context_text or None,
        **kwargs,
    )
    cloud.parent_experiment_id = str(experiment.get("id", ""))
    return cloud


# ── Structured dream variants ──────────────────────────────────────────────────

def _dream_false_premise(
    model, tokenizer, experiment: dict,
    n: int = 3, temperature: float = 0.8, max_new_tokens: int = 512,
) -> DreamCloud:
    """Corrupt the premise to test over-compliance.

    Takes the original problem and subtly corrupts it — wrong error message,
    outdated API, incorrect assumption. The model should push back, not comply.
    """
    problem = experiment.get("problem", "")

    # Ask the model to generate a corrupted version, then answer it
    corruption_prompt = (
        f"Take this problem and introduce one subtle factual error "
        f"(wrong API name, incorrect error message, or false assumption). "
        f"Then solve the corrupted version as if it were real.\n\n"
        f"Original problem:\n{problem}"
    )

    cloud = dream_sample(
        model, tokenizer, corruption_prompt,
        n=n, temperature=temperature, max_new_tokens=max_new_tokens,
    )
    cloud.generator = "false_premise"
    cloud.parent_experiment_id = str(experiment.get("id", ""))
    for s in cloud.samples:
        s.generator = "false_premise"
    return cloud


def _dream_counterfactual(
    model, tokenizer, experiment: dict,
    n: int = 3, temperature: float = 0.8, max_new_tokens: int = 512,
) -> DreamCloud:
    """Change one constraint, see if model adapts or hallucinates the old answer."""
    problem = experiment.get("problem", "")
    reference = experiment.get("reference_solution", "")

    counterfactual_prompt = (
        f"Here is a problem that was previously solved:\n\n"
        f"Problem: {problem}\n\n"
        f"Now imagine ONE constraint changed (different language, different framework, "
        f"stricter requirement, or different error). "
        f"State the changed constraint clearly, then solve the modified version."
    )

    cloud = dream_sample(
        model, tokenizer, counterfactual_prompt,
        n=n, temperature=temperature, max_new_tokens=max_new_tokens,
    )
    cloud.generator = "counterfactual"
    cloud.parent_experiment_id = str(experiment.get("id", ""))
    for s in cloud.samples:
        s.generator = "counterfactual"
    return cloud


def _dream_replay(
    model, tokenizer, experiment: dict,
    n: int = 3, temperature: float = 0.8, max_new_tokens: int = 512,
) -> DreamCloud:
    """Re-present a known failure — highest-value replay target."""
    cloud = dream_from_experiment(
        model, tokenizer, experiment,
        n=n, temperature=temperature, max_new_tokens=max_new_tokens,
    )
    cloud.generator = "replay"
    for s in cloud.samples:
        s.generator = "replay"
    return cloud


DREAM_GENERATORS = {
    "high_temp": dream_from_experiment,
    "false_premise": _dream_false_premise,
    "counterfactual": _dream_counterfactual,
    "replay": _dream_replay,
}


def structured_dream(
    model, tokenizer, experiment: dict,
    types: list[str] | None = None,
    n_per_type: int = 3,
    **kwargs,
) -> list[DreamCloud]:
    """Generate structured dream variants for an experiment.

    Args:
        types: List of generator types. Default: all types.
        n_per_type: Samples per variant type.

    Returns list of DreamClouds, one per type.

    Only real resolved experiments (synthetic=False) should generate dreams.
    Synthetic depth cap: generation_depth must stay at 1.
    """
    if experiment.get("synthetic", False):
        return []  # synthetic items cannot spawn further synthetic items

    if types is None:
        types = ["high_temp", "replay"]  # conservative default

    clouds = []
    for dream_type in types:
        gen_fn = DREAM_GENERATORS.get(dream_type)
        if gen_fn is None:
            continue
        cloud = gen_fn(model, tokenizer, experiment, n=n_per_type, **kwargs)
        clouds.append(cloud)

    return clouds

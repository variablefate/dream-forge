"""C.1 Wake phase — low-temp deterministic inference.

What the model confidently "knows." Single inference at low temperature,
never sees the reference solution. Used to establish a baseline for
comparison against dream outputs and the teacher reference.

Usage:
    from src.engine.wake import wake_inference
    result = wake_inference(model, tokenizer, experiment)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = "You are a helpful coding assistant. Solve the problem accurately."


@dataclass
class WakeResult:
    text: str
    query: str
    temperature: float
    num_tokens: int


def wake_inference(
    model,
    tokenizer: "PreTrainedTokenizer",
    problem: str,
    error_output: str | None = None,
    context: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> WakeResult:
    """Generate a single greedy response to a problem.

    This is what the model "confidently knows" — deterministic, reproducible.
    The model never sees the reference solution. Uses greedy decoding (temp=0)
    for reproducible baselines.
    """
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

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    return WakeResult(
        text=text if text else "(empty generation)",
        query=user_msg,
        temperature=temperature,
        num_tokens=max(0, out.shape[1] - prompt_len),
    )


def wake_from_experiment(model, tokenizer, experiment: dict, **kwargs) -> WakeResult:
    """Convenience wrapper that extracts fields from an experiment dict."""
    context_text = ""
    for cf in (experiment.get("pre_solution_context") or [])[:3]:
        path = cf.get("path", "file") if isinstance(cf, dict) else "file"
        content = cf.get("content", "") if isinstance(cf, dict) else ""
        context_text += f"\n### {path}\n{content[:2000]}\n"

    return wake_inference(
        model, tokenizer,
        problem=experiment.get("problem", ""),
        error_output=experiment.get("error_output"),
        context=context_text or None,
        **kwargs,
    )

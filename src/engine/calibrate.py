"""C.4 Detector calibration on workflow data.

Applies Lane B's CETT extraction + detector probe to Lane C wake/dream
outputs. Detector-only path: this is an eval/analysis tool, NOT runtime.
It answers "which dream samples are risky?" and "how confident should we
be in the wake answer?"

Feeds into:
  - Active replay priority (high detector risk → replay sooner)
  - Confidence module (detector signal as primary input)
  - Eval analysis (confidence-stratified accuracy)

Usage:
    from src.engine.calibrate import calibrate_experiment, CalibrationResult
    result = calibrate_experiment(model, tokenizer, probe, experiment, wake, dreams)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

console = Console()

DEFAULT_PROBE_PATH = Path("models/detector_probe_pilot.pkl")


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ScoredOutput:
    """A single scored output (wake or dream sample)."""
    text: str
    source: str             # "wake" | "dream_high_temp" | "dream_replay" | etc.
    hallucination_risk: float  # 0-1 from detector probe
    sample_index: int = 0


@dataclass
class CalibrationResult:
    """Detector scores for all outputs of a single experiment."""
    experiment_id: str
    wake_score: ScoredOutput | None = None
    dream_scores: list[ScoredOutput] = field(default_factory=list)
    reference_solution: str = ""

    @property
    def wake_risk(self) -> float:
        """Hallucination risk of the wake output."""
        return self.wake_score.hallucination_risk if self.wake_score else 0.5

    @property
    def min_dream_risk(self) -> float:
        """Lowest risk among dream outputs (best candidate)."""
        if not self.dream_scores:
            return 1.0
        return min(s.hallucination_risk for s in self.dream_scores)

    @property
    def max_dream_risk(self) -> float:
        """Highest risk among dream outputs (worst candidate)."""
        if not self.dream_scores:
            return 1.0
        return max(s.hallucination_risk for s in self.dream_scores)

    @property
    def mean_dream_risk(self) -> float:
        """Mean risk across dream outputs."""
        if not self.dream_scores:
            return 1.0
        return sum(s.hallucination_risk for s in self.dream_scores) / len(self.dream_scores)

    @property
    def all_high_risk(self) -> bool:
        """True if ALL outputs (wake + dreams) score high risk (>0.7)."""
        threshold = 0.7
        scores = [self.wake_risk] + [s.hallucination_risk for s in self.dream_scores]
        return all(s > threshold for s in scores)


# ── Core scoring ──────────────────────────────────────────────────────────────

def score_text(
    model, tokenizer: "PreTrainedTokenizer",
    probe, query: str, text: str,
    layer_names: list[str],
    weight_norms: dict,
    num_neurons: int,
    module_dict: dict | None = None,
) -> float:
    """Score a single text with the detector probe. Returns risk 0-1."""
    from src.runtime.best_of_n import extract_cett_for_text

    import numpy as np

    # Build full text matching probe training format: user-only (no system prompt)
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages, enable_thinking=False,
        tokenize=False, add_generation_prompt=True)
    full_text = prompt + text

    features = extract_cett_for_text(
        model, tokenizer, full_text,
        layer_names, weight_norms, num_neurons,
        module_dict=module_dict)

    proba = probe.predict_proba(features.reshape(1, -1))[0]
    return float(proba[1])  # P(hallucinated)


def _init_detector(model, probe) -> tuple[list[str], dict, int, dict]:
    """Initialize layer names, weight norms, and module dict for CETT.

    Matches the lazy init pattern in BestOfN._init_hooks().
    Cached per model object to avoid recomputing weight norms on every call.
    """
    import torch

    # Cache by model identity — weight norms are expensive to recompute
    cache_key = id(model)
    if hasattr(_init_detector, "_cache") and cache_key in _init_detector._cache:
        return _init_detector._cache[cache_key]

    module_dict = dict(model.named_modules())
    layer_names = sorted([
        name for name in module_dict
        if name.endswith("down_proj") and "mlp" in name
    ])

    weight_norms = {}
    for name in layer_names:
        mod = module_dict[name]
        weight_norms[name] = mod.weight.float().norm(dim=0).cpu()

    if not layer_names:
        raise ValueError("No down_proj MLP layers found in model.")

    num_neurons = weight_norms[layer_names[0]].shape[0]
    result = (layer_names, weight_norms, num_neurons, module_dict)

    if not hasattr(_init_detector, "_cache"):
        _init_detector._cache = {}
    _init_detector._cache[cache_key] = result
    return result


# ── Main calibration ──────────────────────────────────────────────────────────

def calibrate_experiment(
    model,
    tokenizer: "PreTrainedTokenizer",
    probe,
    experiment: dict,
    wake_result=None,
    dream_clouds: list | None = None,
) -> CalibrationResult:
    """Score all wake/dream outputs for a single experiment with the detector.

    Args:
        model: The loaded model (8-bit).
        tokenizer: Matching tokenizer.
        probe: Trained detector probe (from joblib.load).
        experiment: The experiment dict.
        wake_result: WakeResult from wake.py (optional).
        dream_clouds: List of DreamCloud objects from dream.py (optional).

    Returns:
        CalibrationResult with detector scores for all outputs.
    """
    layer_names, weight_norms, num_neurons, module_dict = _init_detector(model, probe)

    # Use full user message from wake result if available (includes error_output
    # + context), matching what the model actually saw during inference.
    # Raw problem-only would produce different activations than inference.
    if wake_result is not None and hasattr(wake_result, "query"):
        query = wake_result.query
    else:
        # Build the full query from experiment fields (same format as wake.py)
        query = experiment.get("problem", "")
        error_output = experiment.get("error_output", "") or ""
        if error_output:
            query += f"\n\nError:\n{error_output[:800]}"
        context_files = experiment.get("pre_solution_context") or []
        context_text = ""
        for cf in context_files[:3]:
            path = cf.get("path", "file") if isinstance(cf, dict) else "file"
            content = cf.get("content", "") if isinstance(cf, dict) else ""
            context_text += f"\n### {path}\n{content[:2000]}\n"
        if context_text:
            query += f"\n\nRelevant code:{context_text}"

    result = CalibrationResult(
        experiment_id=str(experiment.get("id", "")),
        reference_solution=experiment.get("reference_solution", ""),
    )

    # Score wake output
    if wake_result is not None and wake_result.text and wake_result.text != "(empty generation)":
        risk = score_text(
            model, tokenizer, probe, query, wake_result.text,
            layer_names, weight_norms, num_neurons, module_dict)
        result.wake_score = ScoredOutput(
            text=wake_result.text, source="wake",
            hallucination_risk=risk)

    # Score dream samples
    if dream_clouds:
        for cloud in dream_clouds:
            for i, sample in enumerate(cloud.samples):
                if not sample.text:
                    continue
                risk = score_text(
                    model, tokenizer, probe, query, sample.text,
                    layer_names, weight_norms, num_neurons, module_dict)
                result.dream_scores.append(ScoredOutput(
                    text=sample.text,
                    source=f"dream_{cloud.generator}",
                    hallucination_risk=risk,
                    sample_index=i,
                ))

    return result


def load_probe(probe_path: Path = DEFAULT_PROBE_PATH):
    """Load detector probe from disk."""
    import joblib
    return joblib.load(str(probe_path))

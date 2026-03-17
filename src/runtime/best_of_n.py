"""Best-of-N inference with detector-scored selection.

Generates N responses for a query, scores each with the hallucination
detector probe, and returns the best one. Layered selection strategy:
  1. Detector selection (pick lowest hallucination risk)
  2. Consistency voting (break ties by frequency)
  3. Hedge trigger (all high-risk -> hedged response)

This is the core runtime feature for the detector-only path (C.8).

Usage as library:
    from src.runtime.best_of_n import BestOfN
    bon = BestOfN.from_pretrained("models/detector_probe.pkl")
    result = bon.generate(model, tokenizer, "What is Python's GIL?", n=4)
    print(result.text, result.confidence, result.strategy)

Usage as CLI:
    uv run python -m src.runtime.best_of_n --query "What is Python's GIL?"
    uv run python -m src.runtime.best_of_n --query "..." --n 2 --fast
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_N = 4
DEFAULT_HEDGE_THRESHOLD = 0.7  # if all samples score above this, hedge
DEFAULT_TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512
DETECTOR_PROBE_PATH = Path("models/detector_probe.pkl")

# Must match the system prompt used during training (tune.py)
SYSTEM_PROMPT = "You are a helpful coding assistant. Solve the problem accurately."

HEDGE_PREFIX = (
    "I'm not fully confident in this answer. "
    "Here's my best attempt, but please verify: "
)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ScoredResponse:
    text: str
    hallucination_risk: float  # 0-1, from detector probe
    cett_features: np.ndarray | None = None


@dataclass
class BestOfNResult:
    text: str
    confidence: float           # 1 - hallucination_risk of selected
    strategy: str               # "detector" | "consistency" | "hedge" | "pass@1"
    n_generated: int
    all_responses: list[ScoredResponse] = field(default_factory=list)
    hedged: bool = False
    elapsed_seconds: float = 0.0


# ── CETT extraction (lightweight, no caching needed at runtime) ────────────────

def extract_cett_for_text(
    model, tokenizer, text: str,
    layer_names: list[str],
    weight_norms: dict[str, torch.Tensor],
    num_neurons: int,
    module_dict: dict | None = None,
) -> np.ndarray:
    """Extract CETT features for a single response. Returns [32*neurons]."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048,
    ).to(next(model.parameters()).device)

    # Reuse pre-built module dict to avoid rebuilding per call
    if module_dict is None:
        module_dict = dict(model.named_modules())

    layer_io: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    for name in layer_names:
        mod = module_dict[name]
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

    layer_io.clear()  # release GPU tensors promptly
    return np.concatenate(layer_cetts)


# ── Main class ─────────────────────────────────────────────────────────────────

class BestOfN:
    """Best-of-N inference with detector-scored selection."""

    def __init__(
        self,
        detector_probe,
        hedge_threshold: float = DEFAULT_HEDGE_THRESHOLD,
    ):
        self.probe = detector_probe
        self.hedge_threshold = hedge_threshold

        # These get populated on first generate() call (lazy init)
        self._layer_names: list[str] | None = None
        self._weight_norms: dict[str, torch.Tensor] | None = None
        self._num_neurons: int | None = None

    @classmethod
    def from_pretrained(cls, probe_path: str | Path = DETECTOR_PROBE_PATH, **kwargs):
        """Load detector probe from disk."""
        import joblib
        probe = joblib.load(str(probe_path))
        return cls(detector_probe=probe, **kwargs)

    def _init_hooks(self, model):
        """Lazily initialize layer names, weight norms, and module dict from the model."""
        if self._layer_names is not None:
            return

        # Build module dict once — reused for all subsequent CETT extractions
        self._module_dict = dict(model.named_modules())

        self._layer_names = sorted([
            name for name in self._module_dict
            if name.endswith("down_proj") and "mlp" in name
        ])
        self._weight_norms = {}
        for name in self._layer_names:
            mod = self._module_dict[name]
            self._weight_norms[name] = mod.weight.float().norm(dim=0).cpu()

        if not self._layer_names:
            raise ValueError(
                "No down_proj MLP layers found in model. "
                "Check model architecture or PEFT wrapper.")
        self._num_neurons = self._weight_norms[self._layer_names[0]].shape[0]

    def score_response(
        self, model, tokenizer, query: str, response_text: str,
    ) -> ScoredResponse:
        """Score a single response with the detector probe."""
        self._init_hooks(model)

        # Build full text for CETT extraction — user-only messages (no system prompt)
        # to match the probe's training data format (sanity_gate uses user-only)
        messages = [
            {"role": "user", "content": query},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False,
            tokenize=False, add_generation_prompt=True)
        full_text = prompt + response_text

        features = extract_cett_for_text(
            model, tokenizer, full_text,
            self._layer_names, self._weight_norms, self._num_neurons,
            module_dict=self._module_dict)

        # Detector: P(hallucinated) is class 1
        proba = self.probe.predict_proba(features.reshape(1, -1))[0]
        hallucination_risk = float(proba[1])

        return ScoredResponse(
            text=response_text,
            hallucination_risk=hallucination_risk,
            cett_features=features,
        )

    def generate(
        self,
        model, tokenizer,
        query: str,
        n: int = DEFAULT_N,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> BestOfNResult:
        """Generate N responses, score, and select the best one."""
        t0 = time.monotonic()

        # Pass@1 fast path
        if n <= 1:
            text = self._generate_one(model, tokenizer, query, temperature, max_new_tokens)
            if not text.strip():
                scored = ScoredResponse(text="", hallucination_risk=1.0)
            else:
                scored = self.score_response(model, tokenizer, query, text)
            return BestOfNResult(
                text=scored.text,
                confidence=1.0 - scored.hallucination_risk,
                strategy="pass@1",
                n_generated=1,
                all_responses=[scored],
                elapsed_seconds=time.monotonic() - t0,
            )

        # Generate N responses
        responses: list[ScoredResponse] = []
        for i in range(n):
            text = self._generate_one(model, tokenizer, query, temperature, max_new_tokens)
            if not text.strip():
                # Empty generation — assign max risk so it's never selected
                responses.append(ScoredResponse(text="", hallucination_risk=1.0))
                continue
            scored = self.score_response(model, tokenizer, query, text)
            responses.append(scored)

        # Selection strategy
        return self._select(responses, n, time.monotonic() - t0)

    def _generate_one(
        self, model, tokenizer, query: str,
        temperature: float, max_new_tokens: int,
    ) -> str:
        """Generate a single response."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False,
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        prompt_len = inputs.input_ids.shape[1]

        # Guard: do_sample=True requires temperature > 0
        safe_temp = max(temperature, 0.1)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=safe_temp,
                top_p=0.9,
                do_sample=True,
            )

        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    def _select(
        self, responses: list[ScoredResponse], n: int, elapsed: float,
    ) -> BestOfNResult:
        """Apply layered selection strategy."""

        # Check hedge trigger: ALL responses high-risk
        if all(r.hallucination_risk > self.hedge_threshold for r in responses):
            # Pick the least-bad one but prepend hedge
            best = min(responses, key=lambda r: r.hallucination_risk)
            # If best response is empty, hedge prefix alone is meaningless
            hedge_text = (HEDGE_PREFIX + best.text) if best.text.strip() else ""
            return BestOfNResult(
                text=hedge_text,
                confidence=1.0 - best.hallucination_risk,
                strategy="hedge",
                n_generated=n,
                all_responses=responses,
                hedged=True,
                elapsed_seconds=elapsed,
            )

        # Sort by risk (lowest first)
        ranked = sorted(responses, key=lambda r: r.hallucination_risk)

        # Layer 1: Detector selection
        best = ranked[0]
        strategy = "detector"

        # Layer 2: Consistency voting (if top-2 scores are within 5%)
        if len(ranked) >= 2:
            score_gap = ranked[1].hallucination_risk - ranked[0].hallucination_risk
            if score_gap < 0.05:
                # Scores are close — prefer the most common answer (if any agreement exists)
                texts = [r.text.strip().lower()[:100] for r in responses]
                counts = Counter(texts)
                most_common_text, most_common_count = counts.most_common(1)[0]

                if most_common_count >= 2:  # actual consistency required
                    for r in ranked:
                        if r.text.strip().lower()[:100] == most_common_text:
                            best = r
                            strategy = "consistency"
                            break

        return BestOfNResult(
            text=best.text,
            confidence=1.0 - best.hallucination_risk,
            strategy=strategy,
            n_generated=n,
            all_responses=responses,
            elapsed_seconds=elapsed,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Best-of-N inference with detector scoring")
    parser.add_argument("--query", type=str, required=True,
                        help="The query to answer")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of samples (default: {DEFAULT_N})")
    parser.add_argument("--fast", action="store_true",
                        help="Pass@1 only (no best-of-N)")
    parser.add_argument("--probe", type=Path, default=DETECTOR_PROBE_PATH,
                        help="Path to detector probe .pkl")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--hedge-threshold", type=float, default=DEFAULT_HEDGE_THRESHOLD,
                        help=f"Hedge if all responses above this risk (default: {DEFAULT_HEDGE_THRESHOLD})")
    args = parser.parse_args()

    from src.engine.model_loader import load_model

    console.print(f"[bold]Best-of-N Inference[/bold] | n={1 if args.fast else args.n} | "
                   f"temp={args.temperature} | hedge={args.hedge_threshold}")

    # Load model
    console.print("\n  Loading model...", style="dim")
    model, tokenizer = load_model()

    # Load detector
    bon = BestOfN.from_pretrained(args.probe, hedge_threshold=args.hedge_threshold)
    console.print(f"  Detector probe loaded from {args.probe}", style="dim")

    # Generate
    n = 1 if args.fast else args.n
    result = bon.generate(model, tokenizer, args.query, n=n, temperature=args.temperature)

    # Display
    console.print(f"\n[bold]Result[/bold] (strategy: {result.strategy}, "
                   f"confidence: {result.confidence:.1%}, "
                   f"{'HEDGED' if result.hedged else 'direct'}, "
                   f"{result.elapsed_seconds:.1f}s):")
    console.print(f"\n{result.text}")

    if len(result.all_responses) > 1:
        console.print(f"\n[dim]All {result.n_generated} responses:[/dim]")
        for i, r in enumerate(sorted(result.all_responses, key=lambda x: x.hallucination_risk)):
            is_selected = (r.text == result.text) or (result.hedged and r.text in result.text)
            marker = " <-- selected" if is_selected else ""
            console.print(f"  [{i+1}] risk={r.hallucination_risk:.3f} | "
                           f"{r.text[:80]}...{marker}", style="dim")


if __name__ == "__main__":
    main()

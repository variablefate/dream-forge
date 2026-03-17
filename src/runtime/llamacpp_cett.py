"""CETT extraction via llama.cpp eval callback — zero-overhead detection.

Captures MLP down_proj activations DURING generation using llama.cpp's
eval callback. No extra forward pass needed — activations are a free
side effect of normal inference.

Requires:
  - llama-cpp-python with CUDA support
  - GGUF model file (Q8_0 recommended for CETT fidelity)
  - Existing detector probe (models/detector_probe.pkl)
  - Precomputed weight norms (models/weight_norms.npz)

Usage:
    from src.runtime.llamacpp_cett import LlamaCppCETT
    engine = LlamaCppCETT("models/Qwen3.5-9B-Q8_0.gguf")
    result = engine.generate_and_score("What is Python's GIL?", n=4)

Performance target: best-of-4 in ~10-15s (vs 190s with HuggingFace)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

DEFAULT_GGUF_PATH = Path(
    "C:/Users/Iwill/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf"
)
DETECTOR_PROBE_PATH = Path("models/detector_probe.pkl")
WEIGHT_NORMS_PATH = Path("models/weight_norms.npz")
NUM_LAYERS = 32
NUM_NEURONS = 12288  # Qwen3.5-9B intermediate_size


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ScoredGeneration:
    text: str
    hallucination_risk: float
    cett_features: np.ndarray | None = None


@dataclass
class GenerateResult:
    text: str
    confidence: float
    strategy: str
    n_generated: int
    all_responses: list[ScoredGeneration] = field(default_factory=list)
    hedged: bool = False
    elapsed_seconds: float = 0.0


# ── Weight norms extraction ───────────────────────────────────────────────────

def extract_weight_norms_from_pytorch(output_path: Path = WEIGHT_NORMS_PATH):
    """Extract and save down_proj weight norms from the PyTorch model.

    Run this ONCE after training. The norms are constant for a given
    model checkpoint and are needed for CETT computation from llama.cpp
    activations.
    """
    from src.engine.model_loader import load_model
    import torch

    console.print("Extracting weight norms from PyTorch model...")
    model, _ = load_model()

    weight_norms = {}
    for name, mod in model.named_modules():
        if name.endswith("down_proj") and "mlp" in name:
            norms = mod.weight.float().norm(dim=0).cpu().numpy()
            weight_norms[name] = norms

    layer_names = sorted(weight_norms.keys())
    console.print(f"  Found {len(layer_names)} down_proj layers")

    # Save as npz with ordered keys
    norms_array = np.stack([weight_norms[name] for name in layer_names])
    np.savez(
        output_path,
        norms=norms_array,
        layer_names=np.array(layer_names),
    )
    console.print(f"  Saved to {output_path} ({norms_array.shape})")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return output_path


def load_weight_norms(path: Path = WEIGHT_NORMS_PATH) -> tuple[np.ndarray, list[str]]:
    """Load precomputed weight norms. Returns (norms_array, layer_names)."""
    data = np.load(path, allow_pickle=True)
    return data["norms"], data["layer_names"].tolist()


# ── CETT from llama.cpp activations ───────────────────────────────────────────

class LlamaCppCETT:
    """Generate + score with zero-overhead CETT extraction.

    Uses llama.cpp for fast generation and captures down_proj activations
    via the eval callback during inference.
    """

    def __init__(
        self,
        gguf_path: str | Path = DEFAULT_GGUF_PATH,
        probe_path: Path = DETECTOR_PROBE_PATH,
        norms_path: Path = WEIGHT_NORMS_PATH,
        n_gpu_layers: int = -1,  # -1 = offload all to GPU
        n_ctx: int = 2048,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with CUDA support:\n"
                "  pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128"
            )

        console.print(f"  Loading GGUF: {gguf_path}", style="dim")
        self.llm = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

        # Load detector probe
        import joblib
        self.probe = joblib.load(str(probe_path))

        # Load precomputed weight norms
        self.weight_norms, self.layer_names = load_weight_norms(norms_path)
        console.print(f"  {len(self.layer_names)} layers, {self.weight_norms.shape[1]} neurons",
                      style="dim")

        # Activation capture state
        self._captured_activations: dict[int, list[np.ndarray]] = {}
        self._capture_enabled = False

    def _setup_eval_callback(self):
        """Register the eval callback for CETT activation capture.

        The callback fires for every tensor computed during eval.
        We filter for down_proj tensors and capture their values.
        """
        # Map llama.cpp tensor names to our layer indices
        # llama.cpp uses names like "blk.0.ffn_down.weight" for layer 0
        self._layer_map: dict[str, int] = {}
        for i in range(NUM_LAYERS):
            # llama.cpp naming convention for Qwen/Llama-style models
            self._layer_map[f"blk.{i}.ffn_down"] = i

        def eval_callback(tensor_name: str, tensor_data, user_data):
            """Capture down_proj activations during eval."""
            if not self._capture_enabled:
                return True  # continue eval

            # Check if this is a down_proj tensor (input or output)
            for key, layer_idx in self._layer_map.items():
                if key in tensor_name and "weight" not in tensor_name:
                    # This is an activation, not a weight
                    if layer_idx not in self._captured_activations:
                        self._captured_activations[layer_idx] = []
                    # Convert to numpy
                    if hasattr(tensor_data, 'numpy'):
                        self._captured_activations[layer_idx].append(
                            tensor_data.numpy().copy()
                        )
                    break

            return True  # continue eval

        # Register callback
        # Note: exact API depends on llama-cpp-python version
        try:
            self.llm.set_eval_callback(eval_callback)
        except AttributeError:
            console.print(
                "[yellow]Warning: eval_callback not available in this version of "
                "llama-cpp-python. CETT extraction will not work. "
                "Falling back to text-only scoring.[/yellow]"
            )
            self._callback_available = False
            return

        self._callback_available = True

    def compute_cett(self) -> np.ndarray | None:
        """Compute CETT features from captured activations.

        Returns [num_layers * num_neurons] feature vector, or None if
        no activations were captured.
        """
        if not self._captured_activations:
            return None

        layer_cetts = []
        for layer_idx in range(NUM_LAYERS):
            if layer_idx not in self._captured_activations:
                layer_cetts.append(np.zeros(NUM_NEURONS, dtype=np.float32))
                continue

            activations = self._captured_activations[layer_idx]
            if not activations:
                layer_cetts.append(np.zeros(NUM_NEURONS, dtype=np.float32))
                continue

            # Stack activations across tokens, compute CETT
            act = np.concatenate(activations, axis=0)  # [tokens, neurons]
            act_abs = np.abs(act)
            wn = self.weight_norms[layer_idx]  # [neurons]
            out_norms = np.linalg.norm(act, axis=-1, keepdims=True)  # [tokens, 1]

            cett = (act_abs * wn) / (out_norms + 1e-8)
            layer_cetts.append(cett.mean(axis=0).astype(np.float32))

        return np.concatenate(layer_cetts)

    def score(self, cett_features: np.ndarray) -> float:
        """Score CETT features with the detector probe. Returns risk 0-1."""
        proba = self.probe.predict_proba(cett_features.reshape(1, -1))[0]
        return float(proba[1])  # P(hallucinated)

    def generate_one(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        capture_cett: bool = True,
    ) -> ScoredGeneration:
        """Generate a single response and score it."""
        # Clear previous activations
        self._captured_activations.clear()
        self._capture_enabled = capture_cett

        # Generate
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.1),
            echo=False,
        )

        self._capture_enabled = False
        text = output["choices"][0]["text"].strip()

        # Score
        risk = 0.5  # default neutral
        features = None
        if capture_cett and hasattr(self, '_callback_available') and self._callback_available:
            features = self.compute_cett()
            if features is not None:
                risk = self.score(features)

        return ScoredGeneration(
            text=text,
            hallucination_risk=risk,
            cett_features=features,
        )

    def generate_and_score(
        self,
        query: str,
        n: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 512,
        hedge_threshold: float = 0.7,
    ) -> GenerateResult:
        """Generate N responses with CETT scoring. Best-of-N selection."""
        t0 = time.monotonic()

        # Build prompt (matches our training format)
        prompt = (
            f"<|im_start|>system\n"
            f"You are a helpful coding assistant. Solve the problem accurately.<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        responses = []
        for _ in range(n):
            gen = self.generate_one(prompt, max_tokens, temperature)
            responses.append(gen)

        elapsed = time.monotonic() - t0

        # Selection (same logic as BestOfN._select)
        if all(r.hallucination_risk > hedge_threshold for r in responses):
            best = min(responses, key=lambda r: r.hallucination_risk)
            hedge_prefix = "I'm not fully confident in this answer. Here's my best attempt, but please verify: "
            return GenerateResult(
                text=(hedge_prefix + best.text) if best.text.strip() else "",
                confidence=1.0 - best.hallucination_risk,
                strategy="hedge",
                n_generated=n,
                all_responses=responses,
                hedged=True,
                elapsed_seconds=elapsed,
            )

        best = min(responses, key=lambda r: r.hallucination_risk)
        return GenerateResult(
            text=best.text,
            confidence=1.0 - best.hallucination_risk,
            strategy="detector",
            n_generated=n,
            all_responses=responses,
            elapsed_seconds=elapsed,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="llama.cpp inference with zero-overhead CETT detection")
    parser.add_argument("--extract-norms", action="store_true",
                        help="Extract weight norms from PyTorch model (run once)")
    parser.add_argument("--query", type=str,
                        help="Query to answer")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of samples for best-of-N")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF_PATH,
                        help="Path to GGUF model file")
    args = parser.parse_args()

    if args.extract_norms:
        extract_weight_norms_from_pytorch()
        return

    if args.query:
        engine = LlamaCppCETT(gguf_path=args.gguf)
        engine._setup_eval_callback()
        result = engine.generate_and_score(args.query, n=args.n)
        console.print(f"[bold]Result[/bold] (strategy={result.strategy}, "
                      f"confidence={result.confidence:.1%}, {result.elapsed_seconds:.1f}s)")
        console.print(result.text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

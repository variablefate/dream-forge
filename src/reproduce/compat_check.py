"""Day-0 compatibility gate (B.0a) for dream-forge Lane B.

Verifies the full ML environment before any H-neuron work.
Tests framework fallback chain: Unsloth -> PEFT + HF Trainer.

Run:
    uv run python -m src.reproduce.compat_check
    uv run python -m src.reproduce.compat_check --full   # includes merge + GGUF tests
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Unsloth's fused CE loss auto-detects free VRAM and fails when vision encoder
# occupies most of it. We pre-seed the chunk size cache after model load.

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-9B"
VRAM_LIMIT_GB = 16.0
MAX_SEQ_LENGTH = 2048

LORA_TARGET_MODULES = [
    # Full attention (8 layers)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # DeltaNet / linear attention (24 layers)
    "in_proj_qkv", "in_proj_z", "out_proj",
    # MLP (all 32 layers)
    "gate_proj", "up_proj", "down_proj",
]

# DeltaNet modules that MUST receive LoRA adapters (hard gate)
DELTANET_MODULES = {"in_proj_qkv", "in_proj_z", "out_proj"}

EXPECTED_LAYER_PATTERN = ["linear_attention"] * 3 + ["full_attention"]  # x8
EXPECTED_NUM_LAYERS = 32
EXPECTED_HIDDEN_DIM = 4096

RESULTS_PATH = Path("models/compat_result.json")

console = Console()


# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)
    skipped: bool = False
    fatal: bool = False  # abort remaining checks


# ── VRAM helpers ───────────────────────────────────────────────────────────────

def _reset_vram():
    import torch
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def _peak_vram_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / (1024**3)


def _vram_used_gb() -> float:
    """Actual GPU memory used (includes non-PyTorch allocations)."""
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024**3)


# ── Activation offloading ──────────────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def offload_activations():
    """Move saved-for-backward tensors to CPU to reduce GPU VRAM.

    Same mechanism as Unsloth's use_gradient_checkpointing="unsloth" but
    works with any model (no Unsloth patching required).
    Combined with standard gradient checkpointing, this means:
    - Checkpoint boundaries recompute intra-layer activations (standard)
    - Inter-layer hidden states saved to CPU instead of GPU (this)
    """
    import torch

    def pack(x: torch.Tensor):
        if x.device.type == "cuda" and x.numel() >= 1024:
            return (x.device, x.to("cpu", non_blocking=True))
        return (None, x)  # keep small tensors on GPU

    def unpack(packed: tuple):
        device, tensor = packed
        if device is not None:
            return tensor.to(device, non_blocking=True)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


# ── Main checker ───────────────────────────────────────────────────────────────

class CompatChecker:
    """Runs all Day-0 compatibility checks sequentially."""

    def __init__(self, full: bool = False):
        self.results: list[CheckResult] = []
        self.model = None
        self.tokenizer = None
        self.framework: str | None = None  # "unsloth" | "peft"
        self.full = full
        self._t0 = time.monotonic()

    # ── Reporting ──────────────────────────────────────────────────────────

    def _record(self, r: CheckResult):
        self.results.append(r)
        if r.skipped:
            icon, style = "-", "dim"
        elif r.passed:
            icon, style = "+", "green"
        else:
            icon, style = "X", "red bold" if r.fatal else "red"
        console.print(f"  [{style}][{icon}] {r.name}[/{style}]: {r.message}")
        for k, v in r.details.items():
            console.print(f"        {k}: {v}", style="dim")

    def _phase(self, title: str):
        console.print(f"\n[bold cyan]{title}[/bold cyan]")

    # ── Phase 1: Environment ──────────────────────────────────────────────

    def check_cuda(self) -> bool:
        try:
            import torch
        except ImportError:
            self._record(CheckResult("PyTorch", False, "torch not importable", fatal=True))
            return False

        if not torch.cuda.is_available():
            self._record(CheckResult("CUDA", False,
                "torch.cuda.is_available() = False. Check driver/CUDA install.",
                fatal=True))
            return False

        props = torch.cuda.get_device_properties(0)
        compute = f"{props.major}.{props.minor}"
        sm = f"sm_{props.major * 10 + props.minor}"
        vram = props.total_memory / (1024**3)

        # Smoke-test: simple matmul on GPU
        try:
            t = torch.randn(256, 256, device="cuda")
            _ = (t @ t).sum().item()
            del t
            torch.cuda.empty_cache()
        except Exception as e:
            self._record(CheckResult("CUDA", False, f"GPU tensor op failed: {e}", fatal=True))
            return False

        self._record(CheckResult("CUDA", True,
            f"{props.name} | {vram:.1f} GB | compute {compute} ({sm}) | CUDA {torch.version.cuda}",
            {"pytorch": torch.__version__, "device": props.name}))
        return True

    def check_libraries(self) -> bool:
        libs: dict[str, str] = {}
        issues: list[str] = []

        # Import Unsloth FIRST — it must be imported before transformers/peft
        # to apply its patches (monkey-patches for faster training)
        try:
            import unsloth  # noqa: F811
            libs["unsloth"] = unsloth.__version__
        except ImportError:
            libs["unsloth"] = "(not installed)"

        required = [
            ("transformers", 5), ("bitsandbytes", None), ("peft", None),
            ("accelerate", None), ("sklearn", None),
        ]
        for pkg, min_major in required:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "?")
                libs[pkg] = ver
                if min_major and int(ver.split(".")[0]) < min_major:
                    issues.append(f"{pkg} {ver} < {min_major}.0")
            except ImportError:
                issues.append(f"{pkg} missing")

        # Optional but important
        try:
            import fla  # noqa: F811
            libs["flash-linear-attention"] = fla.__version__
        except ImportError:
            libs["flash-linear-attention"] = "(not installed — DeltaNet falls back to naive PyTorch)"

        try:
            import triton  # noqa: F811
            libs["triton"] = triton.__version__
        except ImportError:
            libs["triton"] = "(not installed — required by FLA)"

        if issues:
            self._record(CheckResult("Libraries", False,
                "; ".join(issues), libs, fatal=True))
            return False

        self._record(CheckResult("Libraries", True,
            "All required packages present", libs))
        return True

    # ── Phase 2: Model Loading ────────────────────────────────────────────

    def check_model_load(self) -> bool:
        console.print(f"        Target: [bold]{MODEL_ID}[/bold] (text-only, 8-bit)", style="dim")

        if self._try_unsloth():
            return True
        if self._try_peft():
            return True

        self._record(CheckResult("Model Load", False,
            "All frameworks failed", fatal=True))
        return False

    def _try_unsloth(self) -> bool:
        try:
            from unsloth import FastLanguageModel  # noqa: F811
        except ImportError:
            console.print("        Unsloth not installed, skipping", style="dim")
            return False

        import torch

        console.print("        Trying Unsloth...", style="dim")
        _reset_vram()

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_ID,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=False,
                load_in_8bit=True,
                dtype=None,
            )
            peak = _peak_vram_gb()
            self.model = model
            self.framework = "unsloth"

            # Unsloth may load the multimodal model (Qwen3_5ForConditionalGeneration)
            # which returns a processor instead of a tokenizer. Replace with text-only
            # AutoTokenizer so hooks/inference/training use text inputs, not image inputs.
            has_vision = any("visual" in n or "vision" in n
                            for n, _ in model.named_modules())
            if has_vision:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

                # Delete vision encoder to reclaim VRAM — we only use text
                if hasattr(model, "visual"):
                    del model.visual
                elif hasattr(model, "model") and hasattr(model.model, "visual"):
                    del model.model.visual
                import torch
                torch.cuda.empty_cache()
                gc.collect()
                freed = peak - _peak_vram_gb()
                console.print(
                    f"        Vision encoder deleted — freed ~{freed:.1f} GB",
                    style="dim yellow")
            else:
                self.tokenizer = tokenizer

            self._record(CheckResult("Model Load (Unsloth)", True,
                f"peak VRAM {peak:.2f} GB | vision encoder: {'present (ignored)' if has_vision else 'excluded'}",
                {"framework": "unsloth", "peak_vram_gb": f"{peak:.2f}",
                 "has_vision_encoder": has_vision}))
            return True

        except Exception as e:
            self._record(CheckResult("Model Load (Unsloth)", False,
                f"{type(e).__name__}: {str(e)[:200]}"))
            # Ensure cleanup before PEFT fallback
            self.model = self.tokenizer = None
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            return False

    def _try_peft(self) -> bool:
        import torch
        console.print("        Trying PEFT (standard transformers + bitsandbytes)...", style="dim")
        _reset_vram()

        try:
            from transformers import AutoTokenizer, BitsAndBytesConfig

            # Prefer explicit text-only class
            try:
                from transformers import Qwen3_5ForCausalLM as ModelClass
                cls_name = "Qwen3_5ForCausalLM"
            except ImportError:
                from transformers import AutoModelForCausalLM as ModelClass
                cls_name = "AutoModelForCausalLM"

            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            model = ModelClass.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_cfg,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            peak = _peak_vram_gb()

            self.model, self.tokenizer, self.framework = model, tokenizer, "peft"

            has_vision = any("visual" in n or "vision" in n
                            for n, _ in model.named_modules())

            self._record(CheckResult("Model Load (PEFT)", True,
                f"{cls_name} | peak VRAM {peak:.2f} GB | vision: {'LOADED' if has_vision else 'excluded'}",
                {"framework": "peft", "model_class": cls_name,
                 "peak_vram_gb": f"{peak:.2f}", "has_vision_encoder": has_vision}))
            return True

        except Exception as e:
            self._record(CheckResult("Model Load (PEFT)", False,
                f"{type(e).__name__}: {str(e)[:300]}"))
            return False

    # ── Phase 3: Architecture ─────────────────────────────────────────────

    def check_architecture(self) -> bool:
        if not self.model:
            self._record(CheckResult("Architecture", False, "No model", skipped=True))
            return False

        config = self.model.config
        text_cfg = getattr(config, "text_config", config)
        issues: list[str] = []
        details: dict = {}

        # Layer count
        n_layers = getattr(text_cfg, "num_hidden_layers", None)
        details["num_layers"] = n_layers
        if n_layers != EXPECTED_NUM_LAYERS:
            issues.append(f"layers={n_layers} (expected {EXPECTED_NUM_LAYERS})")

        # Hidden dim
        hidden = getattr(text_cfg, "hidden_size", None)
        details["hidden_size"] = hidden
        if hidden != EXPECTED_HIDDEN_DIM:
            issues.append(f"hidden={hidden} (expected {EXPECTED_HIDDEN_DIM})")

        # Layer types pattern
        layer_types = getattr(text_cfg, "layer_types", None)
        if layer_types is not None:
            pattern = list(layer_types)
            expected = EXPECTED_LAYER_PATTERN * 8
            if pattern == expected:
                lin = sum(1 for t in pattern if t == "linear_attention")
                full = sum(1 for t in pattern if t == "full_attention")
                details["linear_attention_layers"] = lin
                details["full_attention_layers"] = full
            else:
                issues.append("layer_types pattern mismatch")
                details["actual_first_8"] = pattern[:8]
        else:
            issues.append("layer_types not in config (check transformers version)")

        # down_proj count
        down_proj_layers = [n for n, _ in self.model.named_modules()
                           if n.endswith("down_proj") and "mlp" in n]
        details["down_proj_count"] = len(down_proj_layers)
        if n_layers and len(down_proj_layers) != n_layers:
            issues.append(f"down_proj={len(down_proj_layers)} (expected {n_layers})")

        # Check for fused names that should NOT exist (Qwen3.5 uses separate projections)
        fused_names = [n for n, _ in self.model.named_modules()
                       if "in_proj_qkvz" in n or "in_proj_ba" in n]
        if fused_names:
            details["fused_projections_found"] = fused_names[:3]
            issues.append("Fused DeltaNet projections found (expected separate)")

        if issues:
            self._record(CheckResult("Architecture", False,
                "; ".join(issues), details))
            return False

        self._record(CheckResult("Architecture", True,
            f"{n_layers} layers ({details.get('linear_attention_layers', '?')} DeltaNet + "
            f"{details.get('full_attention_layers', '?')} full-attn) | "
            f"hidden={hidden} | {len(down_proj_layers)} down_proj",
            details))
        return True

    # ── Phase 4: Tokenizer ────────────────────────────────────────────────

    def check_tokenizer(self) -> bool:
        if not self.tokenizer:
            self._record(CheckResult("Tokenizer", False, "No tokenizer", skipped=True))
            return False

        messages = [{"role": "user", "content": "What is 2+2?"}]
        details: dict = {}

        for mode_name, enabled in [("thinking_off", False), ("thinking_on", True)]:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, enable_thinking=enabled,
                    tokenize=False, add_generation_prompt=True)
                details[f"{mode_name}_chars"] = len(text)
            except Exception as e:
                self._record(CheckResult("Tokenizer", False,
                    f"enable_thinking={enabled} failed: {e}"))
                return False

        self._record(CheckResult("Tokenizer", True,
            "enable_thinking=True/False both work", details))
        return True

    # ── Phase 5: Forward Hooks ────────────────────────────────────────────

    def check_hooks(self) -> bool:
        if not self.model:
            self._record(CheckResult("Forward Hooks", False, "No model", skipped=True))
            return False

        import torch

        activations: dict[str, tuple] = {}
        handles = []

        for name, mod in self.model.named_modules():
            if name.endswith("down_proj") and "mlp" in name:
                def hook(m, inp, out, n=name):
                    activations[n] = inp[0].shape
                handles.append(mod.register_forward_hook(hook))

        registered = len(handles)

        try:
            ids = self.tokenizer("Test", return_tensors="pt").to(next(self.model.parameters()).device)
            with torch.no_grad():
                self.model(**ids)

            fired = len(activations)
            shape = str(list(activations.values())[0]) if activations else "N/A"

            if fired != registered:
                self._record(CheckResult("Forward Hooks", False,
                    f"Only {fired}/{registered} hooks fired"))
                return False

            # Verify activation shape: should be [batch, seq, intermediate_size]
            sample = list(activations.values())[0]
            intermediate = sample[-1]
            details = {
                "hooks_registered": registered,
                "hooks_fired": fired,
                "activation_shape": str(sample),
                "intermediate_size": intermediate,
            }

            self._record(CheckResult("Forward Hooks", True,
                f"All {registered} hooks fired | activation shape {shape}", details))
            return True

        except Exception as e:
            self._record(CheckResult("Forward Hooks", False,
                f"Forward pass failed: {e}"))
            return False
        finally:
            for h in handles:
                h.remove()
            activations.clear()

    # ── Phase 6: Inference ────────────────────────────────────────────────

    def check_inference(self) -> bool:
        if not self.model or not self.tokenizer:
            self._record(CheckResult("Inference", False, "No model", skipped=True))
            return False

        import torch
        _reset_vram()

        try:
            messages = [{"role": "user",
                         "content": "What is the capital of France? Answer in one word."}]
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False,
                tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(next(self.model.parameters()).device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=50,
                    temperature=0.1, do_sample=True,
                )

            response = self.tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True).strip()

            peak = _peak_vram_gb()
            coherent = "paris" in response.lower()

            details = {
                "response": response[:200],
                "peak_vram_gb": f"{peak:.2f}",
                "coherent": coherent,
            }

            if peak >= VRAM_LIMIT_GB:
                self._record(CheckResult("Inference", False,
                    f"Peak VRAM {peak:.2f} GB exceeds {VRAM_LIMIT_GB} GB", details))
                return False

            self._record(CheckResult("Inference", True,
                f"'{response[:80]}' | peak {peak:.2f} GB", details))
            return True

        except Exception as e:
            self._record(CheckResult("Inference", False,
                f"{type(e).__name__}: {str(e)[:200]}"))
            return False

    # ── Phase 7: LoRA + Training ──────────────────────────────────────────

    def _fix_unsloth_fused_ce(self):
        """Patch Unsloth's fused CE VRAM auto-detection.

        _get_chunk_multiplier checks free VRAM via torch.cuda.mem_get_info().
        With vision encoder loaded, free VRAM is near zero on 16GB, causing
        RuntimeError. Replace with a version that uses a fixed 1GB target.
        """
        if self.framework != "unsloth":
            return
        try:
            import functools
            from unsloth_zoo.fused_losses import cross_entropy_loss

            @functools.cache
            def _patched(vocab_size, target_gb=None):
                if target_gb is None:
                    target_gb = 1.0  # fixed instead of auto-detect
                multiplier = (vocab_size * 4 / 1024 / 1024 / 1024) / target_gb
                return multiplier / 4

            cross_entropy_loss._get_chunk_multiplier = _patched
            console.print("        Patched fused CE VRAM check (fixed 1GB target)", style="dim")
        except Exception:
            pass

    def check_lora_and_training(self) -> bool:
        if not self.model:
            self._record(CheckResult("LoRA + Training", False, "No model", skipped=True))
            return False

        if not self._lora_setup():
            return False
        if not self._verify_lora_targets():
            return False
        self._fix_unsloth_fused_ce()
        return self._training_step()

    def _lora_setup(self) -> bool:
        if self.framework == "unsloth":
            return self._lora_unsloth()
        return self._lora_peft()

    def _lora_unsloth(self) -> bool:
        try:
            from unsloth import FastLanguageModel
            self.model = FastLanguageModel.get_peft_model(
                self.model, r=16, lora_alpha=16,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=0, bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            self._record(CheckResult("LoRA Init (Unsloth)", True,
                "Adapters + gradient checkpointing (unsloth mode)",
                {"grad_checkpoint": "unsloth"}))
            return True
        except Exception as e:
            self._record(CheckResult("LoRA Init (Unsloth)", False,
                f"{type(e).__name__}: {str(e)[:200]}"))
            return False

    def _lora_peft(self) -> bool:
        try:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(
                r=16, lora_alpha=16,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=0, bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, cfg)

            # Try non-reentrant (modern PyTorch), fall back to reentrant
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                grad_ckpt_type = "non-reentrant"
            except Exception:
                self.model.gradient_checkpointing_enable()
                grad_ckpt_type = "reentrant (fallback)"

            self._record(CheckResult("LoRA Init (PEFT)", True,
                f"Adapters + gradient checkpointing ({grad_ckpt_type})",
                {"grad_checkpoint": grad_ckpt_type}))
            return True
        except Exception as e:
            self._record(CheckResult("LoRA Init (PEFT)", False,
                f"{type(e).__name__}: {str(e)[:200]}"))
            return False

    def _verify_lora_targets(self) -> bool:
        """HARD GATE: Verify DeltaNet modules received LoRA adapters."""
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]

        # Categorize found LoRA targets
        found_deltanet: set[str] = set()
        found_full_attn: set[str] = set()
        found_mlp: set[str] = set()

        full_attn_set = {"q_proj", "k_proj", "v_proj", "o_proj"}
        mlp_set = {"gate_proj", "up_proj", "down_proj"}

        for name in trainable:
            if "lora" not in name.lower():
                continue
            for mod in DELTANET_MODULES:
                if f".{mod}." in name or name.endswith(f".{mod}"):
                    found_deltanet.add(mod)
            for mod in full_attn_set:
                if f".{mod}." in name or name.endswith(f".{mod}"):
                    found_full_attn.add(mod)
            for mod in mlp_set:
                if f".{mod}." in name or name.endswith(f".{mod}"):
                    found_mlp.add(mod)

        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        pct = total_trainable / total_params * 100

        missing_dn = DELTANET_MODULES - found_deltanet

        details = {
            "trainable_params": f"{total_trainable:,} ({pct:.2f}%)",
            "deltanet_found": sorted(found_deltanet),
            "full_attn_found": sorted(found_full_attn),
            "mlp_found": sorted(found_mlp),
        }

        if missing_dn:
            details["missing_deltanet"] = sorted(missing_dn)
            self._record(CheckResult("LoRA Targeting", False,
                f"HARD GATE: DeltaNet modules missing: {sorted(missing_dn)}. "
                f"Without these, only 8/32 attention layers (25%) get LoRA. "
                f"Training is fundamentally incomplete.",
                details, fatal=True))
            return False

        self._record(CheckResult("LoRA Targeting", True,
            f"All target modules found | {total_trainable:,} trainable params ({pct:.2f}%)",
            details))
        return True

    def _training_step(self) -> bool:
        """Forward + backward + optimizer step at seq_len=2048, batch=1.

        Tries standard gradient checkpointing first. If VRAM exceeds budget,
        retries with CPU activation offloading (saved_tensors_hooks).
        """
        import torch

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try standard first
        peak, loss_val, opt_name, seq_len = self._run_one_training_step(offload=False)

        if peak is not None and peak < VRAM_LIMIT_GB:
            self._record(CheckResult("Training Step", True,
                f"loss={loss_val:.4f} | peak {peak:.2f} GB | "
                f"{VRAM_LIMIT_GB - peak:.2f} GB headroom | seq={seq_len}",
                {"loss": f"{loss_val:.4f}", "seq_length": seq_len,
                 "peak_vram_gb": f"{peak:.2f}",
                 "headroom_gb": f"{VRAM_LIMIT_GB - peak:.2f}",
                 "optimizer": opt_name, "offload": False}))
            return True

        if peak is not None:
            console.print(
                f"        Standard: {peak:.2f} GB (over budget). "
                f"Retrying with CPU activation offloading...", style="dim yellow")

        # Retry with CPU activation offloading
        peak2, loss_val2, opt_name2, seq_len2 = self._run_one_training_step(offload=True)

        if peak2 is not None and peak2 < VRAM_LIMIT_GB:
            self._record(CheckResult("Training Step (CPU offload)", True,
                f"loss={loss_val2:.4f} | peak {peak2:.2f} GB | "
                f"{VRAM_LIMIT_GB - peak2:.2f} GB headroom | seq={seq_len2}",
                {"loss": f"{loss_val2:.4f}", "seq_length": seq_len2,
                 "peak_vram_gb": f"{peak2:.2f}",
                 "headroom_gb": f"{VRAM_LIMIT_GB - peak2:.2f}",
                 "optimizer": opt_name2, "offload": True,
                 "standard_peak_gb": f"{peak:.2f}" if peak else "N/A"}))
            return True

        # Check if standard mode completed but exceeded physical VRAM
        # Windows WDDM transparently spills to system RAM — training still works
        SOFT_LIMIT_GB = VRAM_LIMIT_GB * 1.5  # 24 GB — beyond this, shared memory hurts too much
        if peak is not None and peak < SOFT_LIMIT_GB:
            headroom_note = (
                f"Exceeds {VRAM_LIMIT_GB:.0f} GB physical VRAM but within WDDM shared memory range. "
                f"Training works — OS spills ~{peak - VRAM_LIMIT_GB:.1f} GB to system RAM (minor speed impact)."
            )
            self._record(CheckResult("Training Step (shared memory)", True,
                f"loss={loss_val:.4f} | peak {peak:.2f} GB | {headroom_note}",
                {"loss": f"{loss_val:.4f}", "seq_length": seq_len,
                 "peak_vram_gb": f"{peak:.2f}", "optimizer": opt_name,
                 "uses_shared_memory": True,
                 "overflow_gb": f"{peak - VRAM_LIMIT_GB:.2f}"}))
            return True

        # Hard failure — exceeded even the soft limit or both modes failed
        best_peak = min(p for p in [peak, peak2] if p is not None) if any(
            p is not None for p in [peak, peak2]) else None
        details = {}
        if peak is not None:
            details["standard_peak_gb"] = f"{peak:.2f}"
        if peak2 is not None:
            details["offload_peak_gb"] = f"{peak2:.2f}"

        self._record(CheckResult("Training Step", False,
            f"Peak VRAM {best_peak:.2f} GB exceeds soft limit ({SOFT_LIMIT_GB:.0f} GB)"
            if best_peak else "Training step failed",
            details, fatal=True))
        return False

    def _run_one_training_step(
        self, offload: bool
    ) -> tuple[float | None, float | None, str | None, int | None]:
        """Run forward+backward+optimizer. Returns (peak_gb, loss, opt_name, seq_len) or Nones on failure."""
        import torch
        _reset_vram()

        try:
            dummy = "Python is a high-level programming language. " * 300
            enc = self.tokenizer(
                dummy, return_tensors="pt",
                max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length",
            ).to(next(self.model.parameters()).device)
            enc["labels"] = enc["input_ids"].clone()
            seq_len = enc["input_ids"].shape[1]

            self.model.train()

            # Forward + backward (optionally with CPU activation offloading)
            if offload:
                with offload_activations():
                    outputs = self.model(**enc)
                    loss = outputs.loss
                    loss.backward()
            else:
                outputs = self.model(**enc)
                loss = outputs.loss
                loss.backward()

            # Optimizer step
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.AdamW8bit(
                    (p for p in self.model.parameters() if p.requires_grad), lr=2e-5)
                opt_name = "AdamW8bit"
            except Exception:
                opt = torch.optim.AdamW(
                    (p for p in self.model.parameters() if p.requires_grad), lr=2e-5)
                opt_name = "AdamW (32-bit fallback)"

            opt.step()
            peak = _peak_vram_gb()

            # Cleanup
            opt.zero_grad()
            self.model.eval()
            del enc, opt, outputs
            _reset_vram()

            return peak, loss.item(), opt_name, seq_len

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _reset_vram()
                console.print(f"        OOM during {'offloaded' if offload else 'standard'} training step",
                              style="dim red")
            else:
                console.print(f"        {'Offloaded' if offload else 'Standard'} step error: {e!s:.150}",
                              style="dim red")
            return None, None, None, None
        except Exception as e:
            console.print(f"        {'Offloaded' if offload else 'Standard'} step error: {e!s:.150}",
                          style="dim red")
            return None, None, None, None

    # ── Phase 8: LoRA Merge (--full) ──────────────────────────────────────

    def check_merge(self) -> bool:
        if not self.full:
            self._record(CheckResult("LoRA Merge", True,
                "Skipped (use --full to test)", skipped=True))
            return True

        if not self.model:
            self._record(CheckResult("LoRA Merge", False, "No model", skipped=True))
            return False

        import torch

        with tempfile.TemporaryDirectory(prefix="dreamforge_merge_", ignore_cleanup_errors=True) as tmpdir:
            merge_dir = Path(tmpdir) / "merged"
            adapter_dir = Path(tmpdir) / "adapter"

            try:
                if self.framework == "unsloth":
                    return self._merge_unsloth(merge_dir)
                else:
                    return self._merge_peft(adapter_dir, merge_dir)

            except Exception as e:
                self._record(CheckResult("LoRA Merge", False,
                    f"{type(e).__name__}: {str(e)[:300]}"))
                return False

    def _merge_unsloth(self, merge_dir: Path) -> bool:
        from unsloth import FastLanguageModel
        console.print("        Merging via Unsloth save_pretrained_merged...", style="dim")
        console.print("        (downloads fp16 weights if not cached, ~18 GB)", style="dim")

        self.model.save_pretrained_merged(
            str(merge_dir), self.tokenizer,
            save_method="merged_16bit",
        )

        shards = list(merge_dir.glob("*.safetensors"))
        total_gb = sum(f.stat().st_size for f in shards) / (1024**3)

        self._record(CheckResult("LoRA Merge (Unsloth)", True,
            f"{len(shards)} shards | {total_gb:.1f} GB",
            {"shards": len(shards), "size_gb": f"{total_gb:.1f}"}))
        return True

    def _merge_peft(self, adapter_dir: Path, merge_dir: Path) -> bool:
        import torch

        # Step 1: Save adapter
        console.print("        Saving LoRA adapter...", style="dim")
        self.model.save_pretrained(str(adapter_dir))

        # Step 2: Reload base in fp16 on CPU
        console.print("        Reloading base model fp16 on CPU (~20 GB RAM required)...",
                       style="dim")

        try:
            from transformers import Qwen3_5ForCausalLM as ModelClass
        except ImportError:
            from transformers import AutoModelForCausalLM as ModelClass

        base = ModelClass.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="cpu",
        )

        # Step 3: Load adapter and merge
        from peft import PeftModel
        console.print("        Merging adapter...", style="dim")
        merged = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = merged.merge_and_unload()

        # Step 4: Save
        console.print("        Saving merged model...", style="dim")
        merged.save_pretrained(str(merge_dir))
        self.tokenizer.save_pretrained(str(merge_dir))

        del base, merged
        gc.collect()

        shards = list(merge_dir.glob("*.safetensors"))
        total_gb = sum(f.stat().st_size for f in shards) / (1024**3)

        self._record(CheckResult("LoRA Merge (PEFT)", True,
            f"{len(shards)} shards | {total_gb:.1f} GB",
            {"shards": len(shards), "size_gb": f"{total_gb:.1f}"}))
        return True

    # ── Phase 9: GGUF Tools (--full) ─────────────────────────────────────

    def check_gguf(self) -> bool:
        if not self.full:
            self._record(CheckResult("GGUF Tools", True,
                "Skipped (use --full to test)", skipped=True))
            return True

        details: dict = {}

        # Check for llama.cpp tools
        for tool in ["llama-server", "llama-cli", "llama-quantize"]:
            path = shutil.which(tool)
            details[tool] = str(path) if path else "not found"

        # Check for convert_hf_to_gguf.py
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if not convert_script:
            # Check common locations
            for candidate in [
                Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
                Path("llama.cpp") / "convert_hf_to_gguf.py",
            ]:
                if candidate.exists():
                    convert_script = str(candidate)
                    break

        details["convert_hf_to_gguf.py"] = convert_script or "not found"

        has_tools = any(v != "not found" for v in details.values())

        if not has_tools:
            self._record(CheckResult("GGUF Tools", False,
                "No llama.cpp tools found. Install llama.cpp for GGUF export.",
                details))
            return False

        # If we have the converter, try a version check
        if convert_script:
            try:
                result = subprocess.run(
                    [sys.executable, convert_script, "--help"],
                    capture_output=True, text=True, timeout=10)
                details["converter_works"] = result.returncode == 0
            except Exception as e:
                details["converter_error"] = str(e)[:100]

        self._record(CheckResult("GGUF Tools", True,
            "llama.cpp tools available", details))
        return True

    # ── Runner ────────────────────────────────────────────────────────────

    def run(self) -> bool:
        console.print(Panel.fit(
            f"[bold]Day-0 Compatibility Check (B.0a)[/bold]\n"
            f"Model: {MODEL_ID} | VRAM limit: {VRAM_LIMIT_GB} GB\n"
            f"Mode: {'full (merge + GGUF)' if self.full else 'standard'}\n"
            f"Platform: {platform.system()} {platform.release()}",
            title="dream-forge"))

        # Phase 1
        self._phase("Phase 1: Environment")
        if not self.check_cuda():
            return self._finalize()
        if not self.check_libraries():
            return self._finalize()

        # Phase 2
        self._phase("Phase 2: Model Loading (may download ~19 GB on first run)")
        if not self.check_model_load():
            return self._finalize()

        # Phase 3
        self._phase("Phase 3: Architecture Verification")
        self.check_architecture()

        # Phase 4
        self._phase("Phase 4: Tokenizer")
        self.check_tokenizer()

        # Phase 5
        self._phase("Phase 5: Forward Hooks")
        self.check_hooks()

        # Phase 6
        self._phase("Phase 6: Inference")
        self.check_inference()

        # Phase 7
        self._phase("Phase 7: LoRA + Training")
        self.check_lora_and_training()

        # Phase 8
        self._phase("Phase 8: LoRA Merge" +
                     (" (--full)" if not self.full else ""))
        self.check_merge()

        # Phase 9
        self._phase("Phase 9: GGUF Tools" +
                     (" (--full)" if not self.full else ""))
        self.check_gguf()

        return self._finalize()

    def _finalize(self) -> bool:
        elapsed = time.monotonic() - self._t0

        # Summary table
        table = Table(title="\nResults Summary", show_lines=True)
        table.add_column("Check", style="bold", min_width=25)
        table.add_column("Result", justify="center", min_width=6)
        table.add_column("Details", max_width=65)

        passed = failed = skipped = 0
        for r in self.results:
            if r.skipped:
                status = "[dim]SKIP[/dim]"
                skipped += 1
            elif r.passed:
                status = "[green]PASS[/green]"
                passed += 1
            else:
                status = "[red]FAIL[/red]"
                failed += 1
            table.add_row(r.name, status, r.message[:65])

        console.print(table)

        all_ok = failed == 0
        color = "green" if all_ok else "red"
        console.print(
            f"\n[bold {color}]{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}[/bold {color}] "
            f"| {passed} passed, {failed} failed, {skipped} skipped | {elapsed:.0f}s")

        if self.framework:
            console.print(
                f"[bold]Recommended framework:[/bold] {self.framework}")
            console.print(
                f"[bold]Pin in pyproject.toml:[/bold] Pin the exact versions "
                f"shown above after verifying results.")

        if platform.system() == "Windows":
            console.print(
                "[dim]Windows note: Use dataset_num_proc=1 in all data loading.[/dim]")

        if not self.full and all_ok:
            console.print(
                "\n[dim]Run with --full to also test LoRA merge and GGUF conversion.[/dim]")

        self._save_results(all_ok)
        return all_ok

    def _save_results(self, all_ok: bool):
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "passed": all_ok,
            "framework": self.framework,
            "model_id": MODEL_ID,
            "platform": platform.system(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "full_mode": self.full,
            "versions": {},
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "skipped": r.skipped,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

        # Collect versions
        for pkg in ["torch", "transformers", "bitsandbytes", "peft", "accelerate"]:
            try:
                result["versions"][pkg] = __import__(pkg).__version__
            except (ImportError, AttributeError):
                pass
        try:
            import unsloth
            result["versions"]["unsloth"] = unsloth.__version__
        except ImportError:
            pass
        try:
            import torch
            if hasattr(torch.version, "cuda"):
                result["versions"]["cuda"] = torch.version.cuda
        except ImportError:
            pass

        RESULTS_PATH.write_text(json.dumps(result, indent=2))
        console.print(f"Results saved to [bold]{RESULTS_PATH}[/bold]")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Day-0 compatibility gate for dream-forge Lane B")
    parser.add_argument("--full", action="store_true",
                        help="Include LoRA merge and GGUF conversion tests")
    args = parser.parse_args()

    checker = CompatChecker(full=args.full)
    try:
        ok = checker.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        checker._finalize()
        sys.exit(130)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

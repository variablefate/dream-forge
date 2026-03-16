"""Shared model loading for dream-forge.

Encapsulates the Unsloth + AutoTokenizer swap + fused CE monkeypatch pattern
discovered during Day-0 compat check. Used by:
  - src/reproduce/sanity_gate.py
  - src/reproduce/dataset.py
  - src/engine/tune.py
  - src/runtime/best_of_n.py

Usage:
    from src.engine.model_loader import load_model
    model, tokenizer = load_model()           # inference
    model, tokenizer = load_model(lora=True)  # training (adds LoRA + grad ckpt)
"""

from __future__ import annotations

import functools
import gc
from typing import TYPE_CHECKING

import torch
from rich.console import Console

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from torch.nn import Module

MODEL_ID = "Qwen/Qwen3.5-9B"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",        # full attention (8 layers)
    "in_proj_qkv", "in_proj_z", "out_proj",         # DeltaNet (24 layers)
    "gate_proj", "up_proj", "down_proj",             # MLP (all 32 layers)
]

console = Console()


def _patch_fused_ce():
    """Patch Unsloth's fused CE VRAM auto-detection.

    _get_chunk_multiplier checks free VRAM via torch.cuda.mem_get_info().
    With vision encoder loaded, free VRAM is near zero, causing RuntimeError.
    Replace with a fixed 1 GB target.
    """
    try:
        from unsloth_zoo.fused_losses import cross_entropy_loss

        @functools.cache
        def _patched(vocab_size, target_gb=None):
            if target_gb is None:
                target_gb = 1.0
            return (vocab_size * 4 / 1024 / 1024 / 1024) / target_gb / 4

        cross_entropy_loss._get_chunk_multiplier = _patched
    except Exception:
        pass


def _swap_tokenizer_and_cleanup(model) -> tuple[bool, "PreTrainedTokenizer"]:
    """If Unsloth loaded multimodal model, swap processor for text-only tokenizer."""
    from transformers import AutoTokenizer

    has_vision = any("visual" in n or "vision" in n
                     for n, _ in model.named_modules())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if has_vision:
        # Delete vision encoder to reclaim VRAM
        if hasattr(model, "visual"):
            del model.visual
        elif hasattr(model, "model") and hasattr(model.model, "visual"):
            del model.model.visual
        torch.cuda.empty_cache()
        gc.collect()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return has_vision, tokenizer


def load_model(
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    model_id: str = MODEL_ID,
) -> tuple["Module", "PreTrainedTokenizer"]:
    """Load Qwen3.5-9B 8-bit with Unsloth (preferred) or PEFT fallback.

    Args:
        lora: If True, add LoRA adapters + gradient checkpointing (for training).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        model_id: HuggingFace model ID.

    Returns:
        (model, tokenizer) tuple ready for inference or training.
    """
    model = None
    tokenizer = None
    framework = None

    # Try Unsloth first
    try:
        from unsloth import FastLanguageModel

        console.print(f"  Loading {model_id} via Unsloth (8-bit)...", style="dim")
        model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_8bit=True,
            dtype=None,
        )

        has_vision, tokenizer = _swap_tokenizer_and_cleanup(model)
        _patch_fused_ce()
        framework = "unsloth"

        vram = torch.cuda.memory_allocated() / 1024**3
        console.print(
            f"  Loaded via Unsloth | {vram:.1f} GB"
            f"{' | vision encoder removed' if has_vision else ''}",
            style="dim")

        if lora:
            model = FastLanguageModel.get_peft_model(
                model, r=lora_r, lora_alpha=lora_alpha,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=0, bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            console.print(f"  LoRA added | {trainable:,} trainable params | grad ckpt: unsloth",
                           style="dim")

        return model, tokenizer

    except ImportError:
        console.print("  Unsloth not available, trying PEFT...", style="dim")
    except Exception as e:
        console.print(f"  Unsloth failed: {e!s:.150}", style="dim yellow")
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

    # PEFT fallback
    from transformers import AutoTokenizer, BitsAndBytesConfig

    try:
        from transformers import Qwen3_5ForCausalLM as ModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelClass

    console.print(f"  Loading {model_id} via PEFT (8-bit)...", style="dim")
    model = ModelClass.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vram = torch.cuda.memory_allocated() / 1024**3
    console.print(f"  Loaded via PEFT | {vram:.1f} GB", style="dim")

    if lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0, bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"  LoRA added | {trainable:,} trainable params | grad ckpt: non-reentrant",
                       style="dim")

    return model, tokenizer

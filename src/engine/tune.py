"""Lane C LoRA training engine with autoresearch enhancements.

Trains a capability + hedging LoRA adapter on Qwen3.5-9B using captured
experiment data. Detector-only path: no H-neuron intervention at training
time; detector probe used for data selection priority.

Autoresearch enhancements (leaderboard-validated):
  1. Wall-clock training budget (WallClockCallback)
  2. Per-layer residual scalars (ResidualScalars)
  3. Cautious weight decay (CautiousAdamW8bit)
  4. Label smoothing 0.1
  5. GC optimization
  6. Warmdown ratio 0.4-0.5
  7. Final LR fraction 0.005
  8. Linear WD decay schedule
  9. BPB metric logging

Run:
    uv run python -m src.engine.tune --data experiments/ --budget 5
    uv run python -m src.engine.tune --data experiments/ --budget 15 --production
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rich.console import Console

from src.engine.model_loader import load_model

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_SEQ_LENGTH = 2048
DEFAULT_LR = 2e-4
DEFAULT_WD = 0.01
LABEL_SMOOTHING = 0.1
WARMDOWN_RATIO = 0.45        # 45% of training time in LR decay (default for LoRA)
                              # autoresearch@home swarm found 1.0 optimal for from-scratch training
                              # but hyperparams interact: FINAL_LR_FRAC=0.03 + WARMDOWN=1.0 REGRESSED
                              # Test both 0.45 and 1.0 in ablation — don't assume transferability
FINAL_LR_FRACTION = 0.005    # don't decay to exactly 0
GC_COLLECT_INTERVAL = 5000   # manual gc.collect() every N steps
WARMUP_STEPS = 5             # skip from wall-clock budget (torch.compile warmup)


# ── Wall-clock training budget callback ────────────────────────────────────────

class WallClockCallback:
    """Stop training after a fixed wall-clock budget.

    Makes ablation comparisons fair across configs with different per-step
    costs. LR schedule driven by progress = elapsed / budget.
    First WARMUP_STEPS steps excluded from the budget (compile warmup).
    """

    def __init__(self, budget_minutes: float):
        self.budget_seconds = budget_minutes * 60
        self.elapsed = 0.0
        self.step_count = 0
        self._step_start = None
        self._warmup_done = False

    def on_step_begin(self):
        self._step_start = time.monotonic()

    def on_step_end(self) -> bool:
        """Returns True if training should stop."""
        if self._step_start is None:
            return False

        step_time = time.monotonic() - self._step_start
        self.step_count += 1

        # Skip warmup steps from budget
        if self.step_count <= WARMUP_STEPS:
            if self.step_count == WARMUP_STEPS:
                self._warmup_done = True
                torch.cuda.synchronize()
            return False

        self.elapsed += step_time
        return self.elapsed >= self.budget_seconds

    @property
    def progress(self) -> float:
        """Training progress 0.0 to 1.0 (time-proportional)."""
        if self.budget_seconds <= 0:
            return 1.0
        return min(self.elapsed / self.budget_seconds, 1.0)


# ── LR schedule with warmdown ─────────────────────────────────────────────────

def get_lr(
    progress: float,
    base_lr: float = DEFAULT_LR,
    warmup_fraction: float = 0.05,
    warmdown_ratio: float = WARMDOWN_RATIO,
    final_lr_fraction: float = FINAL_LR_FRACTION,
) -> float:
    """Compute LR from time-proportional progress.

    Schedule: linear warmup -> constant -> cosine warmdown to final_lr.
    warmdown_ratio=0.45 means last 45% of budget is cosine decay.
    """
    final_lr = base_lr * final_lr_fraction
    warmdown_start = 1.0 - warmdown_ratio

    if progress < warmup_fraction:
        # Linear warmup
        return base_lr * (progress / warmup_fraction)
    elif progress < warmdown_start:
        # Constant LR
        return base_lr
    else:
        # Cosine warmdown to final_lr
        warmdown_progress = (progress - warmdown_start) / warmdown_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * warmdown_progress))
        return final_lr + (base_lr - final_lr) * cosine_decay


# ── Cautious weight decay + linear WD schedule ────────────────────────────────

def cautious_weight_decay_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    wd: float,
    lr: float,
    progress: float,
):
    """Apply cautious weight decay with linear schedule.

    Cautious: only decay where gradient agrees with parameter sign.
    Linear schedule: WD * (1 - progress) — decay WD itself over training.
    """
    effective_wd = wd * (1.0 - progress)
    if effective_wd <= 0:
        return

    # Cautious mask: only decay where update direction agrees with param sign
    mask = (grad * param.data) >= 0
    param.data.mul_(1.0 - lr * effective_wd * mask.float())


# ── Per-layer residual scalars ─────────────────────────────────────────────────

class ResidualScalars(torch.nn.Module):
    """Learnable per-layer residual scalars.

    x = lambda_resid * x + lambda_x0 * x0

    64 total parameters (~128 bytes). Uses forward pre-hooks on decoder layers.
    Learned lambda values are a free interpretability signal — which layers
    need the original embedding vs learned features.
    """

    def __init__(self, num_layers: int = 32):
        super().__init__()
        self.lambda_resid = torch.nn.Parameter(torch.ones(num_layers))
        self.lambda_x0 = torch.nn.Parameter(torch.full((num_layers,), 0.1))
        self.x0: torch.Tensor | None = None
        self._handles: list = []

    def register_hooks(self, model):
        """Register forward pre-hooks on each decoder layer."""
        # Find decoder layers
        layers = []
        for name, mod in model.named_modules():
            if hasattr(mod, "linear_attn") or hasattr(mod, "self_attn"):
                if "layers." in name and name.count(".") <= 4:
                    layers.append((name, mod))

        if not layers:
            console.print("  [yellow]Warning: No decoder layers found for residual scalars[/yellow]")
            return

        # Hook on embedding output to capture x0
        for name, mod in model.named_modules():
            if "embed_tokens" in name:
                def embed_hook(m, inp, out):
                    # Normalize and store x0
                    self.x0 = torch.nn.functional.normalize(out.detach(), dim=-1)
                self._handles.append(mod.register_forward_hook(embed_hook))
                break

        # Hook on each decoder layer
        for i, (name, layer) in enumerate(layers):
            layer_idx = i

            def pre_hook(m, args, idx=layer_idx):
                if self.x0 is None or len(args) == 0:
                    return args
                hidden = args[0]
                scaled = (self.lambda_resid[idx] * hidden +
                          self.lambda_x0[idx] * self.x0.to(hidden.device))
                return (scaled,) + args[1:]

            self._handles.append(layer.register_forward_pre_hook(pre_hook))

        console.print(f"  Residual scalars: {len(layers)} layers hooked "
                       f"(resid init=1.0, x0 init=0.1)", style="dim")

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Data preparation ───────────────────────────────────────────────────────────

def load_training_data(
    data_dir: Path, tokenizer: "PreTrainedTokenizer",
) -> list[dict]:
    """Load experiment JSON files and format as training examples.

    Returns list of tokenized {input_ids, attention_mask, labels} dicts.
    """
    examples = []

    for json_file in sorted(data_dir.glob("*.json")):
        try:
            exp = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # Only resolved experiments with reference_solution
        if exp.get("status") != "resolved":
            continue
        if not exp.get("reference_solution"):
            continue

        # Build training prompt (pre-solution context only)
        problem = exp.get("problem", "")
        error_output = exp.get("error_output", "") or ""
        context_files = exp.get("pre_solution_context") or []

        context_text = ""
        for cf in context_files[:3]:  # max 3 context files
            context_text += f"\n### {cf.get('path', 'file')}\n{cf.get('content', '')[:2000]}\n"

        system_msg = "You are a helpful coding assistant. Solve the problem accurately."
        user_msg = problem
        if error_output:
            user_msg += f"\n\nError:\n{error_output[:800]}"
        if context_text:
            user_msg += f"\n\nRelevant code:{context_text}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": exp["reference_solution"]},
        ]

        # Tokenize full conversation
        text = tokenizer.apply_chat_template(
            messages, enable_thinking=False, tokenize=False)
        encoded = tokenizer(
            text, truncation=True, max_length=MAX_SEQ_LENGTH,
            return_tensors="pt")

        input_ids = encoded["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Mask labels for non-assistant tokens (only train on the response)
        # Tokenize prompt-only (system + user) to find where assistant starts
        prompt_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, enable_thinking=False, tokenize=False,
            add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]

        # Set labels to -100 for all prompt tokens (model doesn't learn to predict these)
        labels[:prompt_len] = -100

        examples.append({
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "source_file": json_file.name,
        })

    return examples


# ── Training loop ──────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_dir: Path
    budget_minutes: float = 5.0
    lr: float = DEFAULT_LR
    wd: float = DEFAULT_WD
    warmdown_ratio: float = WARMDOWN_RATIO
    label_smoothing: float = LABEL_SMOOTHING
    output_dir: Path = Path("models/lora_adapter")
    production: bool = False


def train(config: TrainConfig):
    """Main training loop with all autoresearch enhancements."""

    console.print(f"[bold]Lane C Training[/bold] | budget: {config.budget_minutes}m | "
                   f"lr: {config.lr} | wd: {config.wd} | label_smoothing: {config.label_smoothing}")

    # Phase 1: Load model with LoRA
    console.print("\n[bold cyan]Phase 1: Model + LoRA[/bold cyan]")
    model, tokenizer = load_model(lora=True)

    # Phase 2: Setup residual scalars
    console.print("\n[bold cyan]Phase 2: Residual Scalars[/bold cyan]")
    residual_scalars = ResidualScalars(num_layers=32)
    residual_scalars.to(next(model.parameters()).device)
    residual_scalars.register_hooks(model)

    # Phase 3: Load data
    console.print("\n[bold cyan]Phase 3: Training Data[/bold cyan]")
    examples = load_training_data(config.data_dir, tokenizer)
    if not examples:
        console.print("[red]No valid training examples found.[/red]")
        residual_scalars.remove_hooks()
        return False

    console.print(f"  {len(examples)} training examples loaded")

    # Phase 4: Setup optimizer (8-bit AdamW + separate groups)
    console.print("\n[bold cyan]Phase 4: Optimizer[/bold cyan]")
    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
        opt_name = "AdamW8bit"
    except ImportError:
        optimizer_cls = torch.optim.AdamW
        opt_name = "AdamW"

    # Separate param groups: LoRA params + residual scalar params
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    scalar_params = list(residual_scalars.parameters())

    optimizer = optimizer_cls([
        {"params": lora_params, "lr": config.lr, "weight_decay": 0.0},  # WD applied manually (cautious)
        {"params": scalar_params, "lr": config.lr / 100, "weight_decay": 0.0},
    ])
    # Store original LRs — must scale from these each step, NOT from pg["lr"]
    # (reading back pg["lr"] after modification causes exponential LR decay bug)
    original_lrs = [pg["lr"] for pg in optimizer.param_groups]
    console.print(f"  {opt_name} | LoRA lr={config.lr}, scalar lr={config.lr/100}")

    # Phase 5: Training loop
    console.print(f"\n[bold cyan]Phase 5: Training ({config.budget_minutes}m wall-clock)[/bold cyan]")

    clock = WallClockCallback(config.budget_minutes)
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
        ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
    )

    model.train()
    step = 0
    epoch = 0
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    log_interval = 10
    should_stop = False

    rng = random.Random(42)  # reproducible shuffle across runs

    while not should_stop:
        epoch += 1
        indices = list(range(len(examples)))
        rng.shuffle(indices)

        for idx in indices:
            example = examples[idx]
            input_ids = example["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(model.device)
            labels = example["labels"].unsqueeze(0).to(model.device)

            clock.on_step_begin()

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Custom loss with label smoothing
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            # Backward
            loss.backward()

            # BPB tracking — proper UTF-8 byte counting per autoresearch methodology
            # BPB = sum(CE_nats) / (ln(2) * sum(utf8_byte_lengths))
            valid_mask = targets != -100
            num_tokens = valid_mask.sum().item()
            # Decode target tokens to text, count actual UTF-8 bytes
            valid_ids = targets[valid_mask].cpu().tolist()
            target_text = tokenizer.decode(valid_ids, skip_special_tokens=False)
            num_bytes = len(target_text.encode("utf-8"))
            total_loss += loss.item() * num_tokens  # CE in nats, summed
            total_tokens += num_tokens
            total_bytes += num_bytes

            # Optimizer step with cautious WD
            current_lr = get_lr(clock.progress, config.lr, warmdown_ratio=config.warmdown_ratio)
            lr_scale = current_lr / config.lr if config.lr > 0 else 1.0
            for pg, orig_lr in zip(optimizer.param_groups, original_lrs):
                pg["lr"] = orig_lr * lr_scale

            optimizer.step()

            # Apply cautious WD manually (after optimizer step)
            for p in lora_params:
                if p.grad is not None:
                    cautious_weight_decay_step(p, p.grad, config.wd, current_lr, clock.progress)

            optimizer.zero_grad()
            step += 1

            # GC optimization
            if step == 1:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif step % GC_COLLECT_INTERVAL == 0:
                gc.enable()
                gc.collect()
                gc.disable()

            # Logging
            if step % log_interval == 0:
                avg_loss_per_token = total_loss / max(total_tokens, 1)
                # Proper BPB: total CE (nats) / (ln(2) * total bytes)
                bpb = (total_loss / max(total_bytes, 1)) / math.log(2)
                console.print(
                    f"  step {step} | loss {loss.item():.4f} | avg {avg_loss_per_token:.4f} | "
                    f"BPB {bpb:.3f} | lr {current_lr:.2e} | "
                    f"progress {clock.progress:.0%} | epoch {epoch}",
                    style="dim")

            # Check wall-clock budget
            if clock.on_step_end():
                should_stop = True
                break

    # Re-enable GC
    gc.enable()
    gc.collect()

    # Final stats
    avg_loss_per_token = total_loss / max(total_tokens, 1)
    bpb = (total_loss / max(total_bytes, 1)) / math.log(2)
    console.print(f"\n  Training complete: {step} steps, {epoch} epochs, "
                   f"avg_loss={avg_loss_per_token:.4f}, BPB={bpb:.3f}, "
                   f"elapsed={clock.elapsed:.0f}s")

    # Phase 6: Save
    console.print("\n[bold cyan]Phase 6: Save Adapter[/bold cyan]")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    # Save residual scalar values (interpretability signal)
    scalar_data = {
        "lambda_resid": residual_scalars.lambda_resid.detach().cpu().tolist(),
        "lambda_x0": residual_scalars.lambda_x0.detach().cpu().tolist(),
    }
    (config.output_dir / "residual_scalars.json").write_text(json.dumps(scalar_data, indent=2))

    # Save training metadata
    metadata = {
        "steps": step,
        "epochs": epoch,
        "budget_minutes": config.budget_minutes,
        "avg_loss": avg_loss_per_token,
        "bpb": bpb,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "lr": config.lr,
        "wd": config.wd,
        "label_smoothing": config.label_smoothing,
        "warmdown_ratio": config.warmdown_ratio,
        "final_lr_fraction": FINAL_LR_FRACTION,
        "num_examples": len(examples),
        "enhancements": [
            "wall_clock_budget",
            "residual_scalars",
            "cautious_weight_decay",
            "label_smoothing_0.1",
            "gc_optimization",
            "warmdown_0.45",
            "final_lr_0.005",
            "linear_wd_decay",
            "bpb_metric",
        ],
    }
    (config.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    residual_scalars.remove_hooks()

    console.print(f"  Adapter saved to [bold]{config.output_dir}[/bold]")
    console.print(f"  Residual scalars: resid={[f'{v:.3f}' for v in scalar_data['lambda_resid'][:4]]}... "
                   f"x0={[f'{v:.3f}' for v in scalar_data['lambda_x0'][:4]]}...")

    return True


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lane C LoRA training with autoresearch enhancements")
    parser.add_argument("--data", type=Path, default=Path("experiments"),
                        help="Directory with experiment JSON files")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="Training budget in minutes (default: 5)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Base learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--wd", type=float, default=DEFAULT_WD,
                        help=f"Weight decay (default: {DEFAULT_WD})")
    parser.add_argument("--warmdown", type=float, default=WARMDOWN_RATIO,
                        help=f"Warmdown ratio 0-1 (default: {WARMDOWN_RATIO}). "
                             f"autoresearch@home swarm found 1.0 optimal for from-scratch, "
                             f"but interacts with final_lr — test both in ablation")
    parser.add_argument("--output", type=Path, default=Path("models/lora_adapter"),
                        help="Output directory for adapter")
    parser.add_argument("--production", action="store_true",
                        help="Use production settings (15m budget)")
    args = parser.parse_args()

    budget = 15.0 if args.production else args.budget

    config = TrainConfig(
        data_dir=args.data,
        budget_minutes=budget,
        lr=args.lr,
        wd=args.wd,
        warmdown_ratio=args.warmdown,
        output_dir=args.output,
    )

    ok = train(config)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

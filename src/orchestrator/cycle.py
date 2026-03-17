"""One full sleep/wake cycle — the core orchestration unit.

Composes engine modules into a 10-step pipeline:
  1. Load model + probe + experiments
  2-6. Per-experiment: wake → dream → compare → calibrate → confidence
  7. Free model, data prep, staging
  8. Train LoRA adapter
  9. Eval + rollback decision
  10. Save CycleResult JSON

"Wrap, don't patch" — no engine module is modified.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from src.engine.abstain import (
    AbstainConfig,
    AbstainMetrics,
    check_abstain_regression,
    generate_abstain_examples,
    identify_abstain_candidates,
    load_previous_metrics,
    save_abstain_metrics,
)
from src.engine.calibrate import CalibrationResult, calibrate_experiment, clear_detector_cache, load_probe
from src.engine.compare import CompareResult, compare_dream_cloud, compare_outputs
from src.engine.confidence import ConfidenceResult, ConfidenceScorer
from src.engine.data_prep import DataPrepConfig, PreparedDataset, prepare_training_data
from src.engine.dream import DreamCloud, dream_from_experiment, structured_dream
from src.engine.model_loader import load_model
from src.engine.tune import TrainConfig, train
from src.engine.wake import WakeResult, wake_from_experiment
from src.eval.metrics import EvalReport, EvalSuite

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CycleConfig:
    data_dir: Path = Path("experiments")
    budget_minutes: float = 5.0
    max_cycle_minutes: float = 60.0
    production: bool = False
    dry_run: bool = False
    skip_dream: bool = False
    use_structured_dream: bool = False
    min_training_examples: int = 10
    output_dir: Path = Path("data/cycle_results")
    adapter_dir: Path = Path("models/lora_adapter")
    probe_path: Path = Path("models/detector_probe_pilot.pkl")
    abstain_mix_ratio: float = 0.10
    abstain_max_rate: float = 0.15


@dataclass
class ExperimentResult:
    experiment_id: str
    wake_result: WakeResult | None = None
    dream_clouds: list[DreamCloud] | None = None
    wake_comparison: CompareResult | None = None
    dream_comparisons: list[CompareResult] | None = None
    calibration: CalibrationResult | None = None
    confidence: ConfidenceResult | None = None
    error: str | None = None


@dataclass
class CycleResult:
    cycle_id: str
    cycle_number: int
    started_at: str
    finished_at: str
    config: dict
    cycle_score: float
    experiment_results: list[ExperimentResult]
    training_ok: bool | None = None
    adapter_path: str | None = None
    eval_report: dict | None = None
    abstain_check: dict | None = None
    data_prep_stats: dict | None = None
    skipped_training: bool = False
    skipped_training_reason: str = ""
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    elapsed_seconds: float = 0.0
    hit_time_limit: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cycle_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _get_cycle_number(output_dir: Path) -> int:
    """Count existing cycle result JSONs + 1."""
    if not output_dir.exists():
        return 1
    pattern = re.compile(r"^\d{8}T\d{6}\.json$")
    count = sum(1 for f in output_dir.iterdir() if pattern.match(f.name))
    return count + 1


def _load_experiments(data_dir: Path) -> list[dict]:
    """Load all valid experiment JSONs — minimal filtering (has id field)."""
    experiments = []
    if not data_dir.exists():
        return experiments
    for path in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("id") is not None:
                experiments.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return experiments


def _find_experiment(experiments: list[dict], exp_id: str) -> dict | None:
    """Lookup experiment by id. Returns None if no match."""
    for exp in experiments:
        if str(exp.get("id", "")) == exp_id:
            return exp
    return None


def _prepare_staging_dir(
    output_dir: Path,
    cycle_id: str,
    prepared_examples: list,
    abstain_examples: list,
    source_data_dir: Path,
) -> Path:
    """Copy curated JSONs + write abstain JSONs into a staging directory.

    Defense-in-depth: strips post_solution_artifacts from copied experiments.
    """
    staging = output_dir / f"{cycle_id}_staging"
    staging.mkdir(parents=True, exist_ok=True)

    copied: set[str] = set()
    for ex in prepared_examples:
        src = source_data_dir / ex.source_file
        if src.exists() and ex.source_file not in copied:
            exp_data = json.loads(src.read_text(encoding="utf-8"))
            exp_data.pop("post_solution_artifacts", None)
            (staging / ex.source_file).write_text(
                json.dumps(exp_data, indent=2), encoding="utf-8"
            )
            copied.add(ex.source_file)

    for i, ab in enumerate(abstain_examples):
        synth = {
            "id": f"abstain_{cycle_id}_{i:04d}",
            "status": "resolved",
            "resolution_type": "answer",
            "problem": ab.user_msg,
            "reference_solution": ab.reference_solution,
            "error_output": None,
            "pre_solution_context": None,
        }
        (staging / f"_abstain_{i:04d}.json").write_text(
            json.dumps(synth, indent=2), encoding="utf-8"
        )

    return staging


def _serialize_config(config: CycleConfig) -> dict:
    """CycleConfig → JSON-safe dict."""
    d = dataclasses.asdict(config)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _safe_value(v: Any) -> Any:
    """Recursively convert non-JSON-safe values."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, dict):
        return {k: _safe_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_value(item) for item in v]
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _safe_value(dataclasses.asdict(v))
    # numpy scalars/arrays
    if hasattr(v, "item"):
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    return str(v)


def _serialize_cycle_result(result: CycleResult) -> dict:
    """CycleResult → JSON-safe dict."""
    return _safe_value(dataclasses.asdict(result))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_cycle(config: CycleConfig) -> CycleResult:
    """Run one full sleep/wake cycle."""
    cycle_id = _make_cycle_id()
    started_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Step 1 — LOAD
    # ------------------------------------------------------------------
    console.print(f"[bold]Cycle {cycle_id}[/bold] — loading model...")
    model, tokenizer = load_model(lora=False)
    model.eval()

    probe = None
    if config.probe_path.exists():
        probe = load_probe(config.probe_path)

    experiments = _load_experiments(config.data_dir)
    console.print(f"  Loaded {len(experiments)} experiments")

    # Warn about failed staging dirs from previous cycles
    if config.output_dir.exists():
        failed_staging = [
            d for d in config.output_dir.iterdir()
            if d.is_dir() and d.name.endswith("_staging") and (d / "_TRAINING_FAILED").exists()
        ]
        for d in failed_staging:
            console.print(
                f"[yellow]Warning: previous failed staging dir: {d}[/yellow]"
            )

    cycle_start = time.monotonic()

    # ------------------------------------------------------------------
    # Steps 2-6 — Per-experiment loop with wall-clock budget
    # ------------------------------------------------------------------
    experiment_results: list[ExperimentResult] = []
    calibration_results: dict[str, CalibrationResult] = {}
    stop_sentinel = Path("data/daemon_stop")
    hit_time_limit = False

    scorer = None
    if probe is not None:
        try:
            scorer = ConfidenceScorer.from_pretrained()
        except Exception:
            scorer = ConfidenceScorer()

    for experiment in experiments:
        elapsed = (time.monotonic() - cycle_start) / 60.0
        if elapsed >= config.max_cycle_minutes:
            hit_time_limit = True
            break

        if stop_sentinel.exists():
            break

        exp_id = str(experiment.get("id", ""))
        result = ExperimentResult(experiment_id=exp_id)
        try:
            # Step 2: WAKE
            wake = wake_from_experiment(model, tokenizer, experiment)
            result.wake_result = wake

            # Step 3: DREAM
            dream_clouds: list[DreamCloud] = []
            if not config.skip_dream:
                cloud = dream_from_experiment(model, tokenizer, experiment)
                dream_clouds = [cloud]
                if config.use_structured_dream:
                    structured_clouds = structured_dream(
                        model,
                        tokenizer,
                        experiment,
                        types=["replay", "false_premise"],
                        n_per_type=2,
                    )
                    dream_clouds.extend(structured_clouds)
            result.dream_clouds = dream_clouds

            # Step 4: COMPARE
            reference = experiment.get("reference_solution", "")
            if reference:
                result.wake_comparison = compare_outputs(
                    model, tokenizer, wake.text, reference, experiment
                )
                dream_cmps: list[CompareResult] = []
                for cloud in dream_clouds:
                    dream_cmps.extend(
                        compare_dream_cloud(
                            model, tokenizer, cloud, reference, experiment
                        )
                    )
                result.dream_comparisons = dream_cmps

            # Step 5: CALIBRATE
            if probe is not None:
                cal = calibrate_experiment(
                    model,
                    tokenizer,
                    probe,
                    experiment,
                    wake_result=wake,
                    dream_clouds=dream_clouds,
                )
                result.calibration = cal
                calibration_results[exp_id] = cal

            # Step 6: CONFIDENCE
            if result.calibration is not None and scorer is not None:
                all_dream_texts: list[str] = []
                for cloud in dream_clouds:
                    all_dream_texts.extend(cloud.texts)

                result.confidence = scorer.score(
                    result.calibration,
                    wake_text=wake.text,
                    dream_texts=all_dream_texts or None,
                )

        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)[:300]}"
            if "OutOfMemory" in type(e).__name__:
                torch.cuda.empty_cache()

        experiment_results.append(result)

    # ------------------------------------------------------------------
    # Defaults for dry_run mode (steps 7-9 skipped)
    # ------------------------------------------------------------------
    cycle_score = 0.0
    cycle_number = _get_cycle_number(config.output_dir)
    training_ok: bool | None = None
    adapter_path: str | None = None
    eval_report: EvalReport | None = None
    data_prep_stats: dict | None = None
    should_rollback = False
    reason = ""
    skipped_training = False
    skipped_training_reason = ""
    staging_dir: Path | None = None

    if not config.dry_run:
        # Ensure output directory exists for eval saves + cycle result
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------
        # Step 7 — FREE MODEL + DATA PREP + STAGING
        # --------------------------------------------------------------
        clear_detector_cache()  # release module_dict refs before freeing model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 7a: Standalone tokenizer for token budget check
        from transformers import AutoTokenizer

        prep_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
        prep_config = DataPrepConfig(data_dir=config.data_dir, split="train")
        prepared = prepare_training_data(prep_config, tokenizer=prep_tokenizer)
        data_prep_stats = prepared.stats
        del prep_tokenizer

        # 7b: Abstain examples
        abstain_config = AbstainConfig(
            data_dir=config.data_dir,
            mix_ratio=config.abstain_mix_ratio,
            max_rate=config.abstain_max_rate,
        )
        candidates = identify_abstain_candidates(experiments, calibration_results)
        abstain_examples = generate_abstain_examples(
            abstain_config, candidates, len(prepared.examples), calibration_results
        )

        # 7c: Staging directory
        total = len(prepared.examples) + len(abstain_examples)
        if total < config.min_training_examples:
            skipped_training = True
            skipped_training_reason = (
                f"Only {total} examples (min: {config.min_training_examples})"
            )
        else:
            staging_dir = _prepare_staging_dir(
                config.output_dir,
                cycle_id,
                prepared.examples,
                abstain_examples,
                config.data_dir,
            )

        console.print(
            f"  Data prep: {len(prepared.examples)} curated + "
            f"{len(abstain_examples)} abstain = {total} total"
        )

        # --------------------------------------------------------------
        # Step 8 — TRAIN (skip if too few examples)
        # --------------------------------------------------------------
        if not skipped_training:
            candidate_adapter = config.output_dir / f"{cycle_id}_adapter"
            train_config = TrainConfig(
                data_dir=staging_dir,
                budget_minutes=(
                    15.0 if config.production else config.budget_minutes
                ),
                output_dir=candidate_adapter,
                production=config.production,
            )
            try:
                training_ok = train(train_config)
            except Exception as e:
                training_ok = False
                console.print(f"[red]Training failed: {e}[/red]")
                torch.cuda.empty_cache()

            # Staging dir cleanup: keep on failure (with marker), clean on success
            if staging_dir is not None and staging_dir.exists():
                if training_ok:
                    shutil.rmtree(staging_dir)
                else:
                    (staging_dir / "_TRAINING_FAILED").write_text(
                        f"Training failed at {datetime.now().isoformat()}\n",
                        encoding="utf-8",
                    )
        else:
            console.print(
                f"[yellow]Skipping training: {skipped_training_reason}[/yellow]"
            )

        # --------------------------------------------------------------
        # Step 9 — EVAL + ROLLBACK DECISION (always runs when not dry_run)
        # --------------------------------------------------------------
        eval_suite = EvalSuite()
        for exp_result in experiment_results:
            if exp_result.error or not exp_result.wake_comparison:
                continue
            exp_dict = _find_experiment(experiments, exp_result.experiment_id)
            if exp_dict is None:
                continue
            correct = exp_result.wake_comparison.classification == "correct"
            tags = exp_dict.get("tags", [])
            domain = tags[0] if tags else "general"
            eval_suite.add(
                query=exp_result.wake_result.query,
                prediction=exp_result.wake_result.text,
                gold=exp_dict.get("reference_solution", ""),
                correct=correct,
                domain=domain,
                confidence=(
                    exp_result.confidence.confidence
                    if exp_result.confidence
                    else None
                ),
                hallucinated=(
                    exp_result.calibration.wake_risk > 0.7
                    if exp_result.calibration
                    else False
                ),
            )
        eval_report = eval_suite.compute()
        eval_suite.save(config.output_dir / f"{cycle_id}_eval.json")

        # Composite scalar — THE metric (C.7: coverage-risk tradeoff)
        cycle_score = (
            eval_report.accuracy * eval_report.coverage
            - 0.5 * eval_report.hallucination_rate
        )

        # Abstain regression guard
        cycle_number = _get_cycle_number(config.output_dir)
        current_metrics = AbstainMetrics(
            cycle=cycle_number,
            total_responses=eval_report.total,
            abstain_count=eval_report.abstain_count,
            accuracy_on_answered=eval_report.normal_accuracy,
        )
        previous = load_previous_metrics(Path("data/abstain_metrics.jsonl"))
        should_rollback, reason = check_abstain_regression(
            current_metrics, previous, config.abstain_max_rate
        )
        save_abstain_metrics(current_metrics, Path("data/abstain_metrics.jsonl"))

        # Operational rollback (only relevant if training happened)
        if not skipped_training and should_rollback and training_ok:
            quarantine = config.output_dir / f"{cycle_id}_adapter_quarantined"
            candidate_adapter.rename(quarantine)
            adapter_path = str(quarantine)
            console.print(
                f"[bold red]ROLLBACK[/bold red]: adapter quarantined → {quarantine}"
            )
            console.print(f"  Reason: {reason}")
        elif not skipped_training and training_ok:
            if config.adapter_dir.exists():
                shutil.rmtree(config.adapter_dir)
            candidate_adapter.rename(config.adapter_dir)
            adapter_path = str(config.adapter_dir)

        console.print(f"  cycle_score = {cycle_score:.4f}")

    # ------------------------------------------------------------------
    # Step 10 — SAVE
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - cycle_start
    successful = sum(1 for r in experiment_results if r.error is None)

    cycle_result = CycleResult(
        cycle_id=cycle_id,
        cycle_number=cycle_number,
        started_at=started_at,
        finished_at=datetime.now().isoformat(),
        config=_serialize_config(config),
        cycle_score=cycle_score,
        experiment_results=experiment_results,
        training_ok=training_ok,
        adapter_path=adapter_path,
        eval_report=(
            dataclasses.asdict(eval_report) if eval_report is not None else None
        ),
        abstain_check={"should_rollback": should_rollback, "reason": reason},
        data_prep_stats=data_prep_stats,
        skipped_training=skipped_training,
        skipped_training_reason=skipped_training_reason,
        total_experiments=len(experiment_results),
        successful_experiments=successful,
        failed_experiments=len(experiment_results) - successful,
        elapsed_seconds=elapsed,
        hit_time_limit=hit_time_limit,
    )

    output_path = config.output_dir / f"{cycle_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_serialize_cycle_result(cycle_result), indent=2),
        encoding="utf-8",
    )

    console.print(
        f"[bold green]Cycle {cycle_id} complete[/bold green] — "
        f"{successful}/{len(experiment_results)} experiments, "
        f"score={cycle_score:.4f}, "
        f"elapsed={elapsed:.0f}s"
    )

    return cycle_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one sleep/wake cycle",
        prog="python -m src.orchestrator.cycle",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--max-cycle-minutes", type=float, default=60.0)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dream", action="store_true")
    parser.add_argument("--structured-dream", action="store_true")
    parser.add_argument("--min-training-examples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cycle_results"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/lora_adapter"))
    parser.add_argument(
        "--probe-path", type=Path, default=Path("models/detector_probe_pilot.pkl")
    )
    parser.add_argument("--abstain-mix-ratio", type=float, default=0.10)
    parser.add_argument("--abstain-max-rate", type=float, default=0.15)
    args = parser.parse_args()

    config = CycleConfig(
        data_dir=args.data_dir,
        budget_minutes=args.budget,
        max_cycle_minutes=args.max_cycle_minutes,
        production=args.production,
        dry_run=args.dry_run,
        skip_dream=args.skip_dream,
        use_structured_dream=args.structured_dream,
        min_training_examples=args.min_training_examples,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        probe_path=args.probe_path,
        abstain_mix_ratio=args.abstain_mix_ratio,
        abstain_max_rate=args.abstain_max_rate,
    )

    try:
        result = run_cycle(config)
        sys.exit(
            0 if (result.training_ok is not False and result.successful_experiments > 0)
            else 1
        )
    except Exception as e:
        console.print(f"[bold red]Cycle failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

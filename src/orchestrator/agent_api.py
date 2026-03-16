"""JSON CLI for agents — structured output to stdout, Rich to stderr.

All commands print valid JSON to stdout (exit 0) or a JSON error body (exit 1).
No interactive prompts. Idempotent where possible.

Usage:
    uv run python -m src.orchestrator.agent_api status
    uv run python -m src.orchestrator.agent_api run-cycle --budget 5 --dry-run
    uv run python -m src.orchestrator.agent_api experiments --split train --limit 20
    uv run python -m src.orchestrator.agent_api results --cycle-id "20260316T143022"
    uv run python -m src.orchestrator.agent_api queue
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console

console = Console(stderr=True)


def _json_out(data: dict | list, exit_code: int = 0) -> None:
    """Print JSON to stdout and exit."""
    print(json.dumps(data, indent=2, default=str))
    sys.exit(exit_code)


def _json_error(message: str, exit_code: int = 1) -> None:
    """Print JSON error to stdout and exit."""
    _json_out({"error": message}, exit_code)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_status(args) -> None:
    """Daemon state, last cycle summary, cycle_score trend."""
    state_path = Path("data/daemon_state.json")
    results_dir = Path("data/cycle_results")

    status: dict = {
        "daemon": None,
        "last_cycle": None,
        "cycle_scores": [],
    }

    # Daemon state
    if state_path.exists():
        try:
            status["daemon"] = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status["daemon"] = {"error": "corrupt state file"}

    # Check daemon PID liveness
    pid_path = Path("data/daemon.pid")
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                status["daemon_alive"] = True
            else:
                status["daemon_alive"] = False
        except Exception:
            status["daemon_alive"] = False
    else:
        status["daemon_alive"] = False

    # Cycle results — latest + score trend
    if results_dir.exists():
        import re

        pattern = re.compile(r"^\d{8}T\d{6}\.json$")
        cycle_files = sorted(
            [f for f in results_dir.iterdir() if pattern.match(f.name)],
            key=lambda f: f.name,
        )

        for cf in cycle_files:
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
                status["cycle_scores"].append(
                    {
                        "cycle_id": data.get("cycle_id"),
                        "cycle_score": data.get("cycle_score"),
                        "total_experiments": data.get("total_experiments"),
                        "training_ok": data.get("training_ok"),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue

        if cycle_files:
            try:
                status["last_cycle"] = json.loads(
                    cycle_files[-1].read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

    _json_out(status)


def cmd_run_cycle(args) -> None:
    """Run one cycle synchronously, return results."""
    from src.orchestrator.cycle import CycleConfig, run_cycle

    config = CycleConfig(
        budget_minutes=args.budget,
        dry_run=args.dry_run,
        max_cycle_minutes=args.max_cycle_minutes,
        production=args.production,
    )

    try:
        result = run_cycle(config)
        _json_out(
            {
                "cycle_id": result.cycle_id,
                "cycle_score": result.cycle_score,
                "total_experiments": result.total_experiments,
                "successful_experiments": result.successful_experiments,
                "failed_experiments": result.failed_experiments,
                "training_ok": result.training_ok,
                "skipped_training": result.skipped_training,
                "elapsed_seconds": result.elapsed_seconds,
                "adapter_path": result.adapter_path,
            }
        )
    except Exception as e:
        _json_error(f"Cycle failed: {type(e).__name__}: {e}")


def cmd_experiments(args) -> None:
    """List experiments with metadata."""
    from src.engine.data_prep import DataPrepConfig, prepare_training_data

    config = DataPrepConfig(data_dir=Path("experiments"), split=args.split)

    try:
        prepared = prepare_training_data(config, tokenizer=None)
        examples = prepared.examples[: args.limit] if args.limit else prepared.examples

        _json_out(
            {
                "split": args.split,
                "total": len(prepared.examples),
                "returned": len(examples),
                "stats": prepared.stats,
                "experiments": [
                    {
                        "source_file": ex.source_file,
                        "domain": ex.domain,
                        "task_group_id": ex.task_group_id,
                        "resolution_type": ex.resolution_type,
                        "priority": ex.priority,
                    }
                    for ex in examples
                ],
            }
        )
    except Exception as e:
        _json_error(f"Failed to list experiments: {type(e).__name__}: {e}")


def cmd_results(args) -> None:
    """Show cycle results."""
    import re

    results_dir = Path("data/cycle_results")

    if not results_dir.exists():
        _json_out({"cycles": [], "total": 0})

    if args.cycle_id:
        path = results_dir / f"{args.cycle_id}.json"
        if not path.exists():
            _json_error(f"Cycle not found: {args.cycle_id}")
        try:
            _json_out(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            _json_error(f"Failed to read cycle: {e}")
    else:
        pattern = re.compile(r"^\d{8}T\d{6}\.json$")
        cycle_files = sorted(
            [f for f in results_dir.iterdir() if pattern.match(f.name)],
            key=lambda f: f.name,
        )
        summaries = []
        for cf in cycle_files:
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "cycle_id": data.get("cycle_id"),
                        "cycle_score": data.get("cycle_score"),
                        "total_experiments": data.get("total_experiments"),
                        "training_ok": data.get("training_ok"),
                        "elapsed_seconds": data.get("elapsed_seconds"),
                        "skipped_training": data.get("skipped_training"),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue

        _json_out({"cycles": summaries, "total": len(summaries)})


def cmd_queue(args) -> None:
    """Show pending experiment count and batch info."""
    import re

    experiments_dir = Path("experiments")
    state_path = Path("data/daemon_state.json")

    all_files: set[str] = set()
    if experiments_dir.exists():
        all_files = {f.name for f in experiments_dir.glob("*.json")}

    processed: set[str] = set()
    batch_size = 10
    last_cycle_time: str | None = None

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            processed = set(state.get("processed_files", []))
            batch_size = state.get("batch_size", 10)
            last_cycle_time = state.get("last_cycle_time")
        except (json.JSONDecodeError, OSError):
            pass

    pending = all_files - processed

    _json_out(
        {
            "pending_count": len(pending),
            "total_experiments": len(all_files),
            "processed": len(processed),
            "batch_size": batch_size,
            "needs_trigger": len(pending) >= batch_size,
            "last_cycle_time": last_cycle_time,
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Agent API — JSON CLI for sleep/wake orchestration",
        prog="python -m src.orchestrator.agent_api",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Daemon state + cycle score trend")

    # run-cycle
    rc = sub.add_parser("run-cycle", help="Run one cycle synchronously")
    rc.add_argument("--budget", type=float, default=5.0)
    rc.add_argument("--dry-run", action="store_true")
    rc.add_argument("--max-cycle-minutes", type=float, default=60.0)
    rc.add_argument("--production", action="store_true")

    # experiments
    ex = sub.add_parser("experiments", help="List experiments with metadata")
    ex.add_argument("--split", default="train")
    ex.add_argument("--limit", type=int, default=50)

    # results
    rs = sub.add_parser("results", help="Show cycle results")
    rs.add_argument("--cycle-id", default=None)

    # queue
    sub.add_parser("queue", help="Pending experiment count + batch info")

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "run-cycle": cmd_run_cycle,
        "experiments": cmd_experiments,
        "results": cmd_results,
        "queue": cmd_queue,
    }

    try:
        commands[args.command](args)
    except SystemExit:
        raise
    except Exception as e:
        _json_error(f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

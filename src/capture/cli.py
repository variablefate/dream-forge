"""
CLI for dream-forge experiment capture.

Commands:
  add       — interactive experiment entry
  validate  — validate a JSON experiment file against the schema
  review-groups — review and confirm task_group_id assignments before splitting
  list      — list captured experiments
  stats     — show dataset statistics
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.capture.schema import (
    Experiment,
    ExperimentSource,
    ExperimentStatus,
    ResolutionType,
)
from src.store.db import ExperimentStore

app = typer.Typer(name="dream-forge", help="Experiment capture and management CLI")
console = Console()


@app.command()
def validate(
    file: Path = typer.Argument(..., help="Path to experiment JSON file"),
    fix: bool = typer.Option(False, help="Attempt to fix common issues"),
) -> None:
    """Validate a JSON experiment file against the Pydantic schema.

    Structural + non-empty required fields check. Max 2 retries on fix.
    Non-zero exit code on failure.
    """
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(code=1)

    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        raise typer.Exit(code=1)

    try:
        exp = Experiment.model_validate(data)
    except Exception as e:
        console.print(f"[red]Schema validation failed:[/red] {e}")
        raise typer.Exit(code=1)

    # Semantic checks
    issues = []
    if exp.status == ExperimentStatus.RESOLVED and not exp.reference_solution:
        issues.append("Resolved experiment missing reference_solution")
    if exp.status == ExperimentStatus.RESOLVED and not exp.resolution_type:
        issues.append("Resolved experiment missing resolution_type")
    if not exp.problem.strip():
        issues.append("problem field is empty")
    if not exp.task_group_id.strip():
        issues.append("task_group_id is empty")

    if issues:
        console.print("[yellow]Semantic issues found:[/yellow]")
        for issue in issues:
            console.print(f"  - {issue}")
        raise typer.Exit(code=1)

    console.print(f"[green]Valid experiment:[/green] {exp.id} ({exp.status.value})")


@app.command()
def add(
    source: str = typer.Option("manual", help="Source: claude, codex, or manual"),
    project: str = typer.Option("default", help="Project name"),
    output_dir: Path = typer.Option(Path("experiments"), help="Output directory"),
) -> None:
    """Interactively create a new experiment."""
    console.print("[bold]Dream Forge — New Experiment[/bold]\n")

    problem = typer.prompt("Problem description")
    breakdown_raw = typer.prompt("Breakdown (comma-separated sub-tasks)", default="")
    breakdown = [s.strip() for s in breakdown_raw.split(",") if s.strip()]

    status_str = typer.prompt("Status (resolved/unresolved)", default="resolved")
    status = ExperimentStatus(status_str)

    reference_solution = None
    resolution_type = None
    if status == ExperimentStatus.RESOLVED:
        reference_solution = typer.prompt("Reference solution (final correct code/answer)")
        res_type = typer.prompt(
            "Resolution type (code_change/answer/config_change/research_finding)",
            default="code_change",
        )
        resolution_type = ResolutionType(res_type)

    tags_raw = typer.prompt("Tags (comma-separated)", default="")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    task_group_id = typer.prompt("Task group ID (shared across related experiments)")

    exp = Experiment(
        source=ExperimentSource(source),
        project=project,
        problem=problem,
        breakdown=breakdown,
        status=status,
        task_group_id=task_group_id,
        reference_solution=reference_solution,
        resolution_type=resolution_type,
        tags=tags,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = problem[:40].replace(" ", "-").lower()
    filename = f"{exp.timestamp.strftime('%Y%m%d-%H%M%S')}-{slug}.json"
    filepath = output_dir / filename

    filepath.write_text(exp.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Saved:[/green] {filepath}")

    # Insert into LanceDB
    try:
        store = ExperimentStore()
        store.insert(exp)
        console.print(f"[green]Indexed in LanceDB:[/green] {exp.id}")
    except Exception as e:
        console.print(f"[yellow]LanceDB indexing failed (non-fatal):[/yellow] {e}")


@app.command(name="review-groups")
def review_groups() -> None:
    """Review task_group_id assignments before generating train/validation/test splits."""
    store = ExperimentStore()
    experiments = store.list_all(exclude_superseded=True, exclude_synthetic=True)

    groups: dict[str, list[Experiment]] = {}
    for exp in experiments:
        groups.setdefault(exp.task_group_id, []).append(exp)

    table = Table(title="Task Groups")
    table.add_column("Group ID", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Projects")
    table.add_column("Status Mix")

    for gid, exps in sorted(groups.items()):
        projects = ", ".join(sorted(set(e.project for e in exps)))
        statuses = ", ".join(sorted(set(e.status.value for e in exps)))
        table.add_row(gid, str(len(exps)), projects, statuses)

    console.print(table)
    console.print(f"\n[bold]{len(groups)} groups, {len(experiments)} experiments[/bold]")


@app.command(name="list")
def list_experiments(
    resolved_only: bool = typer.Option(False, help="Show only resolved"),
    limit: int = typer.Option(20, help="Max results"),
) -> None:
    """List captured experiments."""
    store = ExperimentStore()
    experiments = store.list_all(resolved_only=resolved_only)[:limit]

    table = Table(title="Experiments")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Status")
    table.add_column("Problem", max_width=50)
    table.add_column("Tags")
    table.add_column("Source")

    for exp in experiments:
        table.add_row(
            str(exp.id)[:8],
            exp.status.value,
            exp.problem[:50],
            ", ".join(exp.tags[:3]),
            exp.source.value,
        )

    console.print(table)


@app.command()
def stats() -> None:
    """Show dataset statistics."""
    store = ExperimentStore()
    resolved = store.list_all(resolved_only=True, exclude_synthetic=True)
    total = store.count()
    resolved_count = store.count(resolved_only=True)

    console.print(f"[bold]Dataset Statistics[/bold]")
    console.print(f"  Total experiments: {total}")
    console.print(f"  Resolved: {resolved_count}")
    console.print(f"  Unresolved: {total - resolved_count}")

    if resolved:
        tags_count: dict[str, int] = {}
        for exp in resolved:
            for tag in exp.tags:
                tags_count[tag] = tags_count.get(tag, 0) + 1
        if tags_count:
            console.print(f"\n  Tags distribution:")
            for tag, count in sorted(tags_count.items(), key=lambda x: -x[1]):
                console.print(f"    {tag}: {count}")


if __name__ == "__main__":
    app()

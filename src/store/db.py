"""
LanceDB wrapper for experiment storage.

Uses separate tables for resolved vs unresolved experiments to prevent
accidental retrieval of unresolved data in training pipelines.

Vector columns: problem_vec, breakdown_vec, reference_solution_vec (resolved only).
"""

import json
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

from src.capture.schema import Experiment, ExperimentStatus
from src.store.embeddings import embed_text, embed_optional

DB_PATH = Path("data/experiments.lance")

# Schema for LanceDB tables
_RESOLVED_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("task_group_id", pa.string()),
    pa.field("project", pa.string()),
    pa.field("status", pa.string()),
    pa.field("source", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("problem", pa.string()),
    pa.field("reference_solution", pa.string()),
    pa.field("resolution_type", pa.string()),
    pa.field("tags", pa.string()),  # JSON-encoded list
    pa.field("confidence", pa.string()),
    pa.field("synthetic", pa.bool_()),
    pa.field("superseded", pa.bool_()),
    pa.field("retrieval_count", pa.int32()),
    pa.field("positive_outcome_count", pa.int32()),
    pa.field("data_json", pa.string()),  # Full experiment as JSON for reconstruction
    pa.field("problem_vec", pa.list_(pa.float32(), 384)),
    pa.field("breakdown_vec", pa.list_(pa.float32(), 384)),
    pa.field("reference_solution_vec", pa.list_(pa.float32(), 384)),
])

_UNRESOLVED_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("task_group_id", pa.string()),
    pa.field("project", pa.string()),
    pa.field("status", pa.string()),
    pa.field("source", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("problem", pa.string()),
    pa.field("tags", pa.string()),
    pa.field("superseded", pa.bool_()),
    pa.field("data_json", pa.string()),
    pa.field("problem_vec", pa.list_(pa.float32(), 384)),
    pa.field("breakdown_vec", pa.list_(pa.float32(), 384)),
])


class ExperimentStore:
    """LanceDB-backed experiment store with separate resolved/unresolved tables."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    def _get_or_create_table(self, name: str, schema: pa.Schema):
        if name in self.db.table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    @property
    def resolved_table(self):
        return self._get_or_create_table("experiments_resolved", _RESOLVED_SCHEMA)

    @property
    def unresolved_table(self):
        return self._get_or_create_table("experiments_unresolved", _UNRESOLVED_SCHEMA)

    def insert(self, experiment: Experiment) -> str:
        """Insert an experiment into the appropriate table. Returns the experiment ID."""
        problem_vec = embed_text(experiment.problem)
        breakdown_vec = embed_text(" ".join(experiment.breakdown)) if experiment.breakdown else embed_text("")

        exp_id = str(experiment.id)
        data_json = experiment.model_dump_json()

        if experiment.status == ExperimentStatus.RESOLVED:
            ref_vec = embed_text(experiment.reference_solution or "")
            row = {
                "id": exp_id,
                "task_group_id": experiment.task_group_id,
                "project": experiment.project,
                "status": experiment.status.value,
                "source": experiment.source.value,
                "timestamp": experiment.timestamp.isoformat(),
                "problem": experiment.problem,
                "reference_solution": experiment.reference_solution or "",
                "resolution_type": experiment.resolution_type.value if experiment.resolution_type else "",
                "tags": json.dumps(experiment.tags),
                "confidence": experiment.confidence.value,
                "synthetic": experiment.synthetic,
                "superseded": experiment.superseded,
                "retrieval_count": experiment.retrieval_count,
                "positive_outcome_count": experiment.positive_outcome_count,
                "data_json": data_json,
                "problem_vec": problem_vec,
                "breakdown_vec": breakdown_vec,
                "reference_solution_vec": ref_vec,
            }
            self.resolved_table.add([row])
        else:
            row = {
                "id": exp_id,
                "task_group_id": experiment.task_group_id,
                "project": experiment.project,
                "status": experiment.status.value,
                "source": experiment.source.value,
                "timestamp": experiment.timestamp.isoformat(),
                "problem": experiment.problem,
                "tags": json.dumps(experiment.tags),
                "superseded": experiment.superseded,
                "data_json": data_json,
                "problem_vec": problem_vec,
                "breakdown_vec": breakdown_vec,
            }
            self.unresolved_table.add([row])

        return exp_id

    def search_similar_problems(
        self,
        query: str,
        limit: int = 5,
        resolved_only: bool = True,
        exclude_superseded: bool = True,
    ) -> list[Experiment]:
        """Search for experiments with similar problems."""
        query_vec = embed_text(query)
        table = self.resolved_table if resolved_only else self.unresolved_table

        results = table.search(query_vec, vector_column_name="problem_vec").limit(limit)

        if exclude_superseded:
            results = results.where("superseded = false")

        rows = results.to_list()
        return [Experiment.model_validate_json(r["data_json"]) for r in rows]

    def get_by_id(self, exp_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID. Searches both tables."""
        for table in [self.resolved_table, self.unresolved_table]:
            rows = table.search().where(f"id = '{exp_id}'").limit(1).to_list()
            if rows:
                return Experiment.model_validate_json(rows[0]["data_json"])
        return None

    def list_all(
        self,
        resolved_only: bool = False,
        exclude_superseded: bool = True,
        exclude_synthetic: bool = True,
    ) -> list[Experiment]:
        """List all experiments matching filters."""
        experiments = []

        for table, is_resolved in [(self.resolved_table, True), (self.unresolved_table, False)]:
            if resolved_only and not is_resolved:
                continue

            query = table.search()
            filters = []
            if exclude_superseded:
                filters.append("superseded = false")
            if exclude_synthetic and is_resolved:
                filters.append("synthetic = false")
            if filters:
                query = query.where(" AND ".join(filters))

            rows = query.limit(10000).to_list()
            experiments.extend(
                Experiment.model_validate_json(r["data_json"]) for r in rows
            )

        return experiments

    def mark_superseded(self, exp_id: str) -> None:
        """Mark an unresolved experiment as superseded (a resolved child exists)."""
        # LanceDB update: find and update the row
        table = self.unresolved_table
        rows = table.search().where(f"id = '{exp_id}'").limit(1).to_list()
        if rows:
            row = rows[0]
            row["superseded"] = True
            # LanceDB doesn't have native update — delete and re-add
            table.delete(f"id = '{exp_id}'")
            table.add([row])

    def count(self, resolved_only: bool = False) -> int:
        """Count experiments."""
        total = len(self.resolved_table)
        if not resolved_only:
            total += len(self.unresolved_table)
        return total

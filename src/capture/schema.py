"""
Experiment data schema for the dream-forge capture pipeline.

Each experiment captures a 5-stage problem-solving arc from Claude Code or Codex
conversations, plus optional artifacts for verification and provenance.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResolutionType(str, Enum):
    CODE_CHANGE = "code_change"
    ANSWER = "answer"
    CONFIG_CHANGE = "config_change"
    RESEARCH_FINDING = "research_finding"


class ExperimentSource(str, Enum):
    CLAUDE = "claude"
    CODEX = "codex"
    MANUAL = "manual"


class ExperimentStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class ConfidenceTier(str, Enum):
    """Memory governance confidence tiers."""
    VERIFIED = "verified"      # Has Tier 1a test/assertion evidence
    OBSERVED = "observed"      # Has Tier 1b build/lint evidence or Tier 2 embedding similarity
    INFERRED = "inferred"      # Tier 3 Qwen self-judge only
    STALE = "stale"            # Auto-assigned if >90 days since last retrieval AND not verified


class SyntheticGenerator(str, Enum):
    """Types of synthetic dream generation."""
    FALSE_PREMISE = "false_premise"
    COUNTERFACTUAL = "counterfactual"
    REPLAY = "replay"
    HIGH_TEMP = "high_temp"


class ContextFile(BaseModel):
    """A file snippet captured as task context."""
    path: str
    content: str
    revision: Optional[str] = None
    provenance: str  # "error_trace" | "user_provided" | "retrieved_pre" | "materialized_fallback"


class ReviewRound(BaseModel):
    """One round of plan review with issues found and corrections made."""
    round_number: int
    issues_found: list[str]
    corrections_made: list[str]


class TestResult(BaseModel):
    """Tier 1a — strongest label quality."""
    passed: bool
    output: str
    assertion_details: Optional[str] = None


class BuildResult(BaseModel):
    """Tier 1b — weaker label quality."""
    passed: bool
    output: str


class LintResult(BaseModel):
    """Tier 1b — weaker label quality."""
    passed: bool
    output: str


class Experiment(BaseModel):
    """
    Core data model for a captured problem-solving experiment.

    The 5 stages capture the full problem-solving arc:
    1. problem — what the user needs solved
    2. breakdown — initial decomposition into sub-tasks
    3. proposed_solutions — initial plan approaches
    4. review_issues — problems found during review, with corrections per round
    5. final_plan — the corrected planning prose

    Plus reference_solution (the actual resolved outcome) and artifacts for verification.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4)
    source: ExperimentSource
    timestamp: datetime = Field(default_factory=datetime.now)
    project: str

    # 5 core stages
    problem: str
    breakdown: list[str] = Field(default_factory=list)
    proposed_solutions: list[str] = Field(default_factory=list)
    review_issues: list[ReviewRound] = Field(default_factory=list)
    final_plan: str = ""

    # Resolution status and grouping
    status: ExperimentStatus
    task_group_id: str  # shared across experiments about the same task/bug/feature
    superseded: bool = False  # True once a resolved child exists; hidden from default retrieval

    # The actual final answer/implementation — only required for resolved experiments
    # For code tasks: final correct code (function/class/block), NOT a unified diff.
    # For non-code tasks: structured summary of the final answer.
    reference_solution: Optional[str] = None
    resolution_type: Optional[ResolutionType] = None

    # Task context — strictly split by temporal provenance
    pre_solution_context: Optional[list[ContextFile]] = None
    post_solution_artifacts: Optional[list[ContextFile]] = None
    repo_hash: Optional[str] = None  # git revision hash at task start
    repo_dirty: bool = False  # True if working tree had uncommitted changes

    # Git-based provenance
    git_diff: Optional[str] = None  # stored for provenance/debugging, NOT the SFT training target
    git_start_hash: Optional[str] = None  # commit hash at task start

    # Tier 1a (strong labels — feed neuron claims + calibrator)
    test_results: Optional[TestResult] = None

    # Tier 1b (weaker labels — useful context, not for neuron claims alone)
    build_results: Optional[BuildResult] = None
    lint_results: Optional[LintResult] = None
    error_logs: Optional[list[str]] = None
    # Note: per-file diffs removed — use git_diff (consolidated) for all diff needs

    # Commands and error context
    commands_run: Optional[list[str]] = None
    error_output: Optional[str] = None
    constraints: Optional[list[str]] = None

    # Lifecycle
    resolves_experiment_id: Optional[UUID] = None  # links resolved child to unresolved parent

    # Synthetic replay provenance
    synthetic: bool = False
    generator: Optional[SyntheticGenerator] = None
    parent_experiment_id: Optional[UUID] = None
    generation_depth: int = 0  # 0 = real, 1 = synthetic child (max depth)

    # Metadata
    tags: list[str] = Field(default_factory=list)  # domain tags (python, android, web, etc.)
    confidence: ConfidenceTier = ConfidenceTier.INFERRED

    # Memory governance
    retrieval_count: int = 0
    positive_outcome_count: int = 0
    last_retrieved: Optional[datetime] = None


class SplitAssignment(BaseModel):
    """Tracks which split a task_group_id belongs to. Immutable once assigned."""
    task_group_id: str
    split: str  # "train" | "validation" | "frozen_test" | "archival"
    assigned_at: datetime = Field(default_factory=datetime.now)


class SplitAssignmentTable(BaseModel):
    """Persistent split assignment lookup. New groups default to 'train'."""
    assignments: dict[str, SplitAssignment] = Field(default_factory=dict)

    def get_split(self, task_group_id: str) -> str:
        if task_group_id in self.assignments:
            return self.assignments[task_group_id].split
        return "train"  # default for new groups

    def assign(self, task_group_id: str, split: str) -> None:
        if task_group_id in self.assignments:
            raise ValueError(
                f"task_group_id '{task_group_id}' already assigned to "
                f"'{self.assignments[task_group_id].split}'. Split assignments are immutable."
            )
        self.assignments[task_group_id] = SplitAssignment(
            task_group_id=task_group_id, split=split
        )

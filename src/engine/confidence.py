"""C.6 Multi-signal confidence scoring.

Combines detector probe probability, sleep divergence, and token logprobs
into a calibrated confidence score. Detector-only path: detector used for
eval analysis and confidence scoring, not runtime gating.

Two modes:
  - Raw mode (< 100 labeled examples): reports individual signals, no
    calibrated score, no ECE/Brier claims.
  - Calibrated mode (>= 100 Tier 1a + gold-set examples): trained logistic
    regression produces a calibrated confidence. ECE/Brier reportable.

Usage:
    from src.engine.confidence import ConfidenceScorer, ConfidenceResult
    scorer = ConfidenceScorer()  # raw mode
    result = scorer.score(calibration_result, wake_result, dream_clouds, reference)

    # After collecting labeled data:
    scorer = ConfidenceScorer.from_labeled_data(labeled_examples)  # calibrated mode
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from src.engine.calibrate import CalibrationResult

console = Console()

CALIBRATED_MODE_THRESHOLD = 100  # Tier 1a + gold-set examples needed
CALIBRATOR_PATH = Path("models/confidence_calibrator.pkl")


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceSignals:
    """Individual confidence signals before calibration."""
    detector_confidence: float      # 1 - P(hallucination) from detector probe
    sleep_divergence: float         # embedding distance wake vs dream centroid
    avg_logprob: float | None       # mean token log-probability (if available)
    answer_length: int              # response length in characters
    n_dream_samples: int            # number of dream samples scored

    def as_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for the calibrator."""
        return np.array([
            self.detector_confidence,
            self.sleep_divergence,
            self.avg_logprob if self.avg_logprob is not None else 0.0,
            min(self.answer_length / 1000.0, 5.0),  # normalize, cap at 5
        ], dtype=np.float32)


@dataclass
class ConfidenceResult:
    """Confidence assessment for a single experiment output."""
    signals: ConfidenceSignals
    calibrated_confidence: float | None = None  # None if raw mode
    mode: str = "raw"              # "raw" | "calibrated"

    @property
    def confidence(self) -> float:
        """Best available confidence estimate."""
        if self.calibrated_confidence is not None:
            return self.calibrated_confidence
        return self.signals.detector_confidence


# ── Sleep divergence ──────────────────────────────────────────────────────────

def compute_sleep_divergence(
    wake_text: str,
    dream_texts: list[str],
) -> float:
    """Embedding distance between wake output and dream cloud centroid.

    Higher divergence = model is less certain (wake and dream disagree).
    Returns a value in [0, 2] (cosine distance).
    """
    if not wake_text or not dream_texts:
        return 1.0  # maximum uncertainty if either is missing

    from src.store.embeddings import embed_texts

    all_texts = [wake_text] + dream_texts
    vecs = embed_texts(all_texts)

    wake_vec = np.array(vecs[0])
    dream_vecs = np.array(vecs[1:])
    centroid = dream_vecs.mean(axis=0)

    # Cosine distance = 1 - cosine_similarity
    dot = np.dot(wake_vec, centroid)
    norm = np.linalg.norm(wake_vec) * np.linalg.norm(centroid)
    if norm == 0:
        return 1.0
    similarity = dot / norm
    return float(1.0 - similarity)


# ── Confidence scorer ─────────────────────────────────────────────────────────

class ConfidenceScorer:
    """Multi-signal confidence scorer.

    Raw mode: returns individual signals without calibration.
    Calibrated mode: applies trained logistic regression to produce
    a calibrated confidence score.
    """

    def __init__(self, calibrator=None):
        self.calibrator = calibrator
        self.mode = "calibrated" if calibrator is not None else "raw"

    @classmethod
    def from_pretrained(cls, path: Path = CALIBRATOR_PATH) -> "ConfidenceScorer":
        """Load a trained calibrator from disk."""
        import joblib
        calibrator = joblib.load(str(path))
        return cls(calibrator=calibrator)

    @classmethod
    def from_labeled_data(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> "ConfidenceScorer":
        """Train a calibrator from labeled examples.

        Args:
            features: [N, 4] array of signal features.
            labels: [N] binary array (1 = correct, 0 = incorrect).
                    Only use Tier 1a + gold-set labels.

        Returns:
            ConfidenceScorer in calibrated mode.
        """
        if len(features) < CALIBRATED_MODE_THRESHOLD:
            console.print(
                f"[yellow]Warning: only {len(features)} labeled examples "
                f"(need {CALIBRATED_MODE_THRESHOLD} for calibrated mode). "
                f"Using raw mode.[/yellow]")
            return cls(calibrator=None)

        if len(np.unique(labels)) < 2:
            console.print(
                "[yellow]Warning: only one class in labeled data — "
                "cannot train calibrator. Using raw mode.[/yellow]")
            return cls(calibrator=None)

        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=42)
        calibrator.fit(features, labels)

        accuracy = calibrator.score(features, labels)
        console.print(
            f"  Calibrator trained: {len(features)} examples, "
            f"training accuracy {accuracy:.1%}")

        return cls(calibrator=calibrator)

    def save(self, path: Path = CALIBRATOR_PATH):
        """Save calibrator to disk."""
        if self.calibrator is None:
            console.print("[yellow]No calibrator to save (raw mode).[/yellow]")
            return
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.calibrator, str(path))
        console.print(f"  Calibrator saved to {path}")

    def score(
        self,
        calibration_result: "CalibrationResult",
        wake_text: str = "",
        dream_texts: list[str] | None = None,
        avg_logprob: float | None = None,
    ) -> ConfidenceResult:
        """Compute confidence for a calibrated experiment.

        Args:
            calibration_result: Output from calibrate.py with detector scores.
            wake_text: The wake output text.
            dream_texts: All dream sample texts.
            avg_logprob: Mean token log-probability from generation (optional).

        Returns:
            ConfidenceResult with signals and optional calibrated score.
        """
        # Detector confidence
        detector_conf = 1.0 - calibration_result.wake_risk

        # Sleep divergence
        divergence = compute_sleep_divergence(
            wake_text, dream_texts or [])

        # Build signals
        signals = ConfidenceSignals(
            detector_confidence=detector_conf,
            sleep_divergence=divergence,
            avg_logprob=avg_logprob,
            answer_length=len(wake_text),
            n_dream_samples=len(dream_texts) if dream_texts else 0,
        )

        # Calibrated mode: apply trained model
        calibrated = None
        if self.calibrator is not None:
            features = signals.as_feature_vector().reshape(1, -1)
            proba = self.calibrator.predict_proba(features)[0]
            # P(correct) is calibrated confidence
            calibrated = float(proba[1]) if len(proba) > 1 else float(proba[0])

        return ConfidenceResult(
            signals=signals,
            calibrated_confidence=calibrated,
            mode=self.mode,
        )

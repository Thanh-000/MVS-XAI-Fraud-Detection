"""ADWIN-based online drift monitor with graceful fallback."""
from __future__ import annotations

import numpy as np

try:
    from river.drift import ADWIN
except ImportError:
    ADWIN = None


class ADWINDriftMonitor:
    """Sequential score-drift detector using river's ADWIN implementation."""

    def __init__(self, delta=0.002):
        self.delta = delta

    def monitor_score_stream(self, reference_scores, observed_scores):
        reference_scores = np.asarray(reference_scores, dtype=float)
        observed_scores = np.asarray(observed_scores, dtype=float)

        if ADWIN is None:
            print("\n  ADWIN Monitor: river is not installed; skipping ADWIN drift check.")
            return {"available": False, "change_points": []}

        detector = ADWIN(delta=self.delta)
        change_points = []
        full_stream = np.concatenate([reference_scores, observed_scores])

        for idx, score in enumerate(full_stream):
            detector.update(float(score))
            if detector.drift_detected:
                change_points.append(idx)

        observed_changes = [idx - len(reference_scores) for idx in change_points if idx >= len(reference_scores)]

        print("\n  ADWIN Score Drift:")
        print(f"    Delta: {self.delta}")
        print(f"    Detected change points in observed stream: {len(observed_changes)}")
        if observed_changes:
            preview = ", ".join(str(idx) for idx in observed_changes[:10])
            print(f"    First observed change indices: {preview}")

        return {
            "available": True,
            "change_points": observed_changes,
            "num_changes": len(observed_changes),
        }

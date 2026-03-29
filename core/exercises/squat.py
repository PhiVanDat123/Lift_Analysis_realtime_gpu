from __future__ import annotations

import numpy as np
from core.exercises.base import BaseExercise


class SquatChecker(BaseExercise):

    KNEE_DOWN = 100
    KNEE_UP   = 160

    MIN_BAR_VERTICAL = 0.03

    TORSO_STD_THRESHOLD   = 12
    BAR_DRIFT_THRESHOLD   = 0.05

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._torso_angles: list[float] = []
        self._min_knee: float = 180.0

    def reset_state(self) -> None:
        super().reset_state()
        self._torso_angles.clear()
        self._min_knee = 180.0

    def check(self, angles, kp, stage):  
        angles   = self.smooth(angles)
        avg_knee = (angles["left_knee"] + angles["right_knee"]) / 2
        new_stage = stage
        if avg_knee <= self.KNEE_DOWN:
            new_stage = "down"
        elif avg_knee >= self.KNEE_UP:
            new_stage = "up"

        rep_completed = new_stage == "up" and stage == "down"

        if rep_completed and self.barbell_tracker and self.barbell_tracker.enabled:
            v_range = self.barbell_tracker.vertical_range()
            if v_range is not None and v_range < self.MIN_BAR_VERTICAL:
                rep_completed = False
        issues: list[str] = []

        avg_hip = (angles["left_hip"] + angles["right_hip"]) / 2

        if new_stage == "down":
            self._min_knee = min(self._min_knee, avg_knee)

        # Track torso angle across full rep (down + up) for consistency check
        if new_stage in ("down", "up"):
            self._torso_angles.append(avg_hip)

        if new_stage == "down" and avg_hip < 50:
            issues.append("Too much forward lean")

        if rep_completed:

            worst_lean = min(self._torso_angles) if self._torso_angles else 90.0
            torso_std  = (float(np.std(self._torso_angles))
                          if len(self._torso_angles) >= 5 else 0.0)

            bar_drift = 0.0
            h_disp    = None
            if self.barbell_tracker and self.barbell_tracker.enabled:
                h_disp = self.barbell_tracker.horizontal_displacement()
                if h_disp is not None:
                    bar_drift = abs(h_disp)

            self.rep_metrics = {
                "min_knee":   self._min_knee,
                "worst_lean": worst_lean,
                "torso_std":  torso_std,
                "bar_drift":  bar_drift,
            }

            if self._min_knee <= self.KNEE_DOWN:
                issues.append("Good depth")
            else:
                issues.append("Not deep enough")
            self._min_knee = 180.0

            if torso_std > self.TORSO_STD_THRESHOLD:
                issues.append("Inconsistent torso angle")
            self._torso_angles.clear()

            if self.barbell_tracker and self.barbell_tracker.enabled:
                if h_disp is not None and bar_drift > self.BAR_DRIFT_THRESHOLD:
                    direction = "forward" if h_disp > 0 else "backward"
                    issues.append(f"Bar drifting {direction}")
                self.barbell_tracker.reset()

        return new_stage, rep_completed, issues

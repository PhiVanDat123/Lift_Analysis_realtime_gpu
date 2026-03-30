from __future__ import annotations

import math
import numpy as np
from core.exercises.base import BaseExercise

class SquatChecker(BaseExercise):

    KNEE_DOWN = 100
    KNEE_UP   = 160

    MIN_BAR_VERTICAL       = 0.03
    TORSO_STD_THRESHOLD    = 8
    BAR_DRIFT_THRESHOLD    = 0.05
    KNEE_COLLAPSE_RATIO    = 0.85
    LR_IMBALANCE_THRESHOLD = 15

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._torso_angles: list[float] = []
        self._min_knee:  float = 180.0
        self._worst_lean: float = 90.0

    def reset_state(self) -> None:
        super().reset_state()
        self._torso_angles.clear()
        self._min_knee   = 180.0
        self._worst_lean = 90.0

    @staticmethod
    def _torso_angle_vs_vertical(kp: dict) -> float:
        
        sx = (kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2
        sy = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
        hx = (kp["left_hip"][0]      + kp["right_hip"][0])      / 2
        hy = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
        dx = sx - hx
        dy = hy - sy
        return math.degrees(math.atan2(abs(dx), max(dy, 1e-6)))

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

        torso_angle = self._torso_angle_vs_vertical(kp)
        if new_stage in ("down", "up"):
            self._torso_angles.append(torso_angle)
        if torso_angle > 60 and new_stage == "down":
            issues.append("Torso: too much forward lean")

        if new_stage == "down":
            self._min_knee = min(self._min_knee, avg_knee)

        knee_w = abs(kp["left_knee"][0] - kp["right_knee"][0])
        hip_w  = abs(kp["left_hip"][0]  - kp["right_hip"][0])
        if hip_w > 0.01 and new_stage == "down":
            ratio = knee_w / hip_w
            if ratio < self.KNEE_COLLAPSE_RATIO:
                issues.append("Knees collapsing inward")

        lr_diff = abs(angles["left_knee"] - angles["right_knee"])
        if lr_diff > self.LR_IMBALANCE_THRESHOLD and new_stage == "down":
            issues.append(f"Left/right imbalance ({lr_diff:.0f}° diff)")

        if rep_completed:

            if self._min_knee <= self.KNEE_DOWN:
                issues.append("Squat depth: good (below parallel)")
            else:
                issues.append("Squat depth: not deep enough")
            self._min_knee = 180.0

            torso_std = (float(np.std(self._torso_angles))
                         if len(self._torso_angles) >= 5 else 0.0)
            worst_lean = max(self._torso_angles) if self._torso_angles else 0.0
            if torso_std > self.TORSO_STD_THRESHOLD:
                issues.append("Torso: inconsistent angle across rep")
            self._torso_angles.clear()

            bar_drift = 0.0
            if self.barbell_tracker and self.barbell_tracker.enabled:
                h_disp = self.barbell_tracker.horizontal_displacement()
                if h_disp is not None:
                    bar_drift = abs(h_disp)
                    if bar_drift > self.BAR_DRIFT_THRESHOLD:

                        nose_x = kp["nose"][0]
                        hip_x  = (kp["left_hip"][0] + kp["right_hip"][0]) / 2
                        forward_sign = 1 if nose_x > hip_x else -1
                        direction = "forward" if (h_disp * forward_sign) > 0 else "backward"
                        issues.append(f"Bar path: drifting {direction}")
                self.barbell_tracker.reset()

            self.rep_metrics = {
                "min_knee":   self._min_knee if self._min_knee < 180.0 else 100.0,
                "worst_lean": worst_lean,
                "torso_std":  torso_std,
                "bar_drift":  bar_drift,
            }

        return new_stage, rep_completed, issues

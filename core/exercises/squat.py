"""
Barbell Back Squat checker.

Detect  : hips, knees, shoulders, ankles + barbell position
Analyze :
  - Bar path (forward / backward drift)
  - Squat depth (below parallel)
  - Knees collapsing inward
  - Left / right imbalance
  - Torso angle consistency
"""
from __future__ import annotations

import numpy as np
from core.exercises.base import BaseExercise


class SquatChecker(BaseExercise):

    # Phase thresholds
    KNEE_DOWN = 100   # degrees — at or below = down
    KNEE_UP   = 160   # degrees — at or above = up

    MIN_BAR_VERTICAL = 0.03   # any detectable vertical movement qualifies

    # Rule thresholds
    IMBALANCE_THRESHOLD   = 15    # deg difference left vs right
    TORSO_STD_THRESHOLD   = 12    # deg std-dev of hip angle during down phase
    BAR_DRIFT_THRESHOLD   = 0.05  # normalised horizontal displacement

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._torso_angles: list[float] = []   # hip angles during down phase
        self._min_knee: float = 180.0           # lowest knee angle reached this rep

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

        # ── Track torso angle and depth during down phase ─────────────────────
        avg_hip = (angles["left_hip"] + angles["right_hip"]) / 2
        if new_stage == "down":
            self._torso_angles.append(avg_hip)
            self._min_knee = min(self._min_knee, avg_knee)

        # ── Continuous checks (accumulated, reported at rep end) ──────────────

        # 1. Knee cave
        left_cave  = kp["left_knee"][0]  > kp["left_ankle"][0]
        right_cave = kp["right_knee"][0] < kp["right_ankle"][0]
        if left_cave or right_cave:
            issues.append("⚠️ Knees caving in")

        # 2. Excessive forward lean
        if new_stage == "down" and avg_hip < 50:
            issues.append("⚠️ Too much forward lean")

        # 3. Left/right imbalance
        knee_diff = abs(angles["left_knee"] - angles["right_knee"])
        hip_diff  = abs(angles["left_hip"]  - angles["right_hip"])
        if knee_diff > self.IMBALANCE_THRESHOLD or hip_diff > self.IMBALANCE_THRESHOLD:
            issues.append("⚠️ Left/right imbalance")

        # ── Per-rep checks (on completion) ────────────────────────────────────
        if rep_completed:

            # 4. Squat depth — use lowest knee angle reached during down phase
            if self._min_knee <= self.KNEE_DOWN:
                issues.append("✅ Good depth")
            else:
                issues.append("⚠️ Not deep enough")
            self._min_knee = 180.0

            # 5. Torso angle consistency
            if len(self._torso_angles) >= 5:
                std = float(np.std(self._torso_angles))
                if std > self.TORSO_STD_THRESHOLD:
                    issues.append("⚠️ Inconsistent torso angle")
            self._torso_angles.clear()

            # 6. Barbell path direction
            if self.barbell_tracker and self.barbell_tracker.enabled:
                h_disp = self.barbell_tracker.horizontal_displacement()
                if h_disp is not None and abs(h_disp) > self.BAR_DRIFT_THRESHOLD:
                    issues.append("⚠️ Bar drifting forward")
                self.barbell_tracker.reset()

            if not any(i.startswith("⚠️") for i in issues):
                issues = ["✅ Good rep!"]

        return new_stage, rep_completed, issues

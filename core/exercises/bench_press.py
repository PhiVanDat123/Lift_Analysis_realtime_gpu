"""
Barbell Bench Press checker.

Detect  : shoulders, elbows, wrists + barbell movement
Analyze :
  - Bar path pattern (drift)
  - Elbow angle
  - Even lockout (one side higher?)
  - Consistent chest touchpoint (bar Y at bottom across reps)
  - Full range of motion (bar near chest at bottom, arms extended at top)
"""
from __future__ import annotations

import numpy as np
from core.exercises.base import BaseExercise


class BenchPressChecker(BaseExercise):

    # Phase thresholds — wide hysteresis to avoid false rep counts
    ELBOW_DOWN = 80    # degrees — bar clearly at chest
    ELBOW_UP   = 160   # degrees — arms clearly extended (lockout)

    # Rule thresholds
    ASYM_THRESHOLD       = 20    # deg — arm asymmetry during lift
    LOCKOUT_DIFF         = 15    # deg — uneven lockout
    ELBOW_FLARE_MAX      = 75    # deg — upper-arm/torso angle above this = flaring
    FULL_ROM_MIN_ELBOW   = 100   # deg — elbow must go below this at bottom
    FULL_ROM_MAX_ELBOW   = 140   # deg — elbow must reach above this at top
    TOUCHPOINT_STD       = 0.04  # normalised Y std-dev across reps
    BAR_DRIFT_THRESHOLD  = 0.04  # normalised horizontal displacement
    MIN_BAR_VERTICAL     = 0.08  # bar travels chest → lockout

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_elbow: float = 180.0
        self._max_elbow: float = 0.0
        self._bottom_y:  list[float] = []

    def check(self, angles, kp, stage):
        angles    = self.smooth(angles)
        avg_elbow = (angles["left_elbow"] + angles["right_elbow"]) / 2

        # ── Stage detection ───────────────────────────────────────────────────
        # Primary: barbell Y position (reliable for lying-down exercises)
        # Fallback: elbow angle (when barbell not detected)
        bt = self.barbell_tracker
        bar_dir = bt.bar_direction() if (bt and bt.enabled) else None

        if bar_dir == "down":
            new_stage = "down"
        elif bar_dir == "up":
            new_stage = "up"
        else:
            # Barbell stationary or not detected — use elbow angle
            new_stage = stage
            if avg_elbow <= self.ELBOW_DOWN:
                new_stage = "down"
            elif avg_elbow >= self.ELBOW_UP:
                new_stage = "up"

        rep_completed = (new_stage == "up" and stage == "down")

        if rep_completed and self.barbell_tracker and self.barbell_tracker.enabled:
            v_range = self.barbell_tracker.vertical_range()
            if v_range is not None and v_range < self.MIN_BAR_VERTICAL:
                rep_completed = False
        issues: list[str] = []

        # Track elbow ROM per rep
        self._min_elbow = min(self._min_elbow, avg_elbow)
        self._max_elbow = max(self._max_elbow, avg_elbow)

        # Track chest touchpoint (bar Y when at bottom)
        if new_stage == "down" and self.barbell_tracker and self.barbell_tracker._last_raw:
            self._bottom_y.append(self.barbell_tracker._last_raw[1])

        # ── Continuous checks ─────────────────────────────────────────────────

        # 1. Arm asymmetry during lift
        if abs(angles["left_elbow"] - angles["right_elbow"]) > self.ASYM_THRESHOLD:
            issues.append("⚠️ Uneven arms")

        # 2. Elbow flare — checked in down phase when bar is near chest
        if new_stage == "down":
            avg_shoulder = (angles["left_shoulder"] + angles["right_shoulder"]) / 2
            if avg_shoulder > self.ELBOW_FLARE_MAX:
                issues.append("⚠️ Elbows flaring too much")

        # 3. Excessive arch
        avg_hip = (angles["left_hip"] + angles["right_hip"]) / 2
        if avg_hip > 155:
            issues.append("⚠️ Excessive arch")

        # 4. Even lockout — checked when transitioning to "up"
        if new_stage == "up" and stage != "up":
            lockout_diff = abs(angles["left_elbow"] - angles["right_elbow"])
            if lockout_diff > self.LOCKOUT_DIFF:
                issues.append("⚠️ Uneven lockout")

        # ── Per-rep checks (on completion) ────────────────────────────────────
        if rep_completed:

            # 5. Full range of motion
            if self._min_elbow > self.FULL_ROM_MIN_ELBOW:
                issues.append("⚠️ Bar not low enough")
            if self._max_elbow < self.FULL_ROM_MAX_ELBOW:
                issues.append("⚠️ Incomplete lockout")

            # 6. Consistent chest touchpoint (needs ≥ 2 reps of data)
            if len(self._bottom_y) >= 2:
                touchpoint_std = float(np.std(self._bottom_y))
                if touchpoint_std > self.TOUCHPOINT_STD:
                    issues.append("⚠️ Inconsistent touch point")

            # 7. Barbell path drift
            if self.barbell_tracker and self.barbell_tracker.enabled:
                h_disp = self.barbell_tracker.horizontal_displacement()
                if h_disp is not None and abs(h_disp) > self.BAR_DRIFT_THRESHOLD:
                    direction = "toward head" if h_disp > 0 else "toward feet"
                    issues.append(f"⚠️ Bar drifting {direction}")
                self.barbell_tracker.reset()

            self._min_elbow = 180.0
            self._max_elbow = 0.0

            if not issues:
                issues = ["✅ Good rep!"]

        return new_stage, rep_completed, issues

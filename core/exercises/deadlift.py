"""
Barbell Deadlift (conventional) checker.

Detect  : hips, knees, shoulders + barbell position + back angle
Analyze :
  - Bar distance from shins (bar too far forward)
  - Hips rising too early (hips shoot up before shoulders)
  - Back rounding
  - Proper lockout (full hip & knee extension at top)
  - Mid-pull instability (bar path deviation during pull)
"""
from __future__ import annotations

from core.exercises.base import BaseExercise


class DeadliftChecker(BaseExercise):

    # Phase thresholds
    HIP_DOWN = 120   # degrees — hinged position
    HIP_UP   = 160   # degrees — lockout

    # Barbell must travel at least this far vertically (normalised) to confirm a real rep
    MIN_BAR_VERTICAL = 0.15

    # Rule thresholds
    LOCKOUT_HIP_MIN   = 165   # degrees — hip at lockout
    LOCKOUT_KNEE_MIN  = 160   # degrees — knee at lockout
    BAR_SHIN_DIST_MAX = 0.08  # normalised — bar x further than this from ankle = too far
    EARLY_HIP_MARGIN  = 0.05  # normalised Y — hips rise this much more than shoulders

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pull_start_hip_y:      float | None = None
        self._pull_start_shoulder_y: float | None = None
        self._early_hip_flagged:     bool = False

    def check(self, angles, kp, stage):
        angles  = self.smooth(angles)
        avg_hip = (angles["left_hip"] + angles["right_hip"]) / 2

        new_stage = stage
        if avg_hip <= self.HIP_DOWN:
            new_stage = "down"
        elif avg_hip >= self.HIP_UP:
            new_stage = "up"

        rep_completed = new_stage == "up" and stage == "down"

        # Gate: barbell must have travelled enough vertically to confirm a real lift
        if rep_completed and self.barbell_tracker and self.barbell_tracker.enabled:
            v_range = self.barbell_tracker.vertical_range()
            if v_range is not None and v_range < self.MIN_BAR_VERTICAL:
                rep_completed = False
        issues: list[str] = []

        # ── Snapshot at pull initiation ───────────────────────────────────────
        if new_stage == "down" and stage != "down":
            # Just entered down — capture baseline for hip-rise detection
            self._pull_start_hip_y      = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
            self._pull_start_shoulder_y = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
            self._early_hip_flagged     = False

        # ── Continuous checks ─────────────────────────────────────────────────

        # 1. Back rounding (active when in down phase)
        if new_stage == "down":
            avg_shoulder = (angles["left_shoulder"] + angles["right_shoulder"]) / 2
            if avg_shoulder < 30:
                issues.append("⚠️ Back rounding")

        # 2. Grip width (checked continuously)
        shoulder_w = abs(kp["left_shoulder"][0] - kp["right_shoulder"][0])
        wrist_w    = abs(kp["left_wrist"][0]    - kp["right_wrist"][0])
        if shoulder_w > 0:
            ratio = wrist_w / shoulder_w
            if ratio > 1.8:
                issues.append("⚠️ Grip too wide")
            elif ratio < 0.8:
                issues.append("⚠️ Grip too narrow")

        # 3. Bar distance from shins (when bar is near the ground)
        if new_stage == "down" and self.barbell_tracker and self.barbell_tracker.enabled:
            if self.barbell_tracker._last_raw is not None:
                bar_cx  = self.barbell_tracker._last_raw[0]
                ankle_x = (kp["left_ankle"][0] + kp["right_ankle"][0]) / 2
                if abs(bar_cx - ankle_x) > self.BAR_SHIN_DIST_MAX:
                    issues.append("⚠️ Bar too far from shins")

        # 4. Hips rising too early (mid-pull check)
        if (new_stage not in ("", "down")
                and self._pull_start_hip_y is not None
                and not self._early_hip_flagged):
            cur_hip_y      = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
            cur_shoulder_y = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
            hip_rise      = self._pull_start_hip_y      - cur_hip_y        # positive = rising
            shoulder_rise = self._pull_start_shoulder_y - cur_shoulder_y
            if hip_rise > shoulder_rise + self.EARLY_HIP_MARGIN:
                issues.append("⚠️ Hips rising too early")
                self._early_hip_flagged = True

        # 5. Lockout check (when transitioning to "up")
        if new_stage == "up" and stage != "up":
            avg_knee = (angles["left_knee"] + angles["right_knee"]) / 2
            if avg_hip < self.LOCKOUT_HIP_MIN or avg_knee < self.LOCKOUT_KNEE_MIN:
                issues.append("⚠️ Incomplete lockout")

        # ── Per-rep checks (on completion) ────────────────────────────────────
        if rep_completed:

            # 6. Mid-pull instability via barbell path deviation
            if self.barbell_tracker and self.barbell_tracker.enabled:
                if self.barbell_tracker.has_drift():
                    issues.append("⚠️ Bar drifting forward")
                self.barbell_tracker.reset()

            # Reset per-rep state
            self._pull_start_hip_y      = None
            self._pull_start_shoulder_y = None
            self._early_hip_flagged     = False

            if not issues:
                issues = ["✅ Good rep!"]

        return new_stage, rep_completed, issues

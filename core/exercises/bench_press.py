from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from core.exercises.base import BaseExercise


class BenchPressChecker(BaseExercise):

    ELBOW_DOWN           = 80    # degrees — bar near chest
    ELBOW_UP             = 140   # degrees — bar pushed up

    ASYM_THRESHOLD       = 20    # degrees — left/right elbow diff during lift
    LOCKOUT_DIFF         = 15    # degrees — left/right diff at lockout
    ELBOW_FLARE_MAX      = 75    # degrees — shoulder angle max before flagging flare
    FULL_ROM_MIN_ELBOW   = 100   # degrees — min elbow at bottom (must be < this)
    FULL_ROM_MAX_ELBOW   = 140   # degrees — min elbow at top (must be > this)
    TOUCHPOINT_STD       = 0.04  # normalised — std of bar_cy at chest across reps
    BAR_DRIFT_THRESHOLD  = 0.04  # normalised — horizontal displacement
    MIN_BAR_VERTICAL     = 0.08  # normalised — bar must travel this far vertically
    WRIST_BEND_THRESHOLD = 0.06  # normalised — horizontal wrist-elbow offset
    LOCKOUT_PEAK_TOL     = 0.02  # normalised — bar within this of peak = "at top"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_elbow:        float = 180.0
        self._max_elbow:        float = 0.0
        self._lockout_diff_val: float = 0.0
        self._worst_flare:      float = 0.0
        self._worst_back_angle: float = 0.0
        self._worst_wrist_bend: float = 0.0
        self._face_direction:   float = 0.0
        self._bottom_y:         list[float] = []   # bar_cy at chest per rep
        self._rep_min_bar_y:    float = 1.0
        # buffer during "up" phase: (bar_cx, bar_cy, avg_elbow, left_elbow, right_elbow)
        self._up_buffer: deque[tuple[float, float, float, float, float]] = deque(maxlen=120)

    def reset_state(self) -> None:
        super().reset_state()
        self._min_elbow        = 180.0
        self._max_elbow        = 0.0
        self._lockout_diff_val = 0.0
        self._worst_flare      = 0.0
        self._worst_back_angle = 0.0
        self._worst_wrist_bend = 0.0
        self._face_direction   = 0.0
        self._bottom_y.clear()
        self._rep_min_bar_y    = 1.0
        self._up_buffer.clear()

    def _eval_peak_from_buffer(self) -> tuple[float, float, float]:
        """
        Find bar peak (min cy) in up_buffer.
        Returns (avg_elbow_at_peak, left_elbow_at_peak, right_elbow_at_peak).
        Falls back to max elbow seen if no barbell data.
        """
        buf = list(self._up_buffer)
        valid = [(cx, cy, ae, le, re) for cx, cy, ae, le, re in buf if cy > 0.0]
        if valid:
            peak = min(valid, key=lambda v: v[1])
            return peak[2], peak[3], peak[4]
        # fallback: frame with max avg_elbow
        best = max(buf, key=lambda v: v[2])
        return best[2], best[3], best[4]

    def check(self, angles, kp, stage):
        angles    = self.smooth(angles)
        avg_elbow = (angles["left_elbow"] + angles["right_elbow"]) / 2

        bt      = self.barbell_tracker
        bar_dir = bt.bar_direction() if (bt and bt.enabled) else None

        # ── Stage detection ───────────────────────────────────────────────────
        if bar_dir == "down":
            new_stage = "down"
        elif bar_dir == "up":
            new_stage = "up"
        else:
            new_stage = stage
            if avg_elbow <= self.ELBOW_DOWN:
                new_stage = "down"
            elif avg_elbow >= self.ELBOW_UP:
                new_stage = "up"

        # ── Rep detection: completed when bar starts descending after "up" ────
        # i.e. stage was "up" and bar_dir just switched to "down"
        rep_completed = (stage == "up" and new_stage == "down")

        if rep_completed and bt and bt.enabled:
            v_range = bt.vertical_range()
            if v_range is not None and v_range < self.MIN_BAR_VERTICAL:
                rep_completed = False

        issues: list[str] = []

        # ── Continuous tracking ───────────────────────────────────────────────

        # Face direction (shoulder_x - hip_x)
        avg_sx = (kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2
        avg_hx = (kp["left_hip"][0]      + kp["right_hip"][0])      / 2
        self._face_direction = avg_sx - avg_hx

        # Track min elbow (ROM down) every frame
        self._min_elbow = min(self._min_elbow, avg_elbow)

        # Track bar_cy at bottom during "down" phase
        if new_stage == "down" and bt and bt._last_raw:
            self._rep_min_bar_y = min(self._rep_min_bar_y, bt._last_raw[1])

        # Track worst elbow flare during "down" phase
        avg_shoulder = (angles["left_shoulder"] + angles["right_shoulder"]) / 2
        if new_stage == "down":
            self._worst_flare = max(self._worst_flare, avg_shoulder)

        # Back angle: shoulder→hip vs horizontal
        sy = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
        hy = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
        dx = abs(avg_hx - avg_sx)
        dy = abs(hy - sy)
        back_angle = math.degrees(math.atan2(dy, dx)) if dx > 0.01 else 0.0
        self._worst_back_angle = max(self._worst_back_angle, back_angle)

        # Wrist bend
        left_wb  = abs(kp["left_wrist"][0]  - kp["left_elbow"][0])
        right_wb = abs(kp["right_wrist"][0] - kp["right_elbow"][0])
        max_wb   = max(left_wb, right_wb)
        self._worst_wrist_bend = max(self._worst_wrist_bend, max_wb)

        # ── Live issues (shown every frame) ───────────────────────────────────
        if abs(angles["left_elbow"] - angles["right_elbow"]) > self.ASYM_THRESHOLD:
            issues.append("Uneven arms")
        if new_stage == "down" and avg_shoulder > self.ELBOW_FLARE_MAX:
            issues.append("Elbow path: elbows flaring too much")
        if back_angle > 20:
            issues.append("Back angle: excessive arch")
        if max_wb > self.WRIST_BEND_THRESHOLD:
            issues.append("Wrists bending back")

        # ── Accumulate up_buffer ──────────────────────────────────────────────
        if new_stage == "up":
            bar_cx = bar_cy = 0.0
            if bt and bt.enabled and bt._last_raw is not None:
                bar_cx, bar_cy = bt._last_raw[0], bt._last_raw[1]
            self._up_buffer.append((bar_cx, bar_cy, avg_elbow,
                                    angles["left_elbow"], angles["right_elbow"]))

        # Clear buffer when going back down before rep completes (failed push)
        if new_stage == "down" and stage == "up" and not rep_completed:
            self._up_buffer.clear()

        # ── Per-rep checks (at bar peak) ──────────────────────────────────────
        if rep_completed:
            avg_at_peak, left_at_peak, right_at_peak = self._eval_peak_from_buffer()
            self._max_elbow        = avg_at_peak
            self._lockout_diff_val = abs(left_at_peak - right_at_peak)

            # Full ROM — bottom
            if self._min_elbow > self.FULL_ROM_MIN_ELBOW:
                issues.append("Full ROM: bar not low enough")

            # Full ROM — top / lockout
            if avg_at_peak < self.FULL_ROM_MAX_ELBOW:
                issues.append("Full ROM: incomplete lockout")

            # Even lockout
            if self._lockout_diff_val > self.LOCKOUT_DIFF:
                issues.append(f"Even lockout: one side higher ({self._lockout_diff_val:.0f}° diff)")

            # Consistent touchpoint
            if self._rep_min_bar_y < 1.0:
                self._bottom_y.append(self._rep_min_bar_y)
            self._rep_min_bar_y = 1.0
            if len(self._bottom_y) >= 2:
                touchpoint_std = float(np.std(self._bottom_y))
                if touchpoint_std > self.TOUCHPOINT_STD:
                    issues.append("Touchpoint: inconsistent chest contact")

            # Bar path pattern
            bar_drift = 0.0
            if bt and bt.enabled:
                h_disp = bt.horizontal_displacement()
                if h_disp is not None:
                    bar_drift = abs(h_disp)
                    if bar_drift > self.BAR_DRIFT_THRESHOLD:
                        toward_face = (h_disp * self._face_direction) > 0
                        direction = "toward face" if toward_face else "toward belly"
                        issues.append(f"Bar path: drifting {direction}")
                # wobble
                hist = bt._history
                if len(hist) >= 5:
                    wobble = float(np.std([p[0] for p in hist]))
                    if wobble > 0.025:
                        issues.append("Bar path: unstable (wobbling)")
                bt.reset()

            self.rep_metrics = {
                "min_elbow":        self._min_elbow,
                "max_elbow":        self._max_elbow,
                "lockout_diff":     self._lockout_diff_val,
                "worst_flare":      self._worst_flare,
                "worst_back_angle": self._worst_back_angle,
                "worst_wrist_bend": self._worst_wrist_bend,
                "bar_drift":        bar_drift,
            }

            # Reset per-rep state
            self._min_elbow        = 180.0
            self._max_elbow        = 0.0
            self._lockout_diff_val = 0.0
            self._worst_flare      = 0.0
            self._worst_back_angle = 0.0
            self._worst_wrist_bend = 0.0
            self._up_buffer.clear()

        return new_stage, rep_completed, issues

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from core.exercises.base import BaseExercise

class BenchPressChecker(BaseExercise):

    ELBOW_DOWN           = 80
    ELBOW_UP             = 140

    ASYM_THRESHOLD       = 20
    LOCKOUT_DIFF         = 15
    ELBOW_FLARE_MAX      = 75
    FULL_ROM_MIN_ELBOW   = 100
    FULL_ROM_MAX_ELBOW   = 140
    TOUCHPOINT_STD       = 0.04
    BAR_DRIFT_THRESHOLD  = 0.04
    MIN_BAR_VERTICAL     = 0.08
    WRIST_BEND_THRESHOLD = 0.06
    LOCKOUT_PEAK_TOL     = 0.02

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_elbow:        float = 180.0
        self._max_elbow:        float = 0.0
        self._lockout_diff_val: float = 0.0
        self._worst_flare:      float = 0.0
        self._worst_back_angle: float = 0.0
        self._worst_wrist_bend: float = 0.0
        self._face_direction:   float = 0.0
        self._bottom_y:         list[float] = []
        self._rep_min_bar_y:    float = 1.0
        self._in_push:          bool  = False

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
        self._in_push          = False
        self._up_buffer.clear()

    def _eval_peak_from_buffer(self) -> tuple[float, float, float]:
        
        buf = list(self._up_buffer)
        valid = [(cx, cy, ae, le, re) for cx, cy, ae, le, re in buf if cy > 0.0]
        if valid:
            peak = min(valid, key=lambda v: v[1])
            return peak[2], peak[3], peak[4]

        best = max(buf, key=lambda v: v[2])
        return best[2], best[3], best[4]

    def check(self, angles, kp, stage):
        angles    = self.smooth(angles)
        avg_elbow = (angles["left_elbow"] + angles["right_elbow"]) / 2

        bt      = self.barbell_tracker
        bar_dir = bt.bar_direction() if (bt and bt.enabled) else None

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

        if new_stage == "up" and stage == "down":
            self._in_push = True

        rep_completed = self._in_push and bar_dir == "down"

        if rep_completed and bt and bt.enabled:
            v_range = bt.vertical_range()
            if v_range is not None and v_range < self.MIN_BAR_VERTICAL:
                rep_completed = False

        issues: list[str] = []

        avg_sx = (kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2
        avg_hx = (kp["left_hip"][0]      + kp["right_hip"][0])      / 2
        self._face_direction = avg_sx - avg_hx

        self._min_elbow = min(self._min_elbow, avg_elbow)

        if new_stage == "down" and bt and bt._last_raw:
            self._rep_min_bar_y = min(self._rep_min_bar_y, bt._last_raw[1])

        avg_shoulder = (angles["left_shoulder"] + angles["right_shoulder"]) / 2
        if new_stage == "down":
            self._worst_flare = max(self._worst_flare, avg_shoulder)

        sy = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
        hy = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
        dx = abs(avg_hx - avg_sx)
        dy = abs(hy - sy)
        back_angle = math.degrees(math.atan2(dy, dx)) if dx > 0.01 else 0.0
        self._worst_back_angle = max(self._worst_back_angle, back_angle)

        left_wb  = abs(kp["left_wrist"][0]  - kp["left_elbow"][0])
        right_wb = abs(kp["right_wrist"][0] - kp["right_elbow"][0])
        max_wb   = max(left_wb, right_wb)
        self._worst_wrist_bend = max(self._worst_wrist_bend, max_wb)

        if abs(angles["left_elbow"] - angles["right_elbow"]) > self.ASYM_THRESHOLD:
            issues.append("Uneven arms")
        if new_stage == "down" and avg_shoulder > self.ELBOW_FLARE_MAX:
            issues.append("Elbow path: elbows flaring too much")
        if back_angle > 20:
            issues.append("Back angle: excessive arch")
        if max_wb > self.WRIST_BEND_THRESHOLD:
            issues.append("Wrists bending back")

        if new_stage == "up":
            bar_cx = bar_cy = 0.0
            if bt and bt.enabled and bt._last_raw is not None:
                bar_cx, bar_cy = bt._last_raw[0], bt._last_raw[1]
            self._up_buffer.append((bar_cx, bar_cy, avg_elbow,
                                    angles["left_elbow"], angles["right_elbow"]))

        if rep_completed:
            avg_at_peak, left_at_peak, right_at_peak = self._eval_peak_from_buffer()
            self._max_elbow        = avg_at_peak
            self._lockout_diff_val = abs(left_at_peak - right_at_peak)

            if self._min_elbow > self.FULL_ROM_MIN_ELBOW:
                issues.append("Full ROM: bar not low enough")

            if avg_at_peak < self.FULL_ROM_MAX_ELBOW:
                issues.append("Full ROM: incomplete lockout")

            if self._lockout_diff_val > self.LOCKOUT_DIFF:
                issues.append(f"Even lockout: one side higher ({self._lockout_diff_val:.0f}° diff)")

            if self._rep_min_bar_y < 1.0:
                self._bottom_y.append(self._rep_min_bar_y)
            self._rep_min_bar_y = 1.0
            if len(self._bottom_y) >= 2:
                touchpoint_std = float(np.std(self._bottom_y))
                if touchpoint_std > self.TOUCHPOINT_STD:
                    issues.append("Touchpoint: inconsistent chest contact")

            bar_drift = 0.0
            if bt and bt.enabled:
                h_disp = bt.horizontal_displacement()
                if h_disp is not None:
                    bar_drift = abs(h_disp)
                    if bar_drift > self.BAR_DRIFT_THRESHOLD:
                        toward_face = (h_disp * self._face_direction) > 0
                        direction = "toward face" if toward_face else "toward belly"
                        issues.append(f"Bar path: drifting {direction}")

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

            self._min_elbow        = 180.0
            self._max_elbow        = 0.0
            self._lockout_diff_val = 0.0
            self._worst_flare      = 0.0
            self._worst_back_angle = 0.0
            self._worst_wrist_bend = 0.0
            self._in_push          = False
            self._up_buffer.clear()

        return new_stage, rep_completed, issues

    def finalize(self) -> tuple[bool, list[str]]:
        if not self._in_push or not self._up_buffer:
            return False, []
        bt = self.barbell_tracker
        issues: list[str] = []
        avg_at_peak, left_at_peak, right_at_peak = self._eval_peak_from_buffer()
        self._max_elbow        = avg_at_peak
        self._lockout_diff_val = abs(left_at_peak - right_at_peak)

        if self._min_elbow > self.FULL_ROM_MIN_ELBOW:
            issues.append("Full ROM: bar not low enough")
        if avg_at_peak < self.FULL_ROM_MAX_ELBOW:
            issues.append("Full ROM: incomplete lockout")
        if self._lockout_diff_val > self.LOCKOUT_DIFF:
            issues.append(f"Even lockout: one side higher ({self._lockout_diff_val:.0f}° diff)")

        if self._rep_min_bar_y < 1.0:
            self._bottom_y.append(self._rep_min_bar_y)
        if len(self._bottom_y) >= 2:
            if float(np.std(self._bottom_y)) > self.TOUCHPOINT_STD:
                issues.append("Touchpoint: inconsistent chest contact")

        bar_drift = 0.0
        if bt and bt.enabled:
            h_disp = bt.horizontal_displacement()
            if h_disp is not None:
                bar_drift = abs(h_disp)
                if bar_drift > self.BAR_DRIFT_THRESHOLD:
                    toward_face = (h_disp * self._face_direction) > 0
                    direction = "toward face" if toward_face else "toward belly"
                    issues.append(f"Bar path: drifting {direction}")
            hist = bt._history
            if len(hist) >= 5:
                if float(np.std([p[0] for p in hist])) > 0.025:
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
        self._in_push = False
        self._up_buffer.clear()
        return True, issues

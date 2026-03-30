from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from core.exercises.base import BaseExercise

class DeadliftChecker(BaseExercise):

    HIP_DOWN = 120
    HIP_UP   = 160

    MIN_BAR_VERTICAL = 0.15

    LOCKOUT_HIP_MIN     = 165
    LOCKOUT_KNEE_MIN    = 165
    LOCKOUT_TORSO_MAX   = 0.06
    LOCKOUT_HOLD_FRAMES = 3
    LOCKOUT_PEAK_TOL    = 0.02
    BAR_SHIN_DIST_MAX   = 0.08
    BAR_DRIFT_THRESHOLD = 0.04
    BAR_WOBBLE_THRESHOLD = 0.025
    EARLY_HIP_MARGIN    = 0.05
    EARLY_PHASE_HIP_MIN = 120
    EARLY_PHASE_HIP_MAX = 145

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._early_phase_hip_y:      float | None = None
        self._early_phase_shoulder_y: float | None = None
        self._in_early_phase:         bool = False
        self._early_hip_flagged:      bool = False

        self._up_buffer: deque[tuple[float, float, float, float, float, float]] = deque(maxlen=120)

        self._lockout_hip:        float = 0.0
        self._lockout_knee:       float = 0.0
        self._lockout_torso_ok:   bool  = False
        self._lockout_hold_ok:    bool  = False
        self._max_bar_shin_dist:  float = 0.0
        self._max_hip_rise_delta: float = 0.0

    def reset_state(self) -> None:
        super().reset_state()
        self._early_phase_hip_y      = None
        self._early_phase_shoulder_y = None
        self._in_early_phase         = False
        self._early_hip_flagged      = False
        self._up_buffer.clear()
        self._lockout_hip            = 0.0
        self._lockout_knee           = 0.0
        self._lockout_torso_ok       = False
        self._lockout_hold_ok        = False
        self._max_bar_shin_dist      = 0.0
        self._max_hip_rise_delta     = 0.0

    @staticmethod
    def _dist_point_to_segment(px: float, py: float,
                                ax: float, ay: float,
                                bx: float, by: float) -> float:
        
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _eval_lockout_from_buffer(self) -> tuple[float, float, bool, bool]:
        
        if not self._up_buffer:
            return 0.0, 0.0, False, False

        buf = list(self._up_buffer)

        valid = [(cx, cy, h, k, sx, hx) for cx, cy, h, k, sx, hx in buf if cy > 0.0]

        if valid:
            peak_cy    = min(v[1] for v in valid)
            peak_frame = min(valid, key=lambda v: v[1])
            hip_at_peak, knee_at_peak = peak_frame[2], peak_frame[3]
            sx_peak, hx_peak = peak_frame[4], peak_frame[5]
            torso_ok = abs(sx_peak - hx_peak) <= self.LOCKOUT_TORSO_MAX

            near_peak = [
                (h >= self.LOCKOUT_HIP_MIN
                 and k >= self.LOCKOUT_KNEE_MIN
                 and abs(sx - hx) <= self.LOCKOUT_TORSO_MAX)
                for cx, cy, h, k, sx, hx in valid
                if cy <= peak_cy + self.LOCKOUT_PEAK_TOL
            ]
            max_hold = best = 0
            for ok in near_peak:
                best = best + 1 if ok else 0
                max_hold = max(max_hold, best)

            hold_ok = max_hold >= self.LOCKOUT_HOLD_FRAMES
        else:

            hip_at_peak  = max(h for _, _, h, _, _, _ in buf)
            knee_at_peak = max(k for _, _, _, k, _, _ in buf)
            torso_ok     = False
            hold_ok      = False

        return hip_at_peak, knee_at_peak, torso_ok, hold_ok

    def _eval_bar_stability(self) -> list[str]:
        
        buf = list(self._up_buffer)
        xs = [cx for cx, cy, _, _, _, _ in buf if cy > 0.0]

        if len(xs) < 5:
            return []

        issues: list[str] = []

        net_dx = xs[-1] - xs[0]
        if abs(net_dx) > self.BAR_DRIFT_THRESHOLD:
            direction = "forward" if net_dx > 0 else "backward"
            issues.append(f"Bar drifting {direction}")

        wobble = float(np.std(xs))
        if wobble > self.BAR_WOBBLE_THRESHOLD:
            issues.append("Bar path unstable")

        return issues

    def compute_form_score(self) -> int:
        
        m = self.rep_metrics
        if not m:
            return 0

        score = 100.0

        l_hip = m.get("lockout_hip", 0.0)
        if l_hip > 0:
            score -= max(0.0, min(25.0, (self.LOCKOUT_HIP_MIN  - l_hip)  / 20 * 25))
        else:
            score -= 25.0

        l_knee = m.get("lockout_knee", 0.0)
        if l_knee > 0:
            score -= max(0.0, min(25.0, (self.LOCKOUT_KNEE_MIN - l_knee) / 20 * 25))
        else:
            score -= 25.0

        bar_shin = m.get("max_bar_shin", 0.0)
        if bar_shin > self.BAR_SHIN_DIST_MAX:
            score -= min(25.0, (bar_shin - self.BAR_SHIN_DIST_MAX) / 0.10 * 25)

        hip_delta = m.get("hip_rise_delta", 0.0)
        if hip_delta > self.EARLY_HIP_MARGIN:
            score -= min(15.0, (hip_delta - self.EARLY_HIP_MARGIN) / 0.15 * 15)

        if m.get("bar_drift", False):
            score -= 10.0

        return max(0, round(score))

    def check(self, angles, kp, stage):
        angles  = self.smooth(angles)
        avg_hip = (angles["left_hip"] + angles["right_hip"]) / 2

        new_stage = stage
        if avg_hip <= self.HIP_DOWN:
            new_stage = "down"
        elif avg_hip >= self.HIP_UP:
            new_stage = "up"

        rep_completed = new_stage == "up" and stage == "down"

        if rep_completed and self.barbell_tracker and self.barbell_tracker.enabled:
            v_range = self.barbell_tracker.vertical_range()
            if v_range is not None and v_range < self.MIN_BAR_VERTICAL:
                rep_completed = False

        issues: list[str] = []

        in_window = self.EARLY_PHASE_HIP_MIN <= avg_hip <= self.EARLY_PHASE_HIP_MAX

        if in_window and not self._in_early_phase:
            self._early_phase_hip_y      = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
            self._early_phase_shoulder_y = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
            self._in_early_phase         = True
            self._early_hip_flagged      = False

        if not in_window and self._in_early_phase and avg_hip > self.EARLY_PHASE_HIP_MAX:
            self._in_early_phase = False

        shoulder_w = abs(kp["left_shoulder"][0] - kp["right_shoulder"][0])
        wrist_w    = abs(kp["left_wrist"][0]    - kp["right_wrist"][0])
        if shoulder_w > 0:
            ratio = wrist_w / shoulder_w
            if ratio > 1.8:
                issues.append("Grip too wide")
            elif ratio < 0.8:
                issues.append("Grip too narrow")

        if new_stage in ("down", "up") and self.barbell_tracker and self.barbell_tracker.enabled:
            if self.barbell_tracker._last_raw is not None:
                bar_cx = self.barbell_tracker._last_raw[0]
                bar_cy = self.barbell_tracker._last_raw[1]
                l_dist = self._dist_point_to_segment(
                    bar_cx, bar_cy,
                    kp["left_ankle"][0],  kp["left_ankle"][1],
                    kp["left_knee"][0],   kp["left_knee"][1],
                )
                r_dist = self._dist_point_to_segment(
                    bar_cx, bar_cy,
                    kp["right_ankle"][0], kp["right_ankle"][1],
                    kp["right_knee"][0],  kp["right_knee"][1],
                )
                shin_dist = (l_dist + r_dist) / 2
                self._max_bar_shin_dist = max(self._max_bar_shin_dist, shin_dist)
                if shin_dist > self.BAR_SHIN_DIST_MAX:
                    issues.append("Bar too far from shins")

        if self._in_early_phase and self._early_phase_hip_y is not None:
            cur_hip_y      = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
            cur_shoulder_y = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
            hip_rise      = self._early_phase_hip_y      - cur_hip_y
            shoulder_rise = self._early_phase_shoulder_y - cur_shoulder_y
            delta = hip_rise - shoulder_rise
            self._max_hip_rise_delta = max(self._max_hip_rise_delta, delta)
            if delta > self.EARLY_HIP_MARGIN and not self._early_hip_flagged:
                issues.append("Hips rising too early")
                self._early_hip_flagged = True

        if new_stage == "up":
            avg_knee      = (angles["left_knee"]      + angles["right_knee"])      / 2
            avg_shoulder_x = (kp["left_shoulder"][0]  + kp["right_shoulder"][0])   / 2
            avg_hip_x      = (kp["left_hip"][0]       + kp["right_hip"][0])        / 2
            bar_cx = bar_cy = 0.0
            if self.barbell_tracker and self.barbell_tracker.enabled:
                if self.barbell_tracker._last_raw is not None:
                    bar_cx = self.barbell_tracker._last_raw[0]
                    bar_cy = self.barbell_tracker._last_raw[1]
            self._up_buffer.append((bar_cx, bar_cy, avg_hip, avg_knee, avg_shoulder_x, avg_hip_x))

        if new_stage == "down" and stage == "up":
            self._up_buffer.clear()

        if rep_completed:

            hip_at_peak, knee_at_peak, torso_ok, hold_ok = self._eval_lockout_from_buffer()
            self._lockout_hip      = hip_at_peak
            self._lockout_knee     = knee_at_peak
            self._lockout_torso_ok = torso_ok
            self._lockout_hold_ok  = hold_ok

            if knee_at_peak < self.LOCKOUT_KNEE_MIN:
                issues.append("Incomplete lockout — knees not fully extended")
            if hip_at_peak < self.LOCKOUT_HIP_MIN:
                issues.append("Incomplete lockout — hips not fully extended")
            if not torso_ok:
                issues.append("Torso not upright at lockout")
            if knee_at_peak >= self.LOCKOUT_KNEE_MIN and hip_at_peak >= self.LOCKOUT_HIP_MIN and torso_ok and not hold_ok:
                issues.append("Lockout not held")

            stability_issues = self._eval_bar_stability()
            issues.extend(stability_issues)
            bar_drift   = any("drifting" in s for s in stability_issues)
            bar_wobble  = any("unstable" in s for s in stability_issues)

            if self.barbell_tracker and self.barbell_tracker.enabled:
                self.barbell_tracker.reset()

            self.rep_metrics = {
                "lockout_hip":      self._lockout_hip,
                "lockout_knee":     self._lockout_knee,
                "lockout_torso_ok": self._lockout_torso_ok,
                "lockout_hold_ok":  self._lockout_hold_ok,
                "max_bar_shin":     self._max_bar_shin_dist,
                "hip_rise_delta":   self._max_hip_rise_delta,
                "bar_drift":        bar_drift,
                "bar_wobble":       bar_wobble,
            }

            self._early_phase_hip_y      = None
            self._early_phase_shoulder_y = None
            self._in_early_phase         = False
            self._early_hip_flagged      = False
            self._up_buffer.clear()
            self._lockout_hip            = 0.0
            self._lockout_knee           = 0.0
            self._lockout_torso_ok       = False
            self._lockout_hold_ok        = False
            self._max_bar_shin_dist      = 0.0
            self._max_hip_rise_delta     = 0.0

            if len(issues) == 1:
                issues.insert(0, "Good rep!")

        return new_stage, rep_completed, issues

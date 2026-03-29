from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from core.exercises.base import BaseExercise


class DeadliftChecker(BaseExercise):

    # Phase thresholds
    HIP_DOWN = 120   # degrees — hinged position
    HIP_UP   = 160   # degrees — lockout

    # Barbell must travel at least this far vertically (normalised) to confirm a real rep
    MIN_BAR_VERTICAL = 0.15

    # Rule thresholds
    LOCKOUT_HIP_MIN     = 170   # degrees — full hip extension at lockout
    LOCKOUT_KNEE_MIN    = 170   # degrees — full knee extension at lockout
    LOCKOUT_HOLD_FRAMES = 3     # minimum consecutive frames near bar peak to confirm hold
    LOCKOUT_PEAK_TOL    = 0.02  # normalised Y — bar within this of peak = "at top"
    BAR_SHIN_DIST_MAX   = 0.08  # normalised — Euclidean dist from bar to shin segment
    BAR_DRIFT_THRESHOLD = 0.04  # normalised X — net horizontal displacement = drift
    BAR_WOBBLE_THRESHOLD = 0.025 # normalised — std(x) during pull = instability
    EARLY_HIP_MARGIN    = 0.05  # normalised Y — hips rise this much more than shoulders
    EARLY_PHASE_HIP_MIN = 120   # degrees — start of early-pull window (= HIP_DOWN)
    EARLY_PHASE_HIP_MAX = 145   # degrees — end of early-pull window

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._early_phase_hip_y:      float | None = None
        self._early_phase_shoulder_y: float | None = None
        self._in_early_phase:         bool = False
        self._early_hip_flagged:      bool = False
        # lockout + stability tracking — buffer of (bar_cx, bar_cy, avg_hip, avg_knee)
        # bar_cx/bar_cy = 0.0 when barbell not detected that frame
        self._up_buffer: deque[tuple[float, float, float, float]] = deque(maxlen=120)
        # severity tracking — reset each rep
        self._lockout_hip:        float = 0.0
        self._lockout_knee:       float = 0.0
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
        self._lockout_hold_ok        = False
        self._max_bar_shin_dist      = 0.0
        self._max_hip_rise_delta     = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dist_point_to_segment(px: float, py: float,
                                ax: float, ay: float,
                                bx: float, by: float) -> float:
        """
        Euclidean distance from point P to the nearest point on segment A→B.
        Projects P onto AB, clamps t∈[0,1], returns straight-line distance.
        """
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _eval_lockout_from_buffer(self) -> tuple[float, float, bool]:
        """
        Analyse _up_buffer to find the frame where bar is at its peak (min cy),
        read hip/knee angles there, and count how many consecutive frames near
        peak satisfy the lockout angle requirements.

        Returns (hip_at_peak, knee_at_peak, hold_ok).
        Falls back to best angles seen if barbell data is missing.
        """
        if not self._up_buffer:
            return 0.0, 0.0, False

        buf = list(self._up_buffer)

        # Find peak bar position (min cy = highest on screen)
        # bar_cy = 0.0 means no detection that frame — skip
        valid = [(cx, cy, h, k) for cx, cy, h, k in buf if cy > 0.0]

        if valid:
            peak_cy = min(v[1] for v in valid)
            peak_frame = min(valid, key=lambda v: v[1])
            hip_at_peak, knee_at_peak = peak_frame[2], peak_frame[3]

            # Count consecutive frames within LOCKOUT_PEAK_TOL of peak
            # where both angles meet lockout thresholds
            near_peak = [
                (h >= self.LOCKOUT_HIP_MIN and k >= self.LOCKOUT_KNEE_MIN)
                for cx, cy, h, k in valid
                if cy <= peak_cy + self.LOCKOUT_PEAK_TOL
            ]
            max_hold = best = 0
            for ok in near_peak:
                best = best + 1 if ok else 0
                max_hold = max(max_hold, best)

            hold_ok = max_hold >= self.LOCKOUT_HOLD_FRAMES
        else:
            # No barbell data — fall back to best angles in buffer
            hip_at_peak  = max(h for _, _, h, _ in buf)
            knee_at_peak = max(k for _, _, _, k in buf)
            hold_ok = False

        return hip_at_peak, knee_at_peak, hold_ok

    def _eval_bar_stability(self) -> list[str]:
        """
        Analyse horizontal bar path during the pull phase (_up_buffer).

        Two separate issues detected:
          Drift   — net horizontal displacement from pull start to peak.
                    Consistent one-directional movement = bar swinging away.
          Wobble  — std(x) of bar path during pull.
                    High variance = bar path unstable / not vertical.

        Returns list of issue strings (empty = no instability).
        """
        buf = list(self._up_buffer)
        xs = [cx for cx, cy, _, _ in buf if cy > 0.0]   # only detected frames

        if len(xs) < 5:
            return []   # not enough data

        issues: list[str] = []

        # Drift: net displacement from first to last detected position
        net_dx = xs[-1] - xs[0]
        if abs(net_dx) > self.BAR_DRIFT_THRESHOLD:
            direction = "forward" if net_dx > 0 else "backward"
            issues.append(f"Bar drifting {direction}")

        # Wobble: std of x — high = bar path is not straight vertical
        wobble = float(np.std(xs))
        if wobble > self.BAR_WOBBLE_THRESHOLD:
            issues.append("Bar path unstable")

        return issues

    def compute_form_score(self) -> int:
        """
        Convert rep_metrics into a 0-100 form score.

        Penalty breakdown (total 100 pts):
          Lockout hip      25 pts  — lockout_hip
          Lockout knee     25 pts  — lockout_knee
          Bar-shin dist    25 pts  — max_bar_shin
          Early hip rise   15 pts  — hip_rise_delta
          Bar drift        10 pts  — bar_drift (bool)
        """
        m = self.rep_metrics
        if not m:
            return 0

        score = 100.0

        # 1. Lockout hip (25 pts) — 170°+ = no penalty, 150°- = max
        l_hip = m.get("lockout_hip", 0.0)
        if l_hip > 0:
            score -= max(0.0, min(25.0, (self.LOCKOUT_HIP_MIN  - l_hip)  / 20 * 25))
        else:
            score -= 25.0

        # 2. Lockout knee (25 pts) — 170°+ = no penalty, 150°- = max
        l_knee = m.get("lockout_knee", 0.0)
        if l_knee > 0:
            score -= max(0.0, min(25.0, (self.LOCKOUT_KNEE_MIN - l_knee) / 20 * 25))
        else:
            score -= 25.0

        # 3. Bar too far from shins (25 pts) — ≤0.08 = no penalty, ≥0.18 = max
        bar_shin = m.get("max_bar_shin", 0.0)
        if bar_shin > self.BAR_SHIN_DIST_MAX:
            score -= min(25.0, (bar_shin - self.BAR_SHIN_DIST_MAX) / 0.10 * 25)

        # 4. Early hip rise (15 pts) — ≤0.05 = no penalty, ≥0.20 = max
        hip_delta = m.get("hip_rise_delta", 0.0)
        if hip_delta > self.EARLY_HIP_MARGIN:
            score -= min(15.0, (hip_delta - self.EARLY_HIP_MARGIN) / 0.15 * 15)

        # 5. Bar drift (10 pts)
        if m.get("bar_drift", False):
            score -= 10.0

        return max(0, round(score))

    # ── Main check ────────────────────────────────────────────────────────────

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

        # ── Early-phase window (120°→145°) ────────────────────────────────────
        in_window = self.EARLY_PHASE_HIP_MIN <= avg_hip <= self.EARLY_PHASE_HIP_MAX

        if in_window and not self._in_early_phase:
            self._early_phase_hip_y      = (kp["left_hip"][1]      + kp["right_hip"][1])      / 2
            self._early_phase_shoulder_y = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
            self._in_early_phase         = True
            self._early_hip_flagged      = False

        if not in_window and self._in_early_phase and avg_hip > self.EARLY_PHASE_HIP_MAX:
            self._in_early_phase = False

        # ── Continuous checks ─────────────────────────────────────────────────

        # 1. Grip width
        shoulder_w = abs(kp["left_shoulder"][0] - kp["right_shoulder"][0])
        wrist_w    = abs(kp["left_wrist"][0]    - kp["right_wrist"][0])
        if shoulder_w > 0:
            ratio = wrist_w / shoulder_w
            if ratio > 1.8:
                issues.append("Grip too wide")
            elif ratio < 0.8:
                issues.append("Grip too narrow")

        # 2. Bar distance from shins — Euclidean dist to ankle→knee segment
        #    Checked throughout full lift (bar must stay close at every height)
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

        # 3. Hips rising too early — only inside early-pull window (120°→145°)
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

        # 4. Lockout + stability — accumulate buffer during "up" phase
        #    Each frame: (bar_cx, bar_cy, avg_hip, avg_knee)
        #    bar_cx/bar_cy = 0.0 sentinel when barbell tracker unavailable
        if new_stage == "up":
            avg_knee = (angles["left_knee"] + angles["right_knee"]) / 2
            bar_cx = bar_cy = 0.0
            if self.barbell_tracker and self.barbell_tracker.enabled:
                if self.barbell_tracker._last_raw is not None:
                    bar_cx = self.barbell_tracker._last_raw[0]
                    bar_cy = self.barbell_tracker._last_raw[1]
            self._up_buffer.append((bar_cx, bar_cy, avg_hip, avg_knee))

        # Clear buffer when dropping back to "down" (failed rep / lowering)
        if new_stage == "down" and stage == "up":
            self._up_buffer.clear()

        # ── Per-rep checks (on completion) ────────────────────────────────────
        if rep_completed:

            # Evaluate lockout at bar peak
            hip_at_peak, knee_at_peak, hold_ok = self._eval_lockout_from_buffer()
            self._lockout_hip   = hip_at_peak
            self._lockout_knee  = knee_at_peak
            self._lockout_hold_ok = hold_ok

            if hip_at_peak < self.LOCKOUT_HIP_MIN or knee_at_peak < self.LOCKOUT_KNEE_MIN:
                issues.append("Incomplete lockout")
            elif not hold_ok:
                issues.append("Lockout not held")

            # Bar path stability — drift and wobble from buffer
            stability_issues = self._eval_bar_stability()
            issues.extend(stability_issues)
            bar_drift   = any("drifting" in s for s in stability_issues)
            bar_wobble  = any("unstable" in s for s in stability_issues)

            if self.barbell_tracker and self.barbell_tracker.enabled:
                self.barbell_tracker.reset()

            self.rep_metrics = {
                "lockout_hip":      self._lockout_hip,
                "lockout_knee":     self._lockout_knee,
                "lockout_hold_ok":  self._lockout_hold_ok,
                "max_bar_shin":     self._max_bar_shin_dist,
                "hip_rise_delta":   self._max_hip_rise_delta,
                "bar_drift":        bar_drift,
                "bar_wobble":       bar_wobble,
            }

            form_score = self.compute_form_score()
            self.rep_metrics["form_score"] = form_score
            issues.append(f"Form score: {form_score}/100")

            # Reset per-rep state
            self._early_phase_hip_y      = None
            self._early_phase_shoulder_y = None
            self._in_early_phase         = False
            self._early_hip_flagged      = False
            self._up_buffer.clear()
            self._lockout_hip            = 0.0
            self._lockout_knee           = 0.0
            self._lockout_hold_ok        = False
            self._max_bar_shin_dist      = 0.0
            self._max_hip_rise_delta     = 0.0
            self._up_buffer.clear()

            if len(issues) == 1:  # only the score line → perfect rep
                issues.insert(0, "Good rep!")

        return new_stage, rep_completed, issues

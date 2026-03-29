from __future__ import annotations

import numpy as np
from core.exercises.base import BaseExercise


class BenchPressChecker(BaseExercise):

    ELBOW_DOWN = 80    
    ELBOW_UP   = 160   

    ASYM_THRESHOLD       = 20    
    LOCKOUT_DIFF         = 15    
    ELBOW_FLARE_MAX      = 75    
    FULL_ROM_MIN_ELBOW   = 100  
    FULL_ROM_MAX_ELBOW   = 140   
    TOUCHPOINT_STD       = 0.04  
    BAR_DRIFT_THRESHOLD  = 0.04  
    MIN_BAR_VERTICAL     = 0.08  
    WRIST_BEND_THRESHOLD = 0.06  

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_elbow: float = 180.0
        self._max_elbow: float = 0.0
        self._bottom_y:      list[float] = []  
        self._rep_min_bar_y: float = 1.0       
        self._worst_flare:      float = 0.0
        self._worst_arch:       float = 0.0
        self._lockout_diff_val: float = 0.0
        self._worst_wrist_bend: float = 0.0

    def reset_state(self) -> None:
        super().reset_state()
        self._min_elbow        = 180.0
        self._max_elbow        = 0.0
        self._bottom_y.clear()
        self._rep_min_bar_y    = 1.0
        self._worst_flare      = 0.0
        self._worst_arch       = 0.0
        self._lockout_diff_val = 0.0
        self._worst_wrist_bend = 0.0

    def check(self, angles, kp, stage):
        angles    = self.smooth(angles)
        avg_elbow = (angles["left_elbow"] + angles["right_elbow"]) / 2

        bt = self.barbell_tracker
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

        rep_completed = (new_stage == "up" and stage == "down")

        if rep_completed and self.barbell_tracker and self.barbell_tracker.enabled:
            v_range = self.barbell_tracker.vertical_range()
            if v_range is not None and v_range < self.MIN_BAR_VERTICAL:
                rep_completed = False
        issues: list[str] = []

        self._min_elbow = min(self._min_elbow, avg_elbow)
        if avg_elbow > self._max_elbow:
            self._max_elbow = avg_elbow
            self._lockout_diff_val = abs(angles["left_elbow"] - angles["right_elbow"])

        if new_stage == "down" and self.barbell_tracker and self.barbell_tracker._last_raw:
            self._rep_min_bar_y = min(self._rep_min_bar_y, self.barbell_tracker._last_raw[1])

        asym = abs(angles["left_elbow"] - angles["right_elbow"])
        if asym > self.ASYM_THRESHOLD:
            issues.append("Uneven arms")

        avg_shoulder = (angles["left_shoulder"] + angles["right_shoulder"]) / 2
        if new_stage == "down":
            self._worst_flare = max(self._worst_flare, avg_shoulder)
            if avg_shoulder > self.ELBOW_FLARE_MAX:
                issues.append("Elbows flaring too much")

        avg_hip = (angles["left_hip"] + angles["right_hip"]) / 2
        self._worst_arch = max(self._worst_arch, avg_hip)
        if avg_hip > 155:
            issues.append("Excessive arch")

        left_wrist_bend  = abs(kp["left_wrist"][0]  - kp["left_elbow"][0])
        right_wrist_bend = abs(kp["right_wrist"][0] - kp["right_elbow"][0])
        max_wrist_bend   = max(left_wrist_bend, right_wrist_bend)
        self._worst_wrist_bend = max(self._worst_wrist_bend, max_wrist_bend)
        if max_wrist_bend > self.WRIST_BEND_THRESHOLD:
            issues.append("Wrists bending back")

        if rep_completed:

            if self._min_elbow > self.FULL_ROM_MIN_ELBOW:
                issues.append("Bar not low enough")
            if self._max_elbow < self.FULL_ROM_MAX_ELBOW:
                issues.append("Incomplete lockout")
            if self._lockout_diff_val > self.LOCKOUT_DIFF:
                issues.append("Uneven lockout")

            if self._rep_min_bar_y < 1.0:
                self._bottom_y.append(self._rep_min_bar_y)
            self._rep_min_bar_y = 1.0

            touchpoint_std = 0.0
            if len(self._bottom_y) >= 2:
                touchpoint_std = float(np.std(self._bottom_y))
                if touchpoint_std > self.TOUCHPOINT_STD:
                    issues.append("Inconsistent touch point")

            bar_drift = 0.0
            if self.barbell_tracker and self.barbell_tracker.enabled:
                h_disp = self.barbell_tracker.horizontal_displacement()
                if h_disp is not None:
                    bar_drift = max(0.0, h_disp)
                    if bar_drift > self.BAR_DRIFT_THRESHOLD:
                        issues.append("Bar drifting toward face")
                self.barbell_tracker.reset()

            self.rep_metrics = {
                "min_elbow":        self._min_elbow,
                "max_elbow":        self._max_elbow,
                "lockout_diff":     self._lockout_diff_val,
                "worst_flare":      self._worst_flare,
                "worst_arch":       self._worst_arch,
                "worst_wrist_bend": self._worst_wrist_bend,
                "bar_drift":        bar_drift,
            }

            self._min_elbow        = 180.0
            self._max_elbow        = 0.0
            self._worst_flare      = 0.0
            self._worst_arch       = 0.0
            self._lockout_diff_val = 0.0
            self._worst_wrist_bend = 0.0

        return new_stage, rep_completed, issues

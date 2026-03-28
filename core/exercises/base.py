"""
Abstract base class for all exercise checkers.

Each subclass implements `check()` which is called once per frame
and returns the updated stage, whether a rep just completed,
and a list of feedback strings (empty = no issues detected yet).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.barbell import BarbellTracker


class BaseExercise(ABC):

    # EMA smoothing factor: 0 = không đổi, 1 = không smooth
    # 0.25 ở 20fps ≈ time constant 200ms — lọc noise nhưng vẫn responsive
    _EMA_ALPHA = 0.5

    def __init__(self, barbell_tracker: Optional["BarbellTracker"] = None):
        self.barbell_tracker    = barbell_tracker
        self._smoothed: dict[str, float] = {}

    def smooth(self, angles: dict[str, float]) -> dict[str, float]:
        """EMA smoothing trên angle values — lọc nhiễu pose detection."""
        out = {}
        for k, v in angles.items():
            prev = self._smoothed.get(k, v)
            out[k] = self._EMA_ALPHA * v + (1 - self._EMA_ALPHA) * prev
        self._smoothed = out
        return out

    @abstractmethod
    def check(
        self,
        angles: dict[str, float],
        kp: dict[str, list[float]],
        stage: str,
    ) -> tuple[str, bool, list[str]]:
        """
        Parameters
        ----------
        angles : dict of joint-name → angle in degrees
        kp     : dict of joint-name → [x, y] normalized [0,1]
        stage  : current phase ("up" | "down" | "")

        Returns
        -------
        new_stage     : updated phase string
        rep_completed : True when a full up→down→up cycle just finished
        issues        : list of feedback strings (⚠️ warning or ✅ perfect)
        """
        ...

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.barbell import BarbellTracker


class BaseExercise(ABC):

    _EMA_ALPHA = 0.5

    def __init__(self, barbell_tracker: Optional["BarbellTracker"] = None):
        self.barbell_tracker    = barbell_tracker
        self._smoothed: dict[str, float] = {}
        self.rep_metrics: dict = {}   

    def smooth(self, angles: dict[str, float]) -> dict[str, float]:
        out = {}
        for k, v in angles.items():
            prev = self._smoothed.get(k, v)
            out[k] = self._EMA_ALPHA * v + (1 - self._EMA_ALPHA) * prev
        self._smoothed = out
        return out

    def reset_state(self) -> None:
        self._smoothed.clear()

    @abstractmethod
    def check(
        self,
        angles: dict[str, float],
        kp: dict[str, list[float]],
        stage: str,
    ) -> tuple[str, bool, list[str]]:
        ...

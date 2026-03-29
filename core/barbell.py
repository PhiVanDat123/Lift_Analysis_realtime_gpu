from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()  

MODEL_ID   = "barbell-zwl3l-ambrq/1"
MODEL_PATH = Path(__file__).parent.parent / "models" / "barbell.pt"

CONF_THRESHOLD  = 0.20
DRIFT_THRESHOLD = 0.03   

_rf_model   = None
_yolo_model = None


def _load_rf_model():
    global _rf_model
    if _rf_model is None:
        from inference import get_model  
        _rf_model = get_model(
            model_id=MODEL_ID,
            api_key=os.environ["ROBOFLOW_API_KEY"],
        )
    return _rf_model


def _load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO  
        _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model


def _pick_backend() -> str:
    if os.environ.get("ROBOFLOW_API_KEY"):
        return "roboflow"
    if MODEL_PATH.exists():
        return "local"
    return "none"

class BarbellTracker:

    def __init__(self, history_len: int = 60):
        self._history: deque[tuple[float, float]] = deque(maxlen=history_len)
        self._backend: str = _pick_backend()
        self._last_raw: Optional[tuple] = None   # cached last successful detection

    @property
    def enabled(self) -> bool:
        return self._backend != "none"

    @property
    def backend(self) -> str:
        return self._backend


    def detect(self, frame_rgb: np.ndarray) -> Optional[tuple[float, float, float, float]]:
        if not self.enabled:
            return None
        h, w = frame_rgb.shape[:2]
        result = (
            self._detect_rf(frame_rgb, h, w)
            if self._backend == "roboflow"
            else self._detect_local(frame_rgb, h, w)
        )
        if result is not None:
            self._last_raw = result
            self._history.append((result[0], result[1]))
        return result

    def _detect_rf(self, frame_rgb, h, w):
        try:
            model   = _load_rf_model()
            results = model.infer(frame_rgb)[0]

            best, best_conf = None, 0.0
            for pred in results.predictions:
                conf = float(pred.confidence)
                if conf > best_conf:
                    best_conf = conf
                    cx = float(pred.x) / w
                    cy = float(pred.y) / h
                    bw = float(pred.width)  / w
                    bh = float(pred.height) / h
                    best = (cx, cy, bw, bh)

            if best and best_conf >= CONF_THRESHOLD:
                return best
        except Exception as e:
            print(f"[barbell] RF detect error: {e}")
        return None

    def _detect_local(self, frame_rgb, h, w):
        try:
            model   = _load_yolo_model()
            results = model(frame_rgb, verbose=False)[0]

            best, best_conf = None, 0.0
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    best = (cx, cy, bw, bh)

            if best and best_conf >= CONF_THRESHOLD:
                return best
        except Exception as e:
            print(f"[barbell] local detect error: {e}")
        return None

    def path_deviation(self) -> Optional[float]:
        if len(self._history) < 5:
            return None
        return float(np.std([p[0] for p in self._history]))

    def has_drift(self) -> bool:
        dev = self.path_deviation()
        return dev is not None and dev > DRIFT_THRESHOLD

    def horizontal_displacement(self) -> Optional[float]:
        if len(self._history) < 3:
            return None
        return float(self._history[-1][0] - self._history[0][0])

    def vertical_range(self) -> Optional[float]:
        if len(self._history) < 5:
            return None
        ys = [p[1] for p in self._history]
        return float(max(ys) - min(ys))

    def bar_direction(self, window: int = 5) -> Optional[str]:
        if len(self._history) < window:
            return None
        ys  = [p[1] for p in list(self._history)[-window:]]
        dy  = ys[-1] - ys[0]
        if dy >  0.015:   
            return "down"
        if dy < -0.015:   
            return "up"
        return None       

    def reset(self) -> None:
        self._history.clear()
        self._last_raw = None


    def draw(self, frame_bgr: np.ndarray, detection: Optional[tuple]) -> None:
        h, w = frame_bgr.shape[:2]

        box = detection if detection is not None else self._last_raw
        if box is not None:
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if len(self._history) >= 2:
            pts = [(int(p[0] * w), int(p[1] * h)) for p in self._history]
            for i in range(1, len(pts)):
                cv2.line(frame_bgr, pts[i - 1], pts[i], (0, 165, 255), 4)
            cv2.circle(frame_bgr, pts[-1], 8, (0, 0, 255), -1)

barbell_tracker = BarbellTracker()


def warmup() -> None:
    if not barbell_tracker.enabled:
        print("[barbell] No backend available — tracking disabled.")
        print("  → Set ROBOFLOW_API_KEY  (inference SDK)")
        print("  → Or place weights at   models/barbell.pt")
        return
    print(f"[barbell] Loading model via backend='{barbell_tracker.backend}' ...")
    try:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        barbell_tracker.detect(dummy)
        print("[barbell] Model ready.")
    except Exception as e:
        print(f"[barbell] Warmup failed: {e}")

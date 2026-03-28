"""
Frame & video processing orchestrator.

Keeps all OpenCV drawing logic in one place, separate from:
  - Pose math   (core/pose.py)
  - Barbell tracking (core/barbell.py)
  - Exercise rules   (core/exercises/)
  - Scoring          (core/scoring.py)
"""
from __future__ import annotations

import queue
import tempfile
import threading
import time
from typing import Optional

import cv2
import imageio
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from core.barbell import barbell_tracker, warmup as _barbell_warmup

# Pre-load barbell model at import time (runs once when app starts)
_barbell_warmup()
from core.exercises import EXERCISE_CHECKERS
from core.pose import (
    compute_angles,
    extract_keypoints,
    mp_drawing,
    mp_drawing_styles,
    mp_pose,
    pose_detector,
)
from core.scoring import compute_score, score_color_bgr, score_label

# ── HUD / overlay drawing helpers ────────────────────────────────────────────


def _draw_hud(
    frame_bgr: np.ndarray,
    stage: str,
    counter: int,
    last_score: Optional[int],
) -> None:
    box_h = 115 if last_score is not None else 90
    cv2.rectangle(frame_bgr, (0, 0), (340, box_h), (0, 0, 0), -1)
    cv2.putText(frame_bgr, f"Stage : {stage or '-'}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"Reps  : {counter}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if last_score is not None:
        color = score_color_bgr(last_score)
        cv2.putText(
            frame_bgr,
            f"Score : {last_score}/100  {score_label(last_score)}",
            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )


def _draw_angles(frame_bgr: np.ndarray, angles: dict) -> None:
    pass  # removed — joint coordinates no longer shown on screen


# MediaPipe landmark indices for each exercise
_EXERCISE_LANDMARKS: dict[str, list[int]] = {
    "Back Squat":  [11, 12, 23, 24, 25, 26, 27, 28],  # shoulders, hips, knees, ankles
    "Deadlift":    [11, 12, 23, 24, 25, 26],           # shoulders, hips, knees
    "Bench Press": [11, 12, 13, 14, 15, 16],           # shoulders, elbows, wrists
}

_EXERCISE_CONNECTIONS: dict[str, list[tuple[int, int]]] = {
    "Back Squat":  [(11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)],
    "Deadlift":    [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26)],
    "Bench Press": [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16)],
}


def _draw_exercise_landmarks(
    frame_bgr: np.ndarray,
    landmarks,
    exercise: str,
) -> None:
    """Draw only the joints and bones relevant to the current exercise."""
    h, w = frame_bgr.shape[:2]
    lm_list = landmarks.landmark

    indices    = _EXERCISE_LANDMARKS.get(exercise, [])
    conns      = _EXERCISE_CONNECTIONS.get(exercise, [])

    # Bones
    for i, j in conns:
        a = lm_list[i]
        b = lm_list[j]
        if a.visibility > 0.5 and b.visibility > 0.5:
            cv2.line(frame_bgr,
                     (int(a.x * w), int(a.y * h)),
                     (int(b.x * w), int(b.y * h)),
                     (255, 255, 255), 2)

    # Joints
    for idx in indices:
        lm = lm_list[idx]
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 6, (0, 200, 255), -1)   # cyan fill
            cv2.circle(frame_bgr, (cx, cy), 6, (255, 140, 0),  2)   # orange ring


def _draw_arc(
    frame_bgr: np.ndarray,
    vertex: tuple[int, int],
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 28,
) -> None:
    """Draw an angle arc at *vertex* between rays toward p1 and p2."""
    v1 = np.array(p1, dtype=float) - np.array(vertex, dtype=float)
    v2 = np.array(p2, dtype=float) - np.array(vertex, dtype=float)
    # arctan2 with inverted y (screen coords)
    a1 = np.degrees(np.arctan2(-v1[1], v1[0]))
    a2 = np.degrees(np.arctan2(-v2[1], v2[0]))
    start, end = sorted([a1, a2])
    if end - start > 180:
        start, end = end, start + 360
    cv2.ellipse(frame_bgr, vertex, (radius, radius), 0,
                int(start), int(end), color, 2)


def _draw_angle_markers(
    frame_bgr: np.ndarray,
    kp: dict,
    angles: dict,
    exercise: str,
) -> None:
    """
    Draw joint angle arcs + labels on key joints.
    Spec: Bench Press only gets angle markers; Squat & Deadlift get bar path only.
    """
    if exercise != "Bench Press":
        return

    h, w = frame_bgr.shape[:2]

    # (point_a, vertex, point_b, angle_key, BGR colour)
    configs = [
        ("left_shoulder",  "left_elbow",   "left_wrist",    "left_elbow",    (0, 255, 255)),
        ("right_shoulder", "right_elbow",  "right_wrist",   "right_elbow",   (0, 200, 255)),
        ("left_elbow",     "left_shoulder","left_hip",      "left_shoulder",  (255, 200, 0)),
        ("right_elbow",    "right_shoulder","right_hip",    "right_shoulder", (200, 200, 0)),
    ]

    for pa_key, vx_key, pb_key, ang_key, color in configs:
        pa = (int(kp[pa_key][0] * w), int(kp[pa_key][1] * h))
        vx = (int(kp[vx_key][0] * w), int(kp[vx_key][1] * h))
        pb = (int(kp[pb_key][0] * w), int(kp[pb_key][1] * h))

        # Joint circle only (no bone lines)
        cv2.circle(frame_bgr, vx, 6, color, -1)
        # Arc
        _draw_arc(frame_bgr, vx, pa, pb, color)
        # Label
        cv2.putText(frame_bgr, f"{angles[ang_key]:.0f}°",
                    (vx[0] + 10, vx[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _clean(line: str) -> str:
    """Strip emoji/unicode chars that OpenCV cannot render."""
    return (line.replace("✅", "").replace("⚠️", "").replace("⚠", "")
                .replace("\u2014", "-").replace("\u2013", "-")  # em/en dash
                .strip())


_FONT_PATHS = [
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _pil_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _truncate_pil(text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw) -> str:
    """Truncate *text* with '…' so it fits within *max_w* pixels (PIL)."""
    try:
        tw = int(draw.textlength(text, font=font))
    except AttributeError:
        tw = font.getsize(text)[0]
    if tw <= max_w:
        return text
    while text:
        text = text[:-1]
        s = text + "..."
        try:
            tw = int(draw.textlength(s, font=font))
        except AttributeError:
            tw = font.getsize(s)[0]
        if tw <= max_w:
            return s
    return "..."


def _draw_feedback_overlay(frame_bgr: np.ndarray, lines: list[str]) -> None:
    """Compact feedback box at the bottom of the frame — PIL TrueType for sharp text."""
    if not lines:
        return
    fh, fw = frame_bgr.shape[:2]

    # Font size: ~1.8% of frame height, capped so box ≤ 28% of frame
    font_size = max(11, min(int(fh * 0.018), int(fh * 0.28 / len(lines))))
    pad       = 6
    margin    = 6
    line_gap  = 3

    pil_font = _pil_font(font_size)

    # Measure on a scratch draw object (no colour conversion yet)
    _tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_w = fw - margin * 2 - pad * 2

    clean_lines = [_clean(l) for l in lines]
    rendered: list[tuple[str, bool]] = []
    for raw, clean in zip(lines, clean_lines):
        rendered.append((_truncate_pil(clean, pil_font, max_text_w, _tmp), raw.startswith("✅")))

    try:
        line_h = int(_tmp.textbbox((0, 0), "Ag", font=pil_font)[3]) + line_gap
    except AttributeError:
        line_h = pil_font.getsize("Ag")[1] + line_gap

    box_h = pad * 2 + line_h * len(rendered)
    box_w = fw - margin * 2
    x1    = margin
    y1    = fh - box_h - margin

    # Draw solid white box with border using cv2 (no alpha blending = no blur)
    cv2.rectangle(frame_bgr, (x1, y1), (x1 + box_w, y1 + box_h), (255, 255, 255), -1)
    cv2.rectangle(frame_bgr, (x1, y1), (x1 + box_w, y1 + box_h), (80, 80, 80), 1)

    # Render text with PIL directly onto the (already white) region
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)
    draw      = ImageDraw.Draw(pil_img)

    for i, (text, is_good) in enumerate(rendered):
        color = (0, 140, 0) if is_good else (20, 20, 20)
        draw.text((x1 + pad, y1 + pad + line_h * i), text, font=pil_font, fill=color)

    frame_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Background worker: server-side capture + inference ────────────────────────


class _StreamWorker:
    """
    Three threads:
      _capture_loop  — camera at native fps (~30fps)
      _pose_loop     — MediaPipe only (~20-30fps, no longer blocked by YOLO)
      _barbell_loop  — YOLO only (~5fps, runs independently)

    get_display_frame() composites all three at 30fps.
    """

    _INFER_DIM   = 480
    _DISPLAY_DIM = 640

    def __init__(self, device: int = 0) -> None:
        self._lock = threading.Lock()

        self._raw_frame:   Optional[np.ndarray] = None
        self._pose_lm                            = None
        self._barbell_det: Optional[tuple]       = None
        self._exercise:    str                   = "Back Squat"
        self._counter:     int                   = 0
        self._stage:       str                   = ""
        self._feedback:    str                   = ""
        self._last_score:  Optional[int]         = None

        self._pose_q:    queue.Queue = queue.Queue(maxsize=1)
        self._barbell_q: queue.Queue = queue.Queue(maxsize=1)
        self._cap = cv2.VideoCapture(device)

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._pose_loop,    daemon=True).start()
        threading.Thread(target=self._barbell_loop, daemon=True).start()

    def set_exercise(self, exercise: str) -> None:
        with self._lock:
            self._exercise = exercise

    def get_state(self) -> dict:
        with self._lock:
            return {
                "counter":    self._counter,
                "stage":      self._stage,
                "feedback":   self._feedback,
                "last_score": self._last_score,
            }

    def get_display_frame(self) -> Optional[np.ndarray]:
        """Compose latest raw frame + latest inference overlay."""
        with self._lock:
            raw        = self._raw_frame
            landmarks  = self._pose_lm
            barbell    = self._barbell_det
            exercise   = self._exercise
            counter    = self._counter
            stage      = self._stage
            last_score = self._last_score

        if raw is None:
            return None

        h, w  = raw.shape[:2]
        scale = min(1.0, self._DISPLAY_DIM / max(h, w))
        disp  = cv2.resize(raw, (int(w * scale), int(h * scale))) if scale < 1.0 else raw.copy()

        annotated  = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
        dh, dw     = annotated.shape[:2]

        if barbell is not None:
            cx, cy, bw_, bh_ = barbell
            x1 = int((cx - bw_ / 2) * dw);  y1 = int((cy - bh_ / 2) * dh)
            x2 = int((cx + bw_ / 2) * dw);  y2 = int((cy + bh_ / 2) * dh)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if landmarks:
            _draw_exercise_landmarks(annotated, landmarks, exercise)
            kp     = extract_keypoints(landmarks)
            angles = compute_angles(kp)
            _draw_angle_markers(annotated, kp, angles, exercise)

        _draw_hud(annotated, stage, counter, last_score)

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._raw_frame

    def reset(self) -> dict:
        with self._lock:
            self._counter    = 0
            self._stage      = ""
            self._feedback   = ""
            self._last_score = None
        barbell_tracker.reset()
        return {"counter": 0, "stage": "", "feedback": "", "last_score": None}

    def _capture_loop(self) -> None:
        while True:
            ret, frame_bgr = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame_rgb = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)
            with self._lock:
                self._raw_frame = frame_rgb
            for q in (self._pose_q, self._barbell_q):
                try:
                    q.put_nowait(frame_rgb)
                except queue.Full:
                    pass

    def _pose_loop(self) -> None:
        """MediaPipe only — runs at ~20-30fps, not blocked by YOLO."""
        while True:
            frame = self._pose_q.get()

            with self._lock:
                exercise   = self._exercise
                counter    = self._counter
                stage      = self._stage
                feedback   = self._feedback
                last_score = self._last_score

            h, w  = frame.shape[:2]
            scale = min(1.0, self._INFER_DIM / max(h, w))
            infer = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

            results = pose_detector.process(infer)

            if results.pose_landmarks:
                kp      = extract_keypoints(results.pose_landmarks)
                angles  = compute_angles(kp)
                checker = EXERCISE_CHECKERS[exercise]
                new_stage, rep_done, issues = checker.check(angles, kp, stage)
                stage = new_stage

                if rep_done:
                    counter   += 1
                    last_score = compute_score(issues)
                    feedback   = (
                        f"Score: {last_score}/100 ({score_label(last_score)})\n"
                        + "\n".join(issues)
                    )

            with self._lock:
                self._pose_lm    = results.pose_landmarks
                self._counter    = counter
                self._stage      = stage
                self._feedback   = feedback
                self._last_score = last_score

    def _barbell_loop(self) -> None:
        """YOLO only — runs independently, slow (~5fps) without blocking pose."""
        if not barbell_tracker.enabled:
            return
        while True:
            frame = self._barbell_q.get()

            h, w  = frame.shape[:2]
            scale = min(1.0, self._INFER_DIM / max(h, w))
            infer = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

            detection = barbell_tracker.detect(infer)
            with self._lock:
                self._barbell_det = detection


_stream_worker = _StreamWorker()


# ── Public functions ──────────────────────────────────────────────────────────


def get_latest_frame() -> Optional[np.ndarray]:
    return _stream_worker.get_display_frame()


def get_raw_frame() -> Optional[np.ndarray]:
    return _stream_worker.get_raw_frame()


def set_exercise(exercise: str) -> None:
    _stream_worker.set_exercise(exercise)


def get_current_state() -> tuple[str, dict]:
    state = _stream_worker.get_state()
    return state["feedback"], state


def reset_counter(state: dict) -> dict:
    return _stream_worker.reset()


def process_video(video_path: Optional[str], exercise: str) -> tuple[Optional[str], str]:
    """
    Process an uploaded video file frame-by-frame.

    Performance knobs
    -----------------
    BARBELL_EVERY : run YOLO every N frames, reuse last detection in between
    POSE_EVERY    : run MediaPipe every N frames, reuse last pose in between
    YOLO_MAX_DIM  : resize frame to this max dimension before YOLO inference

    Returns
    -------
    output_path : path to the annotated MP4 (or None on error)
    summary     : multi-line text summary for the Gradio textbox
    """
    if video_path is None:
        return None, "Chưa có video."

    # ── Performance knobs ────────────────────────────────────────────────────
    BARBELL_EVERY = 1    # YOLO once every 5 frames  (~6× speedup for detection)
    POSE_EVERY    = 1    # MediaPipe once every 2 frames (~2× speedup for pose)
    YOLO_MAX_DIM  = 640  # resize before YOLO (normalised coords stay correct)

    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = imageio.get_writer(
        tmp.name, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-crf", "15"],
    )

    # ── State ────────────────────────────────────────────────────────────────
    counter    = 0
    stage      = ""
    feedback   = ""
    last_score: Optional[int] = None

    rep_feedbacks:       list[str] = []
    rep_scores:          list[int] = []
    overlay_lines:       list[str] = []
    overlay_frames_left: int       = 0
    prev_counter:        int       = 0
    OVERLAY_DURATION               = int(fps * 3)

    # Cached heavy-model results
    last_detection:    Optional[tuple]  = None
    last_pose_results                   = None
    last_kp:           Optional[dict]   = None
    last_angles:       Optional[dict]   = None

    barbell_tracker.reset()
    frame_idx = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated = frame_bgr.copy()

        # ── Barbell detection (every BARBELL_EVERY frames) ───────────────────
        if barbell_tracker.enabled:
            if frame_idx % BARBELL_EVERY == 0:
                h, w   = frame_rgb.shape[:2]
                scale  = min(1.0, YOLO_MAX_DIM / max(h, w))
                small  = cv2.resize(frame_rgb, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_rgb
                last_detection = barbell_tracker.detect(small)
            barbell_tracker.draw(annotated, last_detection)

        # ── MediaPipe pose (every POSE_EVERY frames) ─────────────────────────
        if frame_idx % POSE_EVERY == 0:
            last_pose_results = pose_detector.process(frame_rgb)
            if last_pose_results.pose_landmarks:
                last_kp     = extract_keypoints(last_pose_results.pose_landmarks)
                last_angles = compute_angles(last_kp)
            else:
                last_kp     = None
                last_angles = None

        if last_pose_results and last_pose_results.pose_landmarks:
            _draw_exercise_landmarks(annotated, last_pose_results.pose_landmarks, exercise)

        if last_kp and last_angles:
            checker = EXERCISE_CHECKERS[exercise]
            new_stage, rep_done, issues = checker.check(last_angles, last_kp, stage)
            stage = new_stage

            if rep_done:
                counter   += 1
                last_score = compute_score(issues)
                feedback   = (
                    f"Score: {last_score}/100 ({score_label(last_score)})\n"
                    + "\n".join(issues)
                )

            _draw_angles(annotated, last_angles)
            _draw_angle_markers(annotated, last_kp, last_angles, exercise)  # bench press only

        # ── Rep overlay ──────────────────────────────────────────────────────
        if counter > prev_counter:
            score = last_score or 0
            rep_scores.append(score)
            rep_label = f"Rep {counter} - {score}/100 ({score_label(score)}):"
            rep_feedbacks.append(f"{rep_label}\n{feedback}")
            issue_lines = [l for l in (feedback.split("\n") if feedback else [])
                           if not l.startswith("Score:")]
            overlay_lines       = [rep_label] + issue_lines
            overlay_frames_left = OVERLAY_DURATION
            prev_counter        = counter

        if overlay_frames_left > 0:
            _draw_feedback_overlay(annotated, overlay_lines)
            overlay_frames_left -= 1

        writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        frame_idx += 1

        if total > 0 and frame_idx % 60 == 0:
            print(f"[video] {frame_idx}/{total} frames ({frame_idx/total*100:.0f}%)")

    cap.release()
    writer.close()

    avg     = int(sum(rep_scores) / len(rep_scores)) if rep_scores else 0
    summary = (
        f"Tổng số rep   : {counter}\n"
        f"Điểm TB       : {avg}/100 ({score_label(avg)})\n\n"
        + "\n---\n".join(rep_feedbacks)
    )
    return tmp.name, summary

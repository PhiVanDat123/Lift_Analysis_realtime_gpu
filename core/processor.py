from __future__ import annotations

import queue as _queue
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass
from typing import Optional

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch as _torch
_GPU_AVAILABLE = _torch.cuda.is_available()

from core.barbell import barbell_tracker, warmup as _barbell_warmup

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
from core.scoring import compute_score, score_color_bgr

_stream_state: dict = {
    "queue":  _queue.Queue(maxsize=30),
    "active": threading.Event(),
    "token":  0,
}

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

MJPEG_PORT: int = _free_port()

class _MJPEGHandler(BaseHTTPRequestHandler):

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                active = _stream_state["active"].is_set()
                try:
                    frame_jpg = _stream_state["queue"].get(timeout=0.1)
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_jpg
                        + b"\r\n"
                    )
                    self.wfile.flush()
                except _queue.Empty:
                    if not active and _stream_state["queue"].empty():
                        break
        except (BrokenPipeError, ConnectionResetError):
            pass

def _start_mjpeg_server() -> None:
    server = HTTPServer(("127.0.0.1", MJPEG_PORT), _MJPEGHandler)
    server.serve_forever()

threading.Thread(target=_start_mjpeg_server, daemon=True).start()

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
            f"Score : {last_score}/100 ",
            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

_EXERCISE_LANDMARKS: dict[str, list[int]] = {
    "Back Squat":  [11, 12, 23, 24, 25, 26, 27, 28],
    "Deadlift":    [11, 12, 23, 24, 25, 26],
    "Bench Press": [11, 12, 13, 14, 15, 16],
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
    h, w = frame_bgr.shape[:2]
    lm_list = landmarks.landmark
    for i, j in _EXERCISE_CONNECTIONS.get(exercise, []):
        a, b = lm_list[i], lm_list[j]
        if a.visibility > 0.5 and b.visibility > 0.5:
            cv2.line(frame_bgr,
                     (int(a.x * w), int(a.y * h)),
                     (int(b.x * w), int(b.y * h)),
                     (255, 255, 255), 2)
    for idx in _EXERCISE_LANDMARKS.get(exercise, []):
        lm = lm_list[idx]
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 6, (0, 200, 255), -1)
            cv2.circle(frame_bgr, (cx, cy), 6, (255, 140, 0),  2)

def _draw_arc(
    frame_bgr: np.ndarray,
    vertex: tuple[int, int],
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 28,
) -> None:
    v1 = np.array(p1, dtype=float) - np.array(vertex, dtype=float)
    v2 = np.array(p2, dtype=float) - np.array(vertex, dtype=float)
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
    import math as _math
    if exercise != "Bench Press":
        return
    h, w = frame_bgr.shape[:2]

    configs = [
        ("left_shoulder",  "left_elbow",  "left_wrist",  "left_elbow",  (0, 255, 255)),
        ("right_shoulder", "right_elbow", "right_wrist", "right_elbow", (0, 200, 255)),
    ]
    for pa_key, vx_key, pb_key, ang_key, color in configs:
        pa = (int(kp[pa_key][0] * w), int(kp[pa_key][1] * h))
        vx = (int(kp[vx_key][0] * w), int(kp[vx_key][1] * h))
        pb = (int(kp[pb_key][0] * w), int(kp[pb_key][1] * h))
        cv2.circle(frame_bgr, vx, 6, color, -1)
        _draw_arc(frame_bgr, vx, pa, pb, color)
        cv2.putText(frame_bgr, f"{angles[ang_key]:.0f}deg",
                    (vx[0] + 10, vx[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    sx = int((kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2 * w)
    sy = int((kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2 * h)
    hx = int((kp["left_hip"][0]      + kp["right_hip"][0])      / 2 * w)
    hy = int((kp["left_hip"][1]      + kp["right_hip"][1])      / 2 * h)
    dx = abs(hx - sx)
    dy = abs(hy - sy)
    back_angle = _math.degrees(_math.atan2(dy, dx)) if dx > 2 else 0.0
    color_back = (0, 80, 255) if back_angle > 20 else (0, 200, 80)
    cv2.line(frame_bgr, (sx, sy), (hx, hy), color_back, 2)
    cv2.line(frame_bgr, (sx, sy), (sx + dx, sy), (160, 160, 160), 1)
    mid_x, mid_y = (sx + hx) // 2, (sy + hy) // 2
    cv2.putText(frame_bgr, f"Back:{back_angle:.0f}deg",
                (mid_x + 6, mid_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_back, 2)

def _clean(line: str) -> str:
    return (line.replace("✅", "").replace("⚠️", "").replace("⚠", "")
                .replace("\u2014", "-").replace("\u2013", "-")
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

def _truncate_pil(text: str, font, max_w: int, draw) -> str:
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
    if not lines:
        return
    fh, fw = frame_bgr.shape[:2]
    font_size = max(11, min(int(fh * 0.018), int(fh * 0.28 / len(lines))))
    pad, margin, line_gap = 6, 6, 3
    pil_font = _pil_font(font_size)
    _tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_w = fw - margin * 2 - pad * 2
    clean_lines = [_clean(l) for l in lines]
    rendered = [(_truncate_pil(c, pil_font, max_text_w, _tmp), r.startswith("✅"))
                for r, c in zip(lines, clean_lines)]
    try:
        line_h = int(_tmp.textbbox((0, 0), "Ag", font=pil_font)[3]) + line_gap
    except AttributeError:
        line_h = pil_font.getsize("Ag")[1] + line_gap
    box_h = pad * 2 + line_h * len(rendered)
    x1, y1 = margin, fh - box_h - margin
    cv2.rectangle(frame_bgr, (x1, y1), (x1 + fw - margin * 2, y1 + box_h), (255, 255, 255), -1)
    cv2.rectangle(frame_bgr, (x1, y1), (x1 + fw - margin * 2, y1 + box_h), (80, 80, 80), 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    for i, (text, is_good) in enumerate(rendered):
        color = (0, 140, 0) if is_good else (20, 20, 20)
        draw.text((x1 + pad, y1 + pad + line_h * i), text, font=pil_font, fill=color)
    frame_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _build_overlay_cache(frame_shape: tuple, lines: list[str]) -> np.ndarray:
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    _draw_feedback_overlay(canvas, lines)
    return canvas

def _blit_overlay(dst: np.ndarray, overlay: np.ndarray) -> None:
    mask = np.any(overlay != 0, axis=2)
    dst[mask] = overlay[mask]

def process_video_realtime(video_path: Optional[str], exercise: str):
    if video_path is None:
        return

    checker = EXERCISE_CHECKERS[exercise]
    checker.reset_state()
    barbell_tracker.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps              = cap.get(cv2.CAP_PROP_FPS) or 30
    INFER_MAX_DIM    = 640 if _GPU_AVAILABLE else 480
    OVERLAY_DURATION = int(fps * 3)

    DISPLAY_FPS    = 12.0
    FRAME_INTERVAL = 1.0 / DISPLAY_FPS

    SKIP_FRAMES    = max(1, round(fps / DISPLAY_FPS))
    frame_idx      = 0

    _lock  = threading.Lock()
    _state = {
        "landmarks":  None,
        "detection":  None,
        "kp":         None,
        "angles":     None,
        "stage":      "",
        "counter":    0,
        "last_score": None,
        "feedback":   "",
        "rep_scores":    [],
        "rep_feedbacks": [],
    }

    infer_q  = _queue.Queue(maxsize=2)
    stop_evt = threading.Event()

    def _infer_worker():
        while not stop_evt.is_set():
            try:
                infer_rgb = infer_q.get(timeout=0.1)
            except _queue.Empty:
                continue

            detection = barbell_tracker.detect(infer_rgb) if barbell_tracker.enabled else None
            results   = pose_detector.process(infer_rgb)

            with _lock:
                _state["detection"] = detection
                if results.pose_landmarks:
                    _state["landmarks"] = results.pose_landmarks
                    kp     = extract_keypoints(results.pose_landmarks)
                    angles = compute_angles(kp)
                    _state["kp"]     = kp
                    _state["angles"] = angles

                    new_stage, rep_done, issues = checker.check(
                        angles, kp, _state["stage"]
                    )
                    _state["stage"] = new_stage

                    if rep_done:
                        _state["counter"] += 1
                        sc = compute_score(issues, checker.rep_metrics, exercise)
                        _state["last_score"] = sc
                        fb = (
                            f"Score: {sc}/100\n"
                            + "\n".join(issues)
                        )
                        _state["feedback"] = fb
                        _state["rep_scores"].append(sc)
                        _state["rep_feedbacks"].append(
                            f"Rep {_state['counter']} - {sc}/100:\n{fb}"
                        )

            infer_q.task_done()

    worker = threading.Thread(target=_infer_worker, daemon=True)
    worker.start()

    rep_overlay_lines:    list = []
    overlay_frames_left:  int = 0
    prev_counter:         int = 0
    last_frame_rgb: Optional[np.ndarray] = None
    DISPLAY_MAX = 480
    next_yield_time = time.monotonic()

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1

        h, w  = frame_bgr.shape[:2]
        scale = min(1.0, INFER_MAX_DIM / max(h, w))
        if scale < 1.0:
            infer_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            infer_bgr = frame_bgr
        infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)

        try:
            infer_q.put_nowait(infer_rgb)
        except _queue.Full:
            pass

        if frame_idx % SKIP_FRAMES != 0:
            continue

        annotated = frame_bgr.copy()

        with _lock:
            landmarks  = _state["landmarks"]
            detection  = _state["detection"]
            kp         = _state["kp"]
            angles_snap = _state["angles"]
            stage      = _state["stage"]
            counter    = _state["counter"]
            last_score = _state["last_score"]
            feedback   = _state["feedback"]

        if counter > prev_counter:
            issue_lines       = [l for l in feedback.split("\n") if not l.startswith("Score:")]
            rep_overlay_lines = [
                f"Rep {counter} - {last_score}/100:"
            ] + issue_lines
            overlay_frames_left = OVERLAY_DURATION
            prev_counter        = counter

        if landmarks:
            _draw_exercise_landmarks(annotated, landmarks, exercise)
        if kp and angles_snap:
            _draw_angle_markers(annotated, kp, angles_snap, exercise)
        if barbell_tracker.enabled:
            barbell_tracker.draw(annotated, detection)

        stage_line = f"Stage: {stage or '-'}  |  Rep {counter + 1}"
        if overlay_frames_left > 0 and rep_overlay_lines:
            _draw_feedback_overlay(annotated, [stage_line] + rep_overlay_lines)
            overlay_frames_left -= 1
        else:
            _draw_feedback_overlay(annotated, [stage_line])

        dh, dw = annotated.shape[:2]
        if max(dh, dw) > DISPLAY_MAX:
            dscale = DISPLAY_MAX / max(dh, dw)
            display = cv2.resize(annotated, (int(dw * dscale), int(dh * dscale)),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            display = annotated

        last_frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        now = time.monotonic()
        wait = next_yield_time - now
        if wait > 0:
            time.sleep(wait)
        next_yield_time = max(time.monotonic(), next_yield_time) + FRAME_INTERVAL

        yield last_frame_rgb, feedback

    stop_evt.set()
    worker.join(timeout=5)
    cap.release()

    with _lock:
        counter       = _state["counter"]
        rep_scores    = list(_state["rep_scores"])
        rep_feedbacks = list(_state["rep_feedbacks"])

    avg     = int(sum(rep_scores) / len(rep_scores)) if rep_scores else 0
    summary = (
        f"Tổng số rep   : {counter}\n"
        f"Điểm TB       : {avg}/100\n\n"
        + "\n---\n".join(rep_feedbacks)
    )
    yield last_frame_rgb, summary

def process_video_file(video_path: Optional[str], exercise: str):
    
    if video_path is None:
        return

    checker = EXERCISE_CHECKERS[exercise]
    checker.reset_state()
    barbell_tracker.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    INFER_MAX_DIM    = 640 if _GPU_AVAILABLE else 480
    OVERLAY_DURATION = int(fps * 3)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    stage            = ""
    counter          = 0
    last_score: Optional[int] = None
    rep_scores:    list = []
    rep_feedbacks: list = []
    rep_overlay_lines: list = []
    overlay_frames_left = 0
    prev_counter     = 0
    feedback         = "Đang xử lý..."
    frame_idx        = 0

    _pool = ThreadPoolExecutor(max_workers=1)

    yield None, "Đang xử lý video..."

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1

        h, w  = frame_bgr.shape[:2]
        scale = min(1.0, INFER_MAX_DIM / max(h, w))
        infer_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
        infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)

        barbell_fut = _pool.submit(barbell_tracker.detect, infer_rgb) if barbell_tracker.enabled else None
        results     = pose_detector.process(infer_rgb)
        detection   = barbell_fut.result() if barbell_fut else None

        landmarks = results.pose_landmarks
        kp = angles_snap = None
        if landmarks:
            kp          = extract_keypoints(landmarks)
            angles_snap = compute_angles(kp)
            new_stage, rep_done, issues = checker.check(angles_snap, kp, stage)
            stage = new_stage
            if rep_done:
                counter   += 1
                sc         = compute_score(issues, checker.rep_metrics, exercise)
                last_score = sc
                feedback   = (
                    f"Rep {counter} — Score: {sc}/100\n"
                    + "\n".join(issues)
                )
                rep_scores.append(sc)
                rep_feedbacks.append(
                    f"Rep {counter} - {sc}/100:\n{feedback}"
                )

        annotated = frame_bgr.copy()

        if counter > prev_counter:
            issue_lines       = [l for l in feedback.split("\n") if not l.startswith("Rep")]
            rep_overlay_lines = [f"Rep {counter} - {last_score}/100:"] + issue_lines
            overlay_frames_left = OVERLAY_DURATION
            prev_counter        = counter

        if landmarks:
            _draw_exercise_landmarks(annotated, landmarks, exercise)
        if kp and angles_snap:
            _draw_angle_markers(annotated, kp, angles_snap, exercise)
        if barbell_tracker.enabled:
            barbell_tracker.draw(annotated, detection)

        stage_line = f"Stage: {stage or '-'}  |  Rep {counter + 1}"
        if overlay_frames_left > 0 and rep_overlay_lines:
            _draw_feedback_overlay(annotated, [stage_line] + rep_overlay_lines)
            overlay_frames_left -= 1
        else:
            _draw_feedback_overlay(annotated, [stage_line])

        writer.write(annotated)

        if frame_idx % 30 == 0:
            progress = int(frame_idx / total * 100)
            yield None, f"Đang xử lý... {progress}%\n\n{feedback}"

    cap.release()
    writer.release()
    _pool.shutdown(wait=False)

    avg     = int(sum(rep_scores) / len(rep_scores)) if rep_scores else 0
    summary = (
        f"Tổng số rep   : {counter}\n"
        f"Điểm TB       : {avg}/100\n\n"
        + "\n---\n".join(rep_feedbacks)
    )
    yield out_path, summary

def process_video_streaming(video_path: Optional[str], exercise: str):
    if video_path is None:
        return

    _stream_state["active"].clear()
    while not _stream_state["queue"].empty():
        try:
            _stream_state["queue"].get_nowait()
        except _queue.Empty:
            break
    _stream_state["token"] += 1
    token = _stream_state["token"]

    checker = EXERCISE_CHECKERS[exercise]
    checker.reset_state()
    barbell_tracker.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps              = cap.get(cv2.CAP_PROP_FPS) or 30
    INFER_MAX_DIM    = 480 if _GPU_AVAILABLE else 320
    OVERLAY_DURATION = int(fps * 3)

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    DISPLAY_H = 480
    DISPLAY_W = int(DISPLAY_H * vid_w / vid_h) if vid_h else DISPLAY_H

    _pool = ThreadPoolExecutor(max_workers=1)

    _stream_state["active"].set()

    _HTML = (
        f'<div style="display:flex;justify-content:center;align-items:flex-start;">'
        f'<img src="/stream?t={token}" '
        f'style="width:{DISPLAY_W}px;height:{DISPLAY_H}px;object-fit:fill;">'
        f'</div>'
    )
    yield _HTML, "Đang xử lý..."

    rep_overlay_lines:   list = []
    overlay_frames_left: int = 0
    prev_counter:        int = 0
    cur_feedback:        str = ""
    stage:               str = ""
    counter:             int = 0
    last_score:          Optional[int] = None
    feedback:            str = ""
    rep_scores:          list = []
    rep_feedbacks:       list = []
    live_issue_frames:   dict = {}
    ISSUE_STICKY = max(15, int(fps * 1.5))
    frame_interval  = 1.0 / fps
    next_frame_time = time.monotonic()

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        h, w  = frame_bgr.shape[:2]
        scale = min(1.0, INFER_MAX_DIM / max(h, w))
        infer_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
        infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)

        barbell_fut = (
            _pool.submit(barbell_tracker.detect, infer_rgb)
            if barbell_tracker.enabled else None
        )

        results     = pose_detector.process(infer_rgb)
        landmarks   = results.pose_landmarks
        kp          = None
        angles_snap = None
        if landmarks:
            kp          = extract_keypoints(landmarks)
            angles_snap = compute_angles(kp)
            new_stage, rep_done, issues = checker.check(angles_snap, kp, stage)
            stage = new_stage
            if rep_done:
                counter    += 1
                sc          = compute_score(issues, checker.rep_metrics, exercise)
                last_score  = sc
                feedback    = (
                    f"Score: {sc}/100\n"
                    + "\n".join(issues)
                )
                rep_scores.append(sc)
                rep_feedbacks.append(
                    f"Rep {counter} - {sc}/100:\n{feedback}"
                )
            else:

                for issue in issues:
                    live_issue_frames[issue] = ISSUE_STICKY

        live_issue_frames = {k: v - 1 for k, v in live_issue_frames.items() if v > 1}

        detection = barbell_fut.result() if barbell_fut else None

        annotated = frame_bgr.copy()

        if counter > prev_counter:
            issue_lines        = [l for l in feedback.split("\n") if not l.startswith("Score:")]
            rep_overlay_lines  = [
                f"Rep {counter} - {last_score}/100:"
            ] + issue_lines
            overlay_frames_left = OVERLAY_DURATION
            prev_counter        = counter
            live_issue_frames.clear()

        if landmarks:
            _draw_exercise_landmarks(annotated, landmarks, exercise)
        if kp and angles_snap:
            _draw_angle_markers(annotated, kp, angles_snap, exercise)
        if barbell_tracker.enabled:
            barbell_tracker.draw(annotated, detection)

        stage_line  = f"Stage: {stage or '-'}  |  Rep {counter}"
        live_lines  = list(live_issue_frames.keys())
        if overlay_frames_left > 0 and rep_overlay_lines:
            _draw_feedback_overlay(annotated, [stage_line] + rep_overlay_lines)
            overlay_frames_left -= 1
        elif live_lines:
            _draw_feedback_overlay(annotated, [stage_line] + live_lines)
        else:
            _draw_feedback_overlay(annotated, [stage_line])

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            _stream_state["queue"].put_nowait(jpeg.tobytes())
        except _queue.Full:
            pass

        now  = time.monotonic()
        wait = next_frame_time - now
        if wait > 0:
            time.sleep(wait)
        next_frame_time = max(time.monotonic(), next_frame_time) + frame_interval

        if feedback != cur_feedback:
            cur_feedback = feedback
            yield _HTML, feedback

    _stream_state["active"].clear()
    _pool.shutdown(wait=False)
    cap.release()

    final_done, final_issues = checker.finalize()
    if final_done:
        counter   += 1
        sc         = compute_score(final_issues, checker.rep_metrics, exercise)
        rep_scores.append(sc)
        rep_feedbacks.append(
            f"Rep {counter} - {sc}/100:\nScore: {sc}/100\n" + "\n".join(final_issues)
        )

    avg     = int(sum(rep_scores) / len(rep_scores)) if rep_scores else 0
    summary = (
        f"Tổng số rep   : {counter}\n"
        f"Điểm TB       : {avg}/100\n\n"
        + "\n---\n".join(rep_feedbacks)
    )
    yield _HTML, summary

from __future__ import annotations

def _pts(val: float, breakpoints: list[float], scores: list[float]) -> float:
    if val <= breakpoints[0]:
        return float(scores[0])
    if val >= breakpoints[-1]:
        return float(scores[-1])
    for i in range(len(breakpoints) - 1):
        x0, x1 = breakpoints[i], breakpoints[i + 1]
        if x0 <= val <= x1:
            t = (val - x0) / (x1 - x0)
            return scores[i] + t * (scores[i + 1] - scores[i])
    return float(scores[-1])

def _squat_score(m: dict) -> int:
    depth = _pts(
        m.get("min_knee", 100),
        [80,  90,  100, 110, 125],
        [40,  35,   28,  12,   0],
    )
    lean = _pts(
        m.get("worst_lean", 90),
        [40,  50,  60,  70,  80],
        [ 0,   8,  16,  24,  30],
    )
    torso = _pts(
        m.get("torso_std", 0),
        [ 3,   6,  10,  12,  20],
        [20,  15,   8,   3,   0],
    )
    bar = _pts(
        m.get("bar_drift", 0),
        [0.01, 0.03, 0.05, 0.10],
        [10,    7,    3,    0],
    )
    return max(0, min(100, round(depth + lean + torso + bar)))

def _deadlift_score(m: dict) -> int:

    l_hip = _pts(
        m.get("lockout_hip", 170),
        [140, 155, 162, 167, 172],
        [  0,   5,  12,  19,  25],
    )

    l_knee = _pts(
        m.get("lockout_knee", 165),
        [140, 150, 157, 162, 168],
        [  0,   5,  12,  19,  25],
    )

    timing = _pts(
        m.get("hip_rise_delta", 0),
        [0.00, 0.03, 0.05, 0.08, 0.15],
        [25,   20,   12,    4,    0],
    )

    prox = _pts(
        m.get("max_bar_shin", 0),
        [0.01, 0.04, 0.08, 0.12],
        [15,   10,    4,    0],
    )

    drift_pts = 0 if m.get("bar_drift", False) else 10
    return max(0, min(100, round(l_hip + l_knee + timing + prox + drift_pts)))

def _bench_score(m: dict) -> int:
    bot_rom = _pts(
        m.get("min_elbow", 80),
        [60,  70,  80,  90, 100],
        [20,  17,  12,   5,   0],
    )
    top_rom = _pts(
        m.get("max_elbow", 160),
        [140, 148, 155, 160, 168],
        [  0,   5,  10,  16,  20],
    )
    sym = _pts(
        m.get("lockout_diff", 0),
        [ 3,   8,  12,  15,  25],
        [20,  14,   8,   2,   0],
    )
    flare = _pts(
        m.get("worst_flare", 60),
        [45,  58,  67,  75,  90],
        [15,  11,   6,   1,   0],
    )
    arch = _pts(
        m.get("worst_back_angle", 0),
        [ 0,  10,  20,  30,  45],
        [15,  13,   8,   2,   0],
    )
    wrist = _pts(
        m.get("worst_wrist_bend", 0),
        [0.02, 0.04, 0.06, 0.10],
        [5,    4,    2,    0],
    )
    bar = _pts(
        m.get("bar_drift", 0),
        [0.01, 0.02, 0.04, 0.08],
        [5,    4,    2,    0],
    )
    return max(0, min(100, round(bot_rom + top_rom + sym + flare + arch + wrist + bar)))

_ISSUE_PENALTIES: dict[str, int] = {
    "uneven arms":              15,
    "grip too wide":            10,
    "excessive arch":           20,
    "uneven lockout":           15,
    "insufficient depth":       10,
    "incomplete lockout":       10,
    "inconsistent touch point": 10,
    "bar drifting":             15,
    "knees collapsing":         20,
    "excessive forward lean":   15,
    "left/right imbalance":     10,
    "inconsistent torso":       10,
    "back rounding":            25,
    "grip too narrow":          10,
    "hips rising too early":    20,
    "bar too far from shins":   15,
    "bar drifting forward":     15,
}

def _penalty_score(issues: list[str]) -> int:
    score = 100
    for issue in issues:
        lower = issue.lower()
        for keyword, penalty in _ISSUE_PENALTIES.items():
            if keyword in lower:
                score -= penalty
                break
    return max(0, score)

def compute_score(
    issues: list[str],
    metrics: dict | None = None,
    exercise: str = "",
) -> int:
    if metrics:
        if exercise == "Back Squat":
            return _squat_score(metrics)
        if exercise == "Deadlift":
            return _deadlift_score(metrics)
        if exercise == "Bench Press":
            return _bench_score(metrics)
    return _penalty_score(issues)

def score_color_bgr(score: int) -> tuple[int, int, int]:
    if score >= 90:
        return (0, 255, 0)
    if score >= 75:
        return (0, 200, 100)
    if score >= 55:
        return (0, 165, 255)
    return (0, 0, 255)

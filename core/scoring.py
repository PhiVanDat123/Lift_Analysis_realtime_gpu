"""
Rep scoring system.

Each rep starts at 100 points. Every distinct form issue found
deducts a fixed penalty. Final score clamped to [0, 100].

To add a new rule:
  1. Add a (keyword, penalty) entry to ISSUE_PENALTIES.
  2. Make sure the warning string in the exercise checker
     contains that keyword (case-insensitive).
"""
from __future__ import annotations

ISSUE_PENALTIES: dict[str, int] = {
    # ── Bench Press ───────────────────────────────────────────────────────────
    "uneven arms":              15,   # arm asymmetry during lift
    "grip too wide":            10,   # grip too wide
    "excessive arch":           20,   # excessive arch
    "uneven lockout":           15,   # uneven lockout
    "insufficient depth":       10,   # bar not near chest
    "incomplete lockout":       10,   # arms not fully extended at top
    "inconsistent touch point": 10,   # chest touchpoint varies
    "bar drifting":             15,   # bar path drift (bench + deadlift shared keyword)

    # ── Squat ─────────────────────────────────────────────────────────────────
    "knees collapsing":         20,   # knee cave
    "excessive forward lean":   15,   # forward lean
    "insufficient depth":       10,   # depth not reached (shared with bench)
    "left/right imbalance":     10,   # left/right asymmetry
    "inconsistent torso":       10,   # torso angle varies

    # ── Deadlift ──────────────────────────────────────────────────────────────
    "back rounding":            25,   # back rounds during pull
    "grip too narrow":          10,   # grip too narrow
    "hips rising too early":    20,   # hips shoot up before shoulders
    "bar too far from shins":   15,   # bar drifts away from legs
    "incomplete lockout":       15,   # hips/knees not fully extended (shared with bench)
    "bar drifting forward":     15,   # mid-pull bar drift (more specific — matched before "bar drifting")
}


def compute_score(issues: list[str]) -> int:
    """Start at 100, deduct for each issue. Clamp to [0, 100]."""
    score = 100
    for issue in issues:
        if issue.startswith("✅"):
            continue
        lower = issue.lower()
        for keyword, penalty in ISSUE_PENALTIES.items():
            if keyword in lower:
                score -= penalty
                break
    return max(0, score)


def score_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Average"
    return "Needs Work"


def score_color_bgr(score: int) -> tuple[int, int, int]:
    if score >= 90:
        return (0, 255, 0)
    if score >= 75:
        return (0, 200, 100)
    if score >= 55:
        return (0, 165, 255)
    return (0, 0, 255)

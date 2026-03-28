"""
MediaPipe pose detection wrapper and geometry helpers.
"""
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,
)


def calculate_angle(a, b, c) -> float:
    """Angle at vertex b formed by points a-b-c (degrees)."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def extract_keypoints(pose_landmarks) -> dict:
    lm = pose_landmarks.landmark
    PL = mp_pose.PoseLandmark

    def _pt(p) -> list:
        return [p.x, p.y, p.visibility]

    return {
        "nose":           _pt(lm[PL.NOSE]),
        "left_shoulder":  _pt(lm[PL.LEFT_SHOULDER]),
        "right_shoulder": _pt(lm[PL.RIGHT_SHOULDER]),
        "left_elbow":     _pt(lm[PL.LEFT_ELBOW]),
        "right_elbow":    _pt(lm[PL.RIGHT_ELBOW]),
        "left_wrist":     _pt(lm[PL.LEFT_WRIST]),
        "right_wrist":    _pt(lm[PL.RIGHT_WRIST]),
        "left_hip":       _pt(lm[PL.LEFT_HIP]),
        "right_hip":      _pt(lm[PL.RIGHT_HIP]),
        "left_knee":      _pt(lm[PL.LEFT_KNEE]),
        "right_knee":     _pt(lm[PL.RIGHT_KNEE]),
        "left_ankle":     _pt(lm[PL.LEFT_ANKLE]),
        "right_ankle":    _pt(lm[PL.RIGHT_ANKLE]),
    }


def joints_visible(kp: dict, *keys: str, threshold: float = 0.5) -> bool:
    """Return True only if all named joints have visibility ≥ threshold."""
    return all(kp[k][2] >= threshold for k in keys)


def compute_angles(kp: dict) -> dict:
    return {
        "left_elbow":     calculate_angle(kp["left_shoulder"],  kp["left_elbow"],    kp["left_wrist"]),
        "right_elbow":    calculate_angle(kp["right_shoulder"], kp["right_elbow"],   kp["right_wrist"]),
        "left_shoulder":  calculate_angle(kp["left_elbow"],     kp["left_shoulder"], kp["left_hip"]),
        "right_shoulder": calculate_angle(kp["right_elbow"],    kp["right_shoulder"],kp["right_hip"]),
        "left_hip":       calculate_angle(kp["left_shoulder"],  kp["left_hip"],      kp["left_knee"]),
        "right_hip":      calculate_angle(kp["right_shoulder"], kp["right_hip"],     kp["right_knee"]),
        "left_knee":      calculate_angle(kp["left_hip"],       kp["left_knee"],     kp["left_ankle"]),
        "right_knee":     calculate_angle(kp["right_hip"],      kp["right_knee"],    kp["right_ankle"]),
    }


def draw_skeleton(frame_bgr, pose_landmarks) -> None:
    mp_drawing.draw_landmarks(
        frame_bgr,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style(),
    )

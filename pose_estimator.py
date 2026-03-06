import math
import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional
from config import MEDIAPIPE_CONFIG, JOINT_ANGLES

MP_LANDMARKS = {lm.name: lm.value for lm in mp.solutions.pose.PoseLandmark}


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float
    name: str = ""


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp_ms: float
    landmarks: dict = field(default_factory=dict)
    joint_angles: dict = field(default_factory=dict)
    detected: bool = False
    phase: str = "unknown"
    bowler_bbox: tuple = None

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "detected": self.detected,
            "phase": self.phase,
            "bowler_bbox": self.bowler_bbox,
            "landmarks": {
                name: {
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                    "visibility": round(lm.visibility, 3),
                }
                for name, lm in self.landmarks.items()
            },
            "joint_angles": {k: round(v, 2) for k, v in self.joint_angles.items()},
        }


def angle_between_three_points(A, B, C) -> float:
    MIN_VISIBILITY = 0.4
    if A.visibility < MIN_VISIBILITY or B.visibility < MIN_VISIBILITY or C.visibility < MIN_VISIBILITY:
        return 0.0
    BA = np.array([A.x - B.x, A.y - B.y, A.z - B.z])
    BC = np.array([C.x - B.x, C.y - B.y, C.z - B.z])
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    if norm_BA == 0 or norm_BC == 0:
        return 0.0
    cos_angle = np.clip(np.dot(BA, BC) / (norm_BA * norm_BC), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


class BowlerTracker:
    """Tracks the most active (moving) person using background subtraction."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40, detectShadows=False)
        self.tracked_box = None
        self.lost_frames = 0
        self.MAX_LOST = 15
        self.MIN_AREA = 800
        self.warmup_frames = 0

    def update(self, frame: np.ndarray) -> Optional[tuple]:
        h, w = frame.shape[:2]
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)

        self.warmup_frames += 1
        if self.warmup_frames < 10:
            return self.tracked_box

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.lost_frames += 1
            if self.lost_frames > self.MAX_LOST:
                self.tracked_box = None
            return self.tracked_box

        valid = [c for c in contours if cv2.contourArea(c) > self.MIN_AREA]
        if not valid:
            self.lost_frames += 1
            return self.tracked_box

        best_contour = None
        best_score = -1
        for c in valid:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            h_centre_dist = abs(cx / w - 0.5)
            v_ok = 0.1 < cy / h < 0.9
            if not v_ok:
                continue
            score = area * (1 - h_centre_dist * 0.5)
            if score > best_score:
                best_score = score
                best_contour = c

        if best_contour is None:
            self.lost_frames += 1
            return self.tracked_box

        x, y, bw, bh = cv2.boundingRect(best_contour)
        pad_x = int(bw * 0.6)
        pad_y = int(bh * 0.4)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        if (x2 - x1) < 80 or (y2 - y1) < 120:
            self.lost_frames += 1
            return self.tracked_box

        self.tracked_box = (x1, y1, x2, y2)
        self.lost_frames = 0
        return self.tracked_box


class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)
        self.tracker = BowlerTracker()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.pose.close()

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int, timestamp_ms: float) -> PoseFrame:
        pose_frame = PoseFrame(frame_idx=frame_idx, timestamp_ms=timestamp_ms)
        h, w = frame_bgr.shape[:2]

        bbox = self.tracker.update(frame_bgr)
        pose_frame.bowler_bbox = bbox

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame_bgr
                bbox = None
        else:
            crop = frame_bgr

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb.flags.writeable = False
        results = self.pose.process(crop_rgb)
        crop_rgb.flags.writeable = True

        if not results.pose_landmarks:
            pose_frame.detected = False
            return pose_frame

        pose_frame.detected = True

        for lm_name, lm_idx in MP_LANDMARKS.items():
            raw = results.pose_landmarks.landmark[lm_idx]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                abs_x = x1 + raw.x * (x2 - x1)
                abs_y = y1 + raw.y * (y2 - y1)
                norm_x = abs_x / w
                norm_y = abs_y / h
            else:
                norm_x = raw.x
                norm_y = raw.y
            pose_frame.landmarks[lm_name] = Landmark(
                x=norm_x, y=norm_y, z=raw.z, visibility=raw.visibility, name=lm_name)

        for angle_name, lm_A, lm_B, lm_C in JOINT_ANGLES:
            if all(k in pose_frame.landmarks for k in [lm_A, lm_B, lm_C]):
                angle = angle_between_three_points(
                    pose_frame.landmarks[lm_A],
                    pose_frame.landmarks[lm_B],
                    pose_frame.landmarks[lm_C])
                pose_frame.joint_angles[angle_name] = angle

        return pose_frame

    def draw_landmarks(self, frame_bgr: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
        frame_out = frame_bgr.copy()
        h, w = frame_out.shape[:2]

        if pose_frame.bowler_bbox:
            x1, y1, x2, y2 = pose_frame.bowler_bbox
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_out, "Bowler", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if not pose_frame.detected:
            cv2.putText(frame_out, f"Phase: {pose_frame.phase}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_out, f"Frame: {pose_frame.frame_idx}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return frame_out

        for connection in self.mp_pose.POSE_CONNECTIONS:
            s = self.mp_pose.PoseLandmark(connection[0]).name
            e = self.mp_pose.PoseLandmark(connection[1]).name
            if s in pose_frame.landmarks and e in pose_frame.landmarks:
                ls = pose_frame.landmarks[s]
                le = pose_frame.landmarks[e]
                if ls.visibility > 0.3 and le.visibility > 0.3:
                    cv2.line(frame_out,
                             (int(ls.x * w), int(ls.y * h)),
                             (int(le.x * w), int(le.y * h)),
                             (0, 255, 0), 2)

        for lm in pose_frame.landmarks.values():
            if lm.visibility > 0.3:
                cv2.circle(frame_out, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

        key_joints = {"right_elbow": "RIGHT_ELBOW", "left_elbow": "LEFT_ELBOW",
                      "right_shoulder": "RIGHT_SHOULDER", "right_knee": "RIGHT_KNEE"}
        for angle_name, lm_name in key_joints.items():
            if angle_name in pose_frame.joint_angles and lm_name in pose_frame.landmarks:
                lm = pose_frame.landmarks[lm_name]
                if lm.visibility > 0.3:
                    cv2.putText(frame_out, f"{pose_frame.joint_angles[angle_name]:.0f}deg",
                                (int(lm.x * w) + 8, int(lm.y * h) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(frame_out, f"Phase: {pose_frame.phase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_out, f"Frame: {pose_frame.frame_idx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return frame_out


    def reset_tracker(self):
        """Reset BowlerTracker state — call between passes or after long gaps."""
        self.tracker = BowlerTracker()
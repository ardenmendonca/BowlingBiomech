"""
config.py - Central configuration for cricket pose analysis pipeline
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
POSE_JSON_DIR = os.path.join(OUTPUT_DIR, "pose_json")
ANNOTATED_VIDEO_DIR = os.path.join(OUTPUT_DIR, "annotated_videos")
CONTEXT_DIR = os.path.join(OUTPUT_DIR, "llm_context")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create dirs if they don't exist
for d in [VIDEO_DIR, POSE_JSON_DIR, ANNOTATED_VIDEO_DIR, CONTEXT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── MediaPipe Pose Settings ────────────────────────────────────────────────────
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 2,          # 0=lite, 1=full, 2=heavy (most accurate)
    "smooth_landmarks": False,
    "enable_segmentation": False,
    "min_detection_confidence": 0.3,
    "min_tracking_confidence": 0.3,
}

# ── Video Processing ───────────────────────────────────────────────────────────
FRAME_SKIP = 1          # Process every Nth frame (1 = all frames)
MAX_FRAMES = None       # None = process entire video
RESIZE_WIDTH = None     # Resize frame width for processing (None = no resize)

# ── Bowling Phase Detection Thresholds ────────────────────────────────────────
# Phases: run_up → load → delivery_stride → release → follow_through
PHASE_CONFIG = {
    # Wrist height relative to shoulder triggers "release" detection
    "release_wrist_above_shoulder_ratio": 0.8,
    # Minimum frames to consider a phase
    "min_phase_frames": 3,
}

# ── Joint Angle Definitions ───────────────────────────────────────────────────
# Each entry: (joint_name, landmark_A, landmark_B, landmark_C)
# Angle is computed at landmark_B
JOINT_ANGLES = [
    ("right_elbow",    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("left_elbow",     "LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST"),
    ("right_shoulder", "RIGHT_HIP",      "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("left_shoulder",  "LEFT_HIP",       "LEFT_SHOULDER",  "LEFT_ELBOW"),
    ("right_knee",     "RIGHT_HIP",      "RIGHT_KNEE",  "RIGHT_ANKLE"),
    ("left_knee",      "LEFT_HIP",       "LEFT_KNEE",   "LEFT_ANKLE"),
    ("right_hip",      "RIGHT_SHOULDER", "RIGHT_HIP",   "RIGHT_KNEE"),
    ("left_hip",       "LEFT_SHOULDER",  "LEFT_HIP",    "LEFT_KNEE"),
    ("trunk_lean",     "LEFT_HIP",       "LEFT_SHOULDER", "RIGHT_SHOULDER"),
]

# ── Kaggle Dataset IDs ─────────────────────────────────────────────────────────
KAGGLE_DATASETS = {
    "bowl_release": "dzambrano/cricket-bowlrelease-dataset",
    "cricket_videos": "shujaanazhar/cricket-videos-of-legal-wide-lbw-and-no-balls",
}

# ── Output Settings ────────────────────────────────────────────────────────────
ANNOTATED_VIDEO_FPS = 30
SKELETON_COLOR = (0, 255, 0)        # BGR green
JOINT_POINT_COLOR = (0, 0, 255)     # BGR red
JOINT_POINT_RADIUS = 5
SKELETON_THICKNESS = 2
ANGLE_TEXT_COLOR = (255, 255, 0)    # BGR cyan
FONT_SCALE = 0.5

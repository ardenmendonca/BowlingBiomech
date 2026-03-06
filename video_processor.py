import os
import cv2
import json
import sys
import time
import numpy as np
from tqdm import tqdm
from typing import Optional

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from config import (
    FRAME_SKIP, MAX_FRAMES, RESIZE_WIDTH,
    POSE_JSON_DIR, ANNOTATED_VIDEO_DIR, ANNOTATED_VIDEO_FPS,
)
from pose_estimator import PoseEstimator, PoseFrame

KEY_LANDMARKS = [
    "RIGHT_SHOULDER", "LEFT_SHOULDER",
    "RIGHT_ELBOW",    "LEFT_ELBOW",
    "RIGHT_WRIST",    "LEFT_WRIST",
    "RIGHT_HIP",      "LEFT_HIP",
    "RIGHT_KNEE",     "LEFT_KNEE",
    "RIGHT_ANKLE",    "LEFT_ANKLE",
]

MIN_VISIBILITY = 0.30


def frame_reliability(pose_frame: PoseFrame) -> float:
    scores = [
        pose_frame.landmarks[n].visibility
        for n in KEY_LANDMARKS
        if n in pose_frame.landmarks
    ]
    return round(float(np.mean(scores)), 3) if scores else 0.0


def wrist_shoulder_offset(pose_frame: PoseFrame) -> Optional[float]:
    """Right wrist height above right shoulder (positive = wrist higher)."""
    lms = pose_frame.landmarks
    if "RIGHT_WRIST" in lms and "RIGHT_SHOULDER" in lms:
        rw = lms["RIGHT_WRIST"]
        rs = lms["RIGHT_SHOULDER"]
        if rw.visibility > MIN_VISIBILITY and rs.visibility > MIN_VISIBILITY:
            return round(rs.y - rw.y, 4)  # positive = wrist above shoulder
    return None


def resize_frame(frame, width):
    if width is None:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)))


def process_video(
    video_path: str,
    save_annotated: bool = True,
    save_json: bool = True,
    show_preview: bool = False,
) -> dict:

    print(f"START: {video_path}", flush=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    basename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w      = RESIZE_WIDTH or frame_w
    out_h      = int(frame_h * (out_w / frame_w))

    print(f"  {frame_w}x{frame_h} -> {out_w}x{out_h} | {fps:.1f}fps | {total} frames", flush=True)

    # ── Video writer ───────────────────────────────────────────────────────────
    writer = None
    out_video_path = None
    if save_annotated:
        out_video_path = os.path.join(ANNOTATED_VIDEO_DIR, f"{basename}_pose.mp4")
        writer = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            ANNOTATED_VIDEO_FPS,
            (out_w, out_h),
        )

    lean_frames = []
    frame_idx   = 0
    processed   = 0
    max_frames  = MAX_FRAMES or total
    start_time  = time.time()

    pbar = tqdm(total=min(total, max_frames), desc="Processing", unit="fr")

    with PoseEstimator() as estimator:
        while True:
            ret, frame = cap.read()
            if not ret or processed >= max_frames:
                break

            frame_idx += 1
            if (frame_idx - 1) % FRAME_SKIP != 0:
                continue

            processed += 1
            ts_ms = (frame_idx / fps) * 1000
            frame_proc = resize_frame(frame, RESIZE_WIDTH)
            pf = estimator.process_frame(frame_proc, frame_idx, ts_ms)

            if pf.detected:
                rel = frame_reliability(pf)
                wrist_sig = wrist_shoulder_offset(pf)

                if rel >= MIN_VISIBILITY:
                    lean_frames.append({
                        "f":   frame_idx,
                        "t":   round(ts_ms, 0),
                        "rel": rel,
                        "w":   wrist_sig,          
                        "a":   {                    
                            k: round(v, 1)
                            for k, v in pf.joint_angles.items()
                            if v and v > 0
                        },
                    })

            if writer:
                annotated = estimator.draw_landmarks(frame_proc, pf)
                writer.write(annotated)

            if show_preview:
                cv2.imshow("Pose", estimator.draw_landmarks(frame_proc, pf))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"  Processed {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps)", flush=True)
    print(f"  Reliable frames kept: {len(lean_frames)} / {processed}", flush=True)

    out_json_path = None
    if save_json:
        out_json_path = os.path.join(POSE_JSON_DIR, f"{basename}_pose.json")
        output = {
            "video":      os.path.basename(video_path),
            "fps":        round(fps, 2),
            "total_frames": total,
            "reliable_frames": len(lean_frames),
            "frames":     lean_frames,
        }
        with open(out_json_path, "w") as f:
            json.dump(output, f, separators=(",", ":"))  # compact, no whitespace

        size_kb = os.path.getsize(out_json_path) / 1024
        print(f"  JSON saved: {out_json_path} ({size_kb:.0f} KB)", flush=True)

    if out_video_path:
        print(f"  Video saved: {out_video_path}", flush=True)

    return {
        "video_path":          video_path,
        "annotated_video_path": out_video_path,
        "json_path":           out_json_path,
        "reliable_frames":     len(lean_frames),
    }


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else None
    if not video:
        print("Usage: python video_processor.py <video_path>")
        sys.exit(1)
    process_video(video)
import os
import json
import sys
import argparse
from typing import Optional
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARIES_DIR = os.path.join(BASE_DIR, "outputs", "summaries")
os.makedirs(SUMMARIES_DIR, exist_ok=True)

KEY_ANGLES = [
    "right_elbow", "left_elbow",
    "right_shoulder", "left_shoulder",
    "right_knee", "left_knee",
    "trunk_lean",
]

MIN_RELIABILITY = 0.55
MIN_ELBOW_AT_RELEASE = 100  


def find_deliveries(frames: list, fps: float) -> list:
    """Find delivery peaks from wrist signal."""

    # Extract wrist signal
    idxs   = [f["f"] for f in frames if f.get("w") is not None]
    signal = [f["w"] for f in frames if f.get("w") is not None]

    if len(signal) < 10:
        print("Not enough wrist data")
        return []

    signal = np.array(signal)
    idxs   = np.array(idxs)

    # Smooth
    k = max(3, int(fps * 0.1))
    if k % 2 == 0: k += 1
    smoothed = np.convolve(signal, np.ones(k)/k, mode="same")

    # Find peaks
    min_dist = int(fps * 1.5)
    peaks, _ = find_peaks(smoothed, height=0.05, distance=min_dist, prominence=0.05)

    print(f"Found {len(peaks)} delivery peaks", flush=True)

    frame_by_idx = {f["f"]: f for f in frames}
    before = int(fps * 1.5)
    after  = int(fps * 2.0)

    deliveries = []
    for i, peak in enumerate(peaks):
        release_idx = int(idxs[peak])
        start_idx   = max(idxs[0],  release_idx - before)
        end_idx     = min(idxs[-1], release_idx + after)

        window = [f for f in frames if start_idx <= f["f"] <= end_idx]
        if len(window) < 5:
            continue

        deliveries.append({
            "num":         i + 1,
            "release_idx": release_idx,
            "t_sec":       round(release_idx / fps, 2),
            "frames":      window,
        })

    return deliveries


def assign_phases(delivery: dict) -> dict:
    frames      = delivery["frames"]
    release_idx = delivery["release_idx"]
    idxs        = [f["f"] for f in frames]

    try:
        rel_pos = idxs.index(release_idx)
    except ValueError:
        rel_pos = int(np.argmin([abs(i - release_idx) for i in idxs]))

    for i, f in enumerate(frames):
        pos = i - rel_pos
        if pos < -int(0.6 * rel_pos):
            f["phase"] = "run_up"
        elif pos < -3:
            f["phase"] = "load"
        elif pos < 0:
            f["phase"] = "delivery_stride"
        elif pos <= 3:
            f["phase"] = "release"
        else:
            f["phase"] = "follow_through"

    return delivery


def summarize_delivery(delivery: dict) -> Optional[dict]:
    frames = delivery["frames"]

    # Release frame — best reliability in release phase
    rel_frames = [f for f in frames
                  if f.get("phase") == "release" and f.get("rel", 0) >= MIN_RELIABILITY]

    if not rel_frames:
        return None

    best_rel = max(rel_frames, key=lambda f: f["rel"])
    release_angles = best_rel.get("a", {})

    # Filter: if no right elbow or elbow too low — likely not the bowler
    if release_angles.get("right_elbow", 0) < MIN_ELBOW_AT_RELEASE:
        if release_angles.get("left_elbow", 0) < MIN_ELBOW_AT_RELEASE:
            return None  # skip — looks like umpire or fielder

    # Phase means from reliable frames
    phase_means = {}
    for phase in ["run_up", "load", "delivery_stride", "release", "follow_through"]:
        pool = [f for f in frames
                if f.get("phase") == phase and f.get("rel", 0) >= MIN_RELIABILITY]
        if not pool:
            continue
        phase_means[phase] = {}
        for angle in KEY_ANGLES:
            vals = [f["a"][angle] for f in pool if angle in f.get("a", {}) and f["a"][angle] > 0]
            if vals:
                phase_means[phase][angle] = round(float(np.mean(vals)), 1)

    # Progression load → release
    progression = {}
    for angle in KEY_ANGLES:
        load_val    = phase_means.get("load", {}).get(angle)
        release_val = phase_means.get("release", {}).get(angle)
        if load_val and release_val:
            progression[angle] = round(release_val - load_val, 1)

    return {
        "delivery":     delivery["num"],
        "t_sec":        delivery["t_sec"],
        "release_rel":  round(best_rel["rel"], 3),
        "release_angles": release_angles,
        "phase_means":  phase_means,
        "progression":  progression,
        "phases":       list({f.get("phase") for f in frames}),
    }


def to_text(s: dict) -> str:
    lines = []
    tag = "✅" if s["release_rel"] > 0.6 else "⚠️"
    lines.append(f"\nDELIVERY {s['delivery']}  t={s['t_sec']}s  "
                 f"rel={s['release_rel']} {tag}")
    lines.append(f"  Phases: {', '.join(s['phases'])}")

    if s["release_angles"]:
        lines.append("  At release:")
        for k, v in s["release_angles"].items():
            lines.append(f"    {k:<26} {v:>6.1f}°")

    if s["progression"]:
        lines.append("  Progression (load→release):")
        for k, v in s["progression"].items():
            arrow = "↑" if v > 0 else "↓"
            lines.append(f"    {k:<26} {v:>+6.1f}° {arrow}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--max",  type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.json}...", flush=True)
    data   = json.load(open(args.json))
    frames = data.get("frames", [])
    fps    = data.get("fps", 25.0)
    video  = data.get("video", "unknown")

    print(f"Video: {video} | {len(frames)} reliable frames | {fps} fps", flush=True)

    deliveries = find_deliveries(frames, fps)
    if not deliveries:
        print("No deliveries found.")
        sys.exit(1)

    if args.max:
        deliveries = deliveries[:args.max]

    summaries = []
    for d in deliveries:
        d = assign_phases(d)
        s = summarize_delivery(d)
        if s:
            summaries.append(s)

    print(f"Clean deliveries after filtering: {len(summaries)}", flush=True)

    basename = os.path.splitext(os.path.basename(args.json))[0].replace("_pose","")
    out_json = os.path.join(SUMMARIES_DIR, f"{basename}_deliveries.json")
    out_txt  = os.path.join(SUMMARIES_DIR, f"{basename}_deliveries.txt")

    json.dump({
        "video": video, "fps": fps,
        "total_deliveries": len(summaries),
        "generated_at": datetime.now().isoformat(),
        "deliveries": summaries,
    }, open(out_json, "w"), indent=2)

    lines = [
        "=" * 60,
        "DELIVERY-BY-DELIVERY BOWLING ANALYSIS",
        f"Video: {video}",
        f"Deliveries: {len(summaries)}",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
    ]
    for s in summaries:
        lines.append(to_text(s))

    open(out_txt, "w", encoding="utf-8").write("\n".join(lines))

    size_json = os.path.getsize(out_json) / 1024
    size_txt  = os.path.getsize(out_txt)  / 1024
    print(f"Saved: {out_json} ({size_json:.0f} KB)", flush=True)
    print(f"Saved: {out_txt}  ({size_txt:.0f} KB)",  flush=True)

    # Print table
    print(f"\n{'#':>3}  {'Time':>7}  {'Rel':>5}  {'R.Elbow':>8}  {'R.Shoulder':>10}  {'Trunk':>6}")
    print("─" * 50)
    for s in summaries:
        ra = s["release_angles"]
        print(f"{s['delivery']:>3}  {s['t_sec']:>6.1f}s  "
              f"{s['release_rel']:>5.3f}  "
              f"{ra.get('right_elbow', 0):>7.1f}°  "
              f"{ra.get('right_shoulder', 0):>9.1f}°  "
              f"{ra.get('trunk_lean', 0):>5.1f}°")
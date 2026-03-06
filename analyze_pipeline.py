
import os
import cv2
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Optional
from scipy.signal import find_peaks as _find_peaks

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from pose_estimator import PoseEstimator, PoseFrame
from config import FRAME_SKIP, RESIZE_WIDTH, ANNOTATED_VIDEO_DIR

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_VIDEO_DIR, exist_ok=True)

KEY_ANGLES = [
    "right_elbow", "left_elbow",
    "right_shoulder", "left_shoulder",
    "right_knee", "left_knee",
    "trunk_lean",
]

LANDMARK_VIS_FLOOR  = 0.30
PASS1_FRAME_SKIP    = 2
SIGNAL_SMOOTH_SEC   = 0.12
BEFORE_SEC          = 6.0
AFTER_SEC           = 2.5
MIN_FRAMES_DELIVERY = 8


REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# LLM model fallback chain — each has its own free quota bucket
GEMINI_MODEL_CHAIN = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


GROQ_MODEL_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

MAX_RETRIES   = 3
RETRY_BACKOFF = [30, 60, 120]

SYSTEM_PROMPT = """You are an expert cricket bowling biomechanics coach with deep knowledge 
of fast bowling, spin bowling, and injury prevention. You analyse joint angle data 
collected from video pose estimation.

You will receive per-delivery biomechanical data including:
- Joint angles at release (elbow, shoulder, knee, trunk lean)
- Phase-by-phase angle means (run_up, load, delivery_stride, release, follow_through)
- Progression of angles from load to release
- Reliability confidence per delivery (high/medium/low)

Your task is to provide:
1. OBSERVATIONS — identify patterns, inconsistencies, and biomechanical issues
   across deliveries. Be specific — reference delivery numbers and angle values.
2. RECOMMENDATIONS — concrete, actionable coaching improvements
   prioritised by impact (injury risk first, then performance).

Format your response exactly as:

## OBSERVATIONS

### Consistency
[comment on variation across deliveries — standard deviation of key angles]

### Release Mechanics
[elbow extension angle at release, shoulder height, what they indicate]

### Trunk & Posture
[trunk lean patterns across deliveries, what they mean for pace and accuracy]

### Red Flags
[anything suggesting injury risk — hyperextension, asymmetry, collapsed front side]

## RECOMMENDATIONS

### Priority 1 — [title]
[specific drill or correction with clear instruction]

### Priority 2 — [title]
[specific drill or correction with clear instruction]

### Priority 3 — [title]
[specific drill or correction with clear instruction]

Be direct and specific. Reference actual angle values from the data."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def frame_reliability(pf: PoseFrame) -> float:
    key = ["RIGHT_SHOULDER","LEFT_SHOULDER","RIGHT_ELBOW",
           "LEFT_ELBOW","RIGHT_WRIST","RIGHT_HIP","RIGHT_KNEE","LEFT_KNEE"]
    scores = [pf.landmarks[n].visibility for n in key if n in pf.landmarks]
    return round(float(np.mean(scores)), 3) if scores else 0.0


def wrist_signal(pf: PoseFrame) -> Optional[float]:
    lms = pf.landmarks
    rw  = lms.get("RIGHT_WRIST")
    rs  = lms.get("RIGHT_SHOULDER")
    if (rw and rs and
            rw.visibility > LANDMARK_VIS_FLOOR and
            rs.visibility > LANDMARK_VIS_FLOOR):
        return round(rs.y - rw.y, 4)
    return None


def lean_record(pf: PoseFrame, frame_idx: int, ts_ms: float, rel: float) -> dict:
    return {
        "f":   frame_idx,
        "t":   round(ts_ms, 0),
        "rel": rel,
        "w":   wrist_signal(pf),
        "a":   {k: round(v, 1) for k, v in pf.joint_angles.items() if v and v > 0},
    }


def resize(frame, out_w, out_h):
    return cv2.resize(frame, (out_w, out_h)) if RESIZE_WIDTH else frame



def compute_threshold(arr: np.ndarray) -> float:
    """
    Robust threshold between noise floor and genuine release peaks.
    p50 = typical frame value (noise floor across ALL frames)
    p95 = near genuine peaks
    threshold = p50 + 50% of that range
    Avoids mean+std being skewed by noisy baseline frames.
    """
    if len(arr) == 0:
        return 0.04
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    return max(0.04, p50 + 0.5 * (p95 - p50))


def load_api_key(provider: str = "gemini") -> str:
    """
    Load API key for the given provider from env or .env file.
    provider: "gemini" | "groq"
    .env file format:
        GEMINI_API_KEY=your_key
        GROQ_API_KEY=your_key
    """
    env_var = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
    key = os.environ.get(env_var)
    if key:
        return key
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        for line in open(env_path).readlines():
            line = line.strip()
            if line.startswith(f"{env_var}="):
                return line.split("=", 1)[1].strip()
    return None


def get_deliveries_txt(video_path: str) -> str:
    """Find the deliveries txt file for a given video."""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    txt_path = os.path.join(RESULTS_DIR, f"{basename}_deliveries.txt")
    return txt_path if os.path.exists(txt_path) else None


def is_quota_error(e: Exception) -> bool:
    """Check if exception is a 429 quota / rate-limit error."""
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def is_daily_quota_exhausted(e: Exception) -> bool:
    """Distinguish daily quota exhausted (limit: 0) from transient rate limit."""
    return "limit: 0" in str(e)


def parse_retry_delay(e: Exception) -> int:
    """Extract retryDelay seconds from the error message if present."""
    import re
    match = re.search(r"retryDelay.*?(\d+)s", str(e))
    return int(match.group(1)) + 5 if match else 60


# ── Step 1: Video Processor ──────────────────────────────────────────────────


# ── Pass 1: Tracker warmup + batch peak detection ──────────────────────────────

def pass1(video_path: str, fps: float, out_w: int, out_h: int,
          estimator: PoseEstimator) -> dict:
    print(f"\n  Pass 1: Tracker warmup + peak detection "
          f"(every {PASS1_FRAME_SKIP}nd frame)...", flush=True)

    cap       = cv2.VideoCapture(video_path)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    wrist_vals  = []
    wrist_fidxs = []

    pbar = tqdm(total=total // PASS1_FRAME_SKIP, desc="  Pass 1", unit="fr")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % PASS1_FRAME_SKIP != 0:
            continue

        frame_r = resize(frame, out_w, out_h)
        pf = estimator.process_frame(frame_r, frame_idx, frame_idx / fps * 1000)

        if pf.detected:
            w = wrist_signal(pf)
            if w is not None:
                wrist_vals.append(w)
                wrist_fidxs.append(frame_idx)

        pbar.update(1)

    pbar.close()
    cap.release()

    if not wrist_vals:
        print("  No wrist signal found in pass 1", flush=True)
        return {"delivery_frame_targets": [], "threshold": 0.04,
                "gap_frames": int(fps * 3)}

    arr       = np.array(wrist_vals)
    threshold = compute_threshold(arr)

    smooth_k = max(3, int(fps / PASS1_FRAME_SKIP * SIGNAL_SMOOTH_SEC))
    smoothed = np.convolve(arr, np.ones(smooth_k) / smooth_k, mode="same")
    min_dist = int(fps * 3.0 / PASS1_FRAME_SKIP)

    peaks, _ = _find_peaks(
        smoothed,
        height=threshold * 0.8,
        distance=min_dist,
        prominence=threshold * 0.3,
    )

    delivery_frame_targets = [wrist_fidxs[p] for p in peaks]

    if len(peaks) >= 3:
        gaps = np.diff(peaks) * PASS1_FRAME_SKIP
        gap_frames = max(int(fps * 1.0), int(np.median(gaps) * 0.4))
    else:
        gap_frames = int(fps * 2.0)

    print(f"  Threshold: {threshold:.4f}  "
          f"Peaks found: {len(delivery_frame_targets)}  "
          f"Min gap: {gap_frames/fps:.1f}s", flush=True)

    return {
        "delivery_frame_targets": delivery_frame_targets,
        "threshold":  threshold,
        "gap_frames": gap_frames,
        "wrist_p50":  float(np.percentile(arr, 50)),
        "wrist_p95":  float(np.percentile(arr, 95)),
    }


# ── Phase detection ────────────────────────────────────────────────────────────

def find_phases(frames: list, release_idx: int, fps: float) -> list:
    wrist = np.array([r.get("w") or 0.0 for r in frames])
    n     = len(frames)
    idxs  = [r["f"] for r in frames]

    try:    release_pos = idxs.index(release_idx)
    except: release_pos = int(np.argmin([abs(i - release_idx) for i in idxs]))

    if n < 4:
        for i, r in enumerate(frames):
            r["phase"] = ("release"       if i == release_pos else
                          "run_up"        if i < release_pos  else
                          "follow_through")
        return frames

    k        = max(3, min(7, n // 4))
    smoothed = np.convolve(wrist, np.ones(k)/k, mode="same")
    baseline = float(np.mean(smoothed[:max(1, int(n * 0.2))]))
    peak_val = float(smoothed[release_pos])
    rise_thr = baseline + 0.3 * max(peak_val - baseline, 0.01)

    load_start = max(0, release_pos - 1)
    for i in range(release_pos - 1, max(0, release_pos - int(fps * 2)), -1):
        if smoothed[i] < rise_thr:
            load_start = i
            break

    stride_start = load_start + max(1, (release_pos - load_start) // 2)
    release_end  = release_pos + max(1, int(fps * 0.15))

    follow_end = n
    for i in range(release_pos + 1, n):
        if smoothed[i] < rise_thr:
            follow_end = i
            break

    for i, r in enumerate(frames):
        if i < load_start:
            r["phase"] = "run_up"
        elif i < stride_start:
            r["phase"] = "load"
        elif i < release_pos:
            r["phase"] = "delivery_stride"
        elif i <= release_end:
            r["phase"] = "release"
        elif i <= follow_end:
            r["phase"] = "follow_through"
        else:
            r["phase"] = "run_up"

    return frames


# ── Per-delivery summary ───────────────────────────────────────────────────────

def summarize_delivery(frames: list, release_idx: int,
                       num: int, fps: float) -> Optional[dict]:
    rel_frames = [r for r in frames if r.get("phase") == "release"]
    if not rel_frames:
        return None

    best   = max(rel_frames, key=lambda r: r.get("rel", 0))
    angles = best.get("a", {})
    if not angles:
        return None

    rel_score  = best.get("rel", 0)
    confidence = ("high"   if rel_score > 0.65 else
                  "medium" if rel_score > 0.45 else "low")

    phase_means = {}
    for phase in ["run_up","load","delivery_stride","release","follow_through"]:
        pool = [r for r in frames
                if r.get("phase") == phase and r.get("rel", 0) > LANDMARK_VIS_FLOOR]
        if not pool:
            continue
        phase_means[phase] = {}
        for angle in KEY_ANGLES:
            vals = [r["a"][angle] for r in pool
                    if angle in r.get("a", {}) and r["a"][angle] > 0]
            if vals:
                phase_means[phase][angle] = round(float(np.mean(vals)), 1)

    progression = {}
    for angle in KEY_ANGLES:
        lv = phase_means.get("load",    {}).get(angle)
        rv = phase_means.get("release", {}).get(angle)
        if lv and rv:
            progression[angle] = round(rv - lv, 1)

    wrist_vals = [r["w"] for r in frames if r.get("w") is not None]
    wrist_stats = {}
    if wrist_vals:
        base = float(np.mean(wrist_vals[:max(1, len(wrist_vals)//5)]))
        wrist_stats = {
            "peak":     round(float(max(wrist_vals)), 3),
            "baseline": round(base, 3),
            "rise":     round(float(max(wrist_vals)) - base, 3),
        }

    return {
        "delivery":       num,
        "t_sec":          round(release_idx / fps, 2),
        "release_rel":    round(rel_score, 3),
        "confidence":     confidence,
        "phases":         list({r.get("phase","?") for r in frames}),
        "release_angles": angles,
        "phase_means":    phase_means,
        "progression":    progression,
        "wrist_signal":   wrist_stats,
    }


def delivery_to_text(s: dict) -> str:
    icons = {"high": "✅", "medium": "⚠️ ", "low": "❌"}
    lines = [
        f"\nDELIVERY {s['delivery']}  t={s['t_sec']}s  "
        f"rel={s['release_rel']}  confidence={s['confidence']} "
        f"{icons.get(s['confidence'],'')}",
        f"  Phases: {', '.join(sorted(s['phases']))}",
    ]
    if s.get("wrist_signal"):
        ws = s["wrist_signal"]
        lines.append(f"  Wrist: peak={ws['peak']}  "
                     f"baseline={ws['baseline']}  rise={ws['rise']}")
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


def _call_gemini_model(client, model: str, user_prompt: str) -> str:
    """Single Gemini API call — raises on any error."""
    from google.genai import types
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    return response.text



def _step3_groq(txt_path: str, video_path: str, api_key: str) -> str:
    """Call Groq API with model fallback chain."""
    try:
        from groq import Groq
    except ImportError:
        print("  ✗ groq not installed. Run: pip install groq", flush=True)
        return None

    bowling_data = open(txt_path, encoding="utf-8").read()
    user_prompt  = (
        "Here is the biomechanical bowling data from pose estimation analysis.\n"
        "Please provide your observations and recommendations.\n\n"
        + bowling_data
    )

    client     = Groq(api_key=api_key)
    report     = None
    model_used = None

    for model in GROQ_MODEL_CHAIN:
        print(f"  Trying {model}...", flush=True)

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                )
                report     = response.choices[0].message.content
                model_used = model
                print(f" Response from {model}", flush=True)
                break

            except Exception as e:
                if not is_quota_error(e):
                    print(f"  ✗ Non-quota error: {e}", flush=True)
                    return None

                if is_daily_quota_exhausted(e):
                    print(f"  Daily quota exhausted for {model} — "
                          f"trying next model...", flush=True)
                    break

                delay = parse_retry_delay(e)
                if attempt < MAX_RETRIES - 1:
                    print(f" Rate limited — waiting {delay}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES})...", flush=True)
                    time.sleep(delay)
                else:
                    print(f" Still rate limited after {MAX_RETRIES} attempts "
                          f"on {model} — trying next...", flush=True)

        if report:
            break

    if not report:
        print("  ✗ All Groq models exhausted.", flush=True)
        return None

    basename   = os.path.splitext(os.path.basename(video_path))[0]
    out_report = os.path.join(REPORTS_DIR, f"{basename}_report.txt")
    out_md     = os.path.join(REPORTS_DIR, f"{basename}_report.md")

    report_header = "\n".join([
        "=" * 60,
        "GROQ BOWLING ANALYSIS REPORT",
        f"Video:     {os.path.basename(video_path)}",
        f"Model:     {model_used}",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        "",
    ])

    full_report = report_header + report
    open(out_report, "w", encoding="utf-8").write(full_report)
    open(out_md,     "w", encoding="utf-8").write(full_report)

    size = os.path.getsize(out_report) / 1024
    print(f"  Report saved: {out_report} ({size:.0f} KB)", flush=True)
    print(f"\n{'─'*60}", flush=True)
    print(report, flush=True)
    print(f"{'─'*60}", flush=True)

    return out_report


def step3_llm_analysis(txt_path: str, video_path: str, provider: str = "gemini") -> str:
    """
    Sends deliveries txt to the chosen LLM provider.
    provider: "gemini" | "groq"
    - Retries on transient 429 rate limits (waits the suggested delay)
    - Falls back through model chain when quota is exhausted
    Returns path to saved report, or None on failure.
    """
    print(f"\n  Step 3: Sending to Gemini for analysis...", flush=True)

    api_key = load_api_key(provider)
    if not api_key:
        env_var = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
        print(f"  ✗ No API key found for {provider}.", flush=True)
        print(f"    Set {env_var} in your environment or .env file.", flush=True)
        return None

    if provider == "groq":
        return _step3_groq(txt_path, video_path, api_key)

    # Gemini path
    try:
        from google import genai
    except ImportError:
        print("  ✗ google-genai not installed. Run: pip install google-genai",
              flush=True)
        return None

    bowling_data = open(txt_path, encoding="utf-8").read()
    if not bowling_data.strip():
        print("  ✗ Deliveries text file is empty", flush=True)
        return None

    user_prompt = (
        "Here is the biomechanical bowling data from pose estimation analysis.\n"
        "Please provide your observations and recommendations.\n\n"
        + bowling_data
    )

    client      = genai.Client(api_key=api_key)
    report      = None
    model_used  = None

    for model in GEMINI_MODEL_CHAIN:
        print(f"  Trying {model}...", flush=True)

        for attempt in range(MAX_RETRIES):
            try:
                report     = _call_gemini_model(client, model, user_prompt)
                model_used = model
                print(f"  Response from {model}", flush=True)
                break   

            except Exception as e:
                if not is_quota_error(e):
                    print(f"  ✗ Non-quota error: {e}", flush=True)
                    return None

                if is_daily_quota_exhausted(e):
                    
                    print(f"   Daily quota exhausted for {model} — "
                          f"trying next model in chain...", flush=True)
                    break   

                delay = parse_retry_delay(e)
                if attempt < MAX_RETRIES - 1:
                    print(f"   Rate limited — waiting {delay}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})...", flush=True)
                    time.sleep(delay)
                else:
                    print(f"   Still rate limited after {MAX_RETRIES} attempts "
                          f"on {model} — trying next model...", flush=True)

        if report:
            break   

    if not report:
        print("\n  ✗ All models in fallback chain exhausted.", flush=True)
        print("    Your options:", flush=True)
        print("      1. Wait until tomorrow — free tier resets daily", flush=True)
        print("      2. Run with --gemini-only tomorrow to skip re-processing",
              flush=True)
        print("      3. Add billing to https://ai.dev to increase quotas", flush=True)
        return None

    basename   = os.path.splitext(os.path.basename(video_path))[0]
    out_report = os.path.join(REPORTS_DIR, f"{basename}_report.txt")
    out_md     = os.path.join(REPORTS_DIR, f"{basename}_report.md")

    report_header = "\n".join([
        "=" * 60,
        "GEMINI BOWLING ANALYSIS REPORT",
        f"Video:     {os.path.basename(video_path)}",
        f"Model:     {model_used}",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        "",
    ])

    full_report = report_header + report
    open(out_report, "w", encoding="utf-8").write(full_report)
    open(out_md,     "w", encoding="utf-8").write(full_report)

    size = os.path.getsize(out_report) / 1024
    print(f"  Report saved: {out_report} ({size:.0f} KB)", flush=True)
    print(f"\n{'─'*60}", flush=True)
    print(report, flush=True)
    print(f"{'─'*60}", flush=True)

    return out_report


# ── Main pipeline ──────────────────────────────────────────────────────────────

def analyze(video_path: str, save_video: bool = True,
            llm_provider: str = None) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  {os.path.basename(video_path)}", flush=True)
    print(f"{'='*60}", flush=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Not found: {video_path}")

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    basename = os.path.splitext(os.path.basename(video_path))[0]
    cap.release()

    out_w = RESIZE_WIDTH or fw
    out_h = int(fh * (out_w / fw))
    before_frames = int(fps * BEFORE_SEC)
    after_frames  = int(fps * AFTER_SEC)

    print(f"\n  {fw}x{fh}  {fps:.1f}fps  {total} frames  "
          f"{total/fps/60:.1f}min", flush=True)

    # ── Pass 1 ─────────────────────────────────────────────────────────────────
    estimator   = PoseEstimator()
    calibration = pass1(video_path, fps, out_w, out_h, estimator)
    targets     = calibration["delivery_frame_targets"]

    if not targets:
        print("  ⚠️  No deliveries detected in pass 1.", flush=True)
        estimator.close()
        return {"video": os.path.basename(video_path), "deliveries": []}

    estimator.reset_tracker()

    # ── Pass 2 ─────────────────────────────────────────────────────────────────
    print(f"\n  Pass 2: Full analysis — collecting all frames...", flush=True)

    cap        = cv2.VideoCapture(video_path)
    writer     = None
    if save_video:
        out_vid = os.path.join(ANNOTATED_VIDEO_DIR, f"{basename}_analyzed.mp4")
        writer  = cv2.VideoWriter(
            out_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    all_records = []
    frame_idx   = 0
    processed   = 0
    detected    = 0
    start_time  = time.time()

    pbar = tqdm(total=total, desc="  Pass 2", unit="fr")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx - 1) % FRAME_SKIP != 0:
            pbar.update(1)
            continue

        processed += 1
        ts_ms   = (frame_idx / fps) * 1000
        frame_r = resize(frame, out_w, out_h)
        pf      = estimator.process_frame(frame_r, frame_idx, ts_ms)

        if pf.detected:
            detected += 1
            rel = frame_reliability(pf)
            if rel > LANDMARK_VIS_FLOOR:
                all_records.append(lean_record(pf, frame_idx, ts_ms, rel))

        if writer:
            writer.write(estimator.draw_landmarks(frame_r, pf))

        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    estimator.close()

    elapsed = time.time() - start_time
    det_pct = round(100 * detected / processed, 1) if processed else 0
    print(f"  Processed: {processed} frames in {elapsed:.1f}s  "
          f"Detection: {det_pct}%", flush=True)

    print(f"\n  Slicing {len(targets)} delivery windows...", flush=True)

    deliveries   = []
    delivery_num = 0

    for rf in targets:
        window = [r for r in all_records
                  if rf - before_frames <= r["f"] <= rf + after_frames]
        if len(window) < MIN_FRAMES_DELIVERY:
            continue

        window = find_phases(window, rf, fps)
        delivery_num += 1
        s = summarize_delivery(window, rf, delivery_num, fps)
        if s:
            deliveries.append(s)
            print(f"  → Delivery {delivery_num} at t={s['t_sec']}s  "
                  f"confidence={s['confidence']}", flush=True)

    high = sum(1 for d in deliveries if d["confidence"] == "high")
    med  = sum(1 for d in deliveries if d["confidence"] == "medium")
    low  = sum(1 for d in deliveries if d["confidence"] == "low")

    print(f"\n  Total: {len(deliveries)}  "
          f"({high} high / {med} medium / {low} low)", flush=True)

    out_json = os.path.join(RESULTS_DIR, f"{basename}_analysis.json")
    out_txt  = os.path.join(RESULTS_DIR, f"{basename}_analysis.txt")

    output = {
        "video":               os.path.basename(video_path),
        "fps":                 round(fps, 2),
        "duration_min":        round(total/fps/60, 2),
        "detection_rate_pct":  det_pct,
        "total_deliveries":    len(deliveries),
        "confidence_breakdown": {"high": high, "medium": med, "low": low},
        "calibration":         calibration,
        "generated_at":        datetime.now().isoformat(),
        "deliveries":          deliveries,
    }

    json.dump(output, open(out_json, "w"), indent=2)

    txt_lines = [
        "=" * 60,
        "CRICKET BOWLING ANALYSIS",
        f"Video:      {os.path.basename(video_path)}",
        f"Duration:   {output['duration_min']} min",
        f"Detection:  {det_pct}%",
        f"Deliveries: {len(deliveries)}  "
        f"({high} high / {med} medium / {low} low confidence)",
        f"Threshold:  {calibration['threshold']:.4f} (percentile-based, auto)",
        f"Generated:  {output['generated_at']}",
        "=" * 60,
    ]
    for s in deliveries:
        txt_lines.append(delivery_to_text(s))

    open(out_txt, "w", encoding="utf-8").write("\n".join(txt_lines))

    size_json = os.path.getsize(out_json) / 1024
    size_txt  = os.path.getsize(out_txt)  / 1024
    print(f"\n  {out_json} ({size_json:.0f} KB)", flush=True)
    print(f"  {out_txt}  ({size_txt:.0f} KB)", flush=True)

    if llm_provider and out_txt:
        step3_llm_analysis(out_txt, video_path, provider=llm_provider)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-pass cricket bowling analysis pipeline")
    parser.add_argument("--video",    required=True,
                        help="Path to video file")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip saving annotated video")
    parser.add_argument("--llm",      choices=["gemini", "groq"], default=None,
                        help="Run LLM coaching analysis after detection. "
                             "Choices: gemini, groq")
    parser.add_argument("--llm-only", action="store_true",
                        help="Skip pose processing — run LLM on existing "
                             "outputs/results/<name>_analysis.txt only. "
                             "Use with --llm groq or --llm gemini.")
    args = parser.parse_args()

    if args.llm_only:
        if not args.llm:
            print(" --llm-only requires --llm groq or --llm gemini")
            import sys; sys.exit(1)
        basename = os.path.splitext(os.path.basename(args.video))[0]
        txt_path = os.path.join(RESULTS_DIR, f"{basename}_analysis.txt")
        if not os.path.exists(txt_path):
            print(f" No analysis txt found: {txt_path}")
            print(f"   Run without --llm-only first to generate it.")
            import sys; sys.exit(1)
        print(f"\n  Using existing: {txt_path}", flush=True)
        step3_llm_analysis(txt_path, args.video, provider=args.llm)
        import sys; sys.exit(0)

    if args.llm:
        key = load_api_key(args.llm)
        if not key:
            env_var = "GEMINI_API_KEY" if args.llm == "gemini" else "GROQ_API_KEY"
            print(f"\n --llm {args.llm} set but {env_var} not found.")
            print(f"   Add it to your .env file:  {env_var}=your_key_here\n")

    result = analyze(args.video,
                     save_video=not args.no_video,
                     llm_provider=args.llm)

    print(f"\n{chr(9472)*55}")
    print(f"{'#':>3}  {'Time':>7}  {'Conf':>8}  {'R.Elbow':>8}  "
          f"{'R.Shoulder':>10}  {'Trunk':>6}")
    print(chr(9472) * 55)
    for s in result.get("deliveries", []):
        ra = s["release_angles"]
        print(f"{s['delivery']:>3}  {s['t_sec']:>6.1f}s  "
              f"{s['confidence']:>8}  "
              f"{ra.get('right_elbow',0):>7.1f}°  "
              f"{ra.get('right_shoulder',0):>9.1f}°  "
              f"{ra.get('trunk_lean',0):>5.1f}°")
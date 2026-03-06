
import os
import sys
import json
import glob
import time
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR     = os.path.join(BASE_DIR, "data", "videos")
POSE_DIR      = os.path.join(BASE_DIR, "outputs", "pose_json")
SUMMARIES_DIR = os.path.join(BASE_DIR, "outputs", "summaries")
REPORTS_DIR   = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(SUMMARIES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR,   exist_ok=True)


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

def find_videos(video_path: str = None, all_videos: bool = False) -> list:
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        return [video_path]
    if all_videos:
        videos = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            videos += glob.glob(os.path.join(VIDEO_DIR, "**", ext), recursive=True)
        return sorted(videos)
    return []


def header(text: str):
    print(f"\n{'='*60}", flush=True)
    print(f"  {text}", flush=True)
    print(f"{'='*60}", flush=True)


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
    txt_path = os.path.join(SUMMARIES_DIR, f"{basename}_deliveries.txt")
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


# ── Step 1: Video Processor ────────────────────────────────────────────────────

def step1_process_video(video_path: str, save_video: bool,
                        skip_existing: bool) -> str:
    basename  = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(POSE_DIR, f"{basename}_pose.json")

    if skip_existing and os.path.exists(json_path):
        size_kb = os.path.getsize(json_path) / 1024
        print(f"\n  Step 1: Pose JSON exists ({size_kb:.0f} KB) — skipping",
              flush=True)
        return json_path

    print(f"\n  Step 1: Processing video → pose JSON...", flush=True)

    from video_processor import process_video
    result = process_video(
        video_path,
        save_annotated=save_video,
        save_json=True,
        show_preview=False,
    )

    if not result.get("json_path"):
        raise RuntimeError("video_processor returned no JSON path")

    return result["json_path"]


# ── Step 2: Delivery Detector ──────────────────────────────────────────────────

def step2_detect_deliveries(json_path: str) -> str:
    """Returns path to deliveries txt file."""
    print(f"\n  Step 2: Detecting deliveries...", flush=True)

    from detector import find_deliveries, assign_phases, summarize_delivery, to_text

    data   = json.load(open(json_path, encoding="utf-8"))
    frames = data.get("frames", [])
    fps    = data.get("fps", 25.0)
    video  = data.get("video", os.path.basename(json_path))

    print(f"  {len(frames)} reliable frames  fps={fps}", flush=True)

    if not frames:
        print("  ⚠️  No frames in JSON", flush=True)
        return None

    deliveries = find_deliveries(frames, fps)
    if not deliveries:
        print("  ⚠️  No deliveries found", flush=True)
        return None

    summaries = []
    for d in deliveries:
        d = assign_phases(d)
        s = summarize_delivery(d)
        if s:
            summaries.append(s)
            print(f"  → Delivery {s['delivery']} at t={s['t_sec']}s  "
                  f"rel={s['release_rel']}", flush=True)

    print(f"\n  Detected: {len(deliveries)}  After filter: {len(summaries)}",
          flush=True)

    basename = os.path.splitext(os.path.basename(json_path))[0].replace("_pose", "")
    out_json = os.path.join(SUMMARIES_DIR, f"{basename}_deliveries.json")
    out_txt  = os.path.join(SUMMARIES_DIR, f"{basename}_deliveries.txt")

    json.dump({
        "video":            video,
        "fps":              round(fps, 2),
        "total_deliveries": len(summaries),
        "generated_at":     datetime.now().isoformat(),
        "deliveries":       summaries,
    }, open(out_json, "w"), indent=2)

    txt_lines = [
        "=" * 60,
        "CRICKET BOWLING ANALYSIS",
        f"Video:      {video}",
        f"Deliveries: {len(summaries)}",
        f"Generated:  {datetime.now().isoformat()}",
        "=" * 60,
    ]
    for s in summaries:
        txt_lines.append(to_text(s))
    open(out_txt, "w", encoding="utf-8").write("\n".join(txt_lines))

    size_j = os.path.getsize(out_json) / 1024
    size_t = os.path.getsize(out_txt)  / 1024
    print(f"  {out_json} ({size_j:.0f} KB)", flush=True)
    print(f"  {out_txt}  ({size_t:.0f} KB)", flush=True)

    print(f"\n  {'#':>3}  {'Time':>7}  {'Rel':>5}  {'R.Elbow':>8}  "
          f"{'R.Shoulder':>10}  {'Trunk':>6}")
    print("  " + "─"*50)
    for s in summaries:
        ra = s["release_angles"]
        print(f"  {s['delivery']:>3}  {s['t_sec']:>6.1f}s  "
              f"{s['release_rel']:>5.3f}  "
              f"{ra.get('right_elbow', 0):>7.1f}°  "
              f"{ra.get('right_shoulder', 0):>9.1f}°  "
              f"{ra.get('trunk_lean', 0):>5.1f}°")

    return out_txt


# ── Step 3: Gemini Analysis ────────────────────────────────────────────────────

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


# ── Groq provider ──────────────────────────────────────────────────────────────

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
                print(f"  ✅ Response from {model}", flush=True)
                break

            except Exception as e:
                if not is_quota_error(e):
                    print(f"  ✗ Non-quota error: {e}", flush=True)
                    return None

                if is_daily_quota_exhausted(e):
                    print(f"  ⚠️  Daily quota exhausted for {model} — "
                          f"trying next model...", flush=True)
                    break

                delay = parse_retry_delay(e)
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️  Rate limited — waiting {delay}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES})...", flush=True)
                    time.sleep(delay)
                else:
                    print(f"  ⚠️  Still rate limited after {MAX_RETRIES} attempts "
                          f"on {model} — trying next...", flush=True)

        if report:
            break

    if not report:
        print("  ✗ All Groq models exhausted.", flush=True)
        return None

    # Save report
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
    print(f"\n  Step 3: Sending to {provider.upper()} for analysis...", flush=True)

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
                print(f"  ✅ Response from {model}", flush=True)
                break   # success — exit retry loop

            except Exception as e:
                if not is_quota_error(e):
                    # Non-quota error (auth, network, etc.) — fail immediately
                    print(f"  ✗ Non-quota error: {e}", flush=True)
                    return None

                if is_daily_quota_exhausted(e):
                    # Daily limit hit — no point retrying this model
                    print(f"  ⚠️  Daily quota exhausted for {model} — "
                          f"trying next model in chain...", flush=True)
                    break   # exit retry loop, try next model

                # Transient rate limit — wait and retry same model
                delay = parse_retry_delay(e)
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️  Rate limited — waiting {delay}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})...", flush=True)
                    time.sleep(delay)
                else:
                    print(f"  ⚠️  Still rate limited after {MAX_RETRIES} attempts "
                          f"on {model} — trying next model...", flush=True)

        if report:
            break   # exit model fallback loop

    if not report:
        print("\n  ✗ All models in fallback chain exhausted.", flush=True)
        print("    Your options:", flush=True)
        print("      1. Wait until tomorrow — free tier resets daily", flush=True)
        print("      2. Run with --gemini-only tomorrow to skip re-processing",
              flush=True)
        print("      3. Add billing to https://ai.dev to increase quotas", flush=True)
        return None

    # ── Save report ────────────────────────────────────────────────────────────
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


# ── Run single video ───────────────────────────────────────────────────────────

def run_video(video_path: str, save_video: bool, skip_existing: bool,
              use_llm: bool, llm_only: bool, provider: str = "gemini"):
    header(os.path.basename(video_path))
    start    = time.time()
    txt_path = None

    if llm_only:
        txt_path = get_deliveries_txt(video_path)
        if not txt_path:
            print(f"  ✗ No deliveries txt found — run without --gemini-only first",
                  flush=True)
            return
        print(f"  Using existing: {txt_path}", flush=True)

    else:
        try:
            json_path = step1_process_video(video_path, save_video, skip_existing)
        except Exception as e:
            print(f"  ✗ Step 1 failed: {e}", flush=True)
            return

        try:
            txt_path = step2_detect_deliveries(json_path)
        except Exception as e:
            print(f"  ✗ Step 2 failed: {e}", flush=True)
            return

    if (use_llm or llm_only) and txt_path:
        try:
            step3_llm_analysis(txt_path, video_path, provider=provider)
        except Exception as e:
            print(f"  ✗ Step 3 failed: {e}", flush=True)

    elapsed = time.time() - start
    print(f"\n  ✅ Done in {elapsed/60:.1f} min", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cricket bowling pipeline: VideoProcessor → Detector → Gemini")

    parser.add_argument("--video",        help="Path to a single video file")
    parser.add_argument("--all",          action="store_true",
                        help=f"Process all videos in {VIDEO_DIR}")
    parser.add_argument("--no-video",     action="store_true",
                        help="Skip saving annotated video (faster)")
    parser.add_argument("--skip-existing",action="store_true",
                        help="Skip Step 1 if pose JSON already exists")
    parser.add_argument("--llm",          choices=["gemini", "groq"],
                        help="Run LLM analysis after detection (Step 3). "
                             "Choices: gemini, groq")
    parser.add_argument("--llm-only",     action="store_true",
                        help="Skip Steps 1 & 2 — run LLM on existing deliveries txt "
                             "(use with --llm gemini or --llm groq)")

    args = parser.parse_args()

    if not args.video and not args.all:
        parser.print_help()
        sys.exit(1)

    videos = find_videos(args.video, args.all)
    if not videos:
        print(f"No videos found. Use --video <path> or put videos in {VIDEO_DIR}")
        sys.exit(1)

    print(f"\nFound {len(videos)} video(s)", flush=True)

    if args.llm or args.llm_only:
        prov = args.llm or "gemini"
        if not load_api_key(prov):
            env_var = "GEMINI_API_KEY" if prov == "gemini" else "GROQ_API_KEY"
            print(f"\n⚠️  --llm {prov} set but no API key found.")
            print(f"   Create a .env file:  {env_var}=your_key_here\n")

    total_start = time.time()
    provider = args.llm or "gemini"   # default to gemini if --llm-only used alone

    for v in videos:
        run_video(
            v,
            save_video    = not args.no_video,
            skip_existing = args.skip_existing,
            use_llm       = bool(args.llm),
            llm_only      = args.llm_only,
            provider      = provider,
        )

    print(f"\n{'='*60}", flush=True)
    print(f"  All done in {(time.time()-total_start)/60:.1f} minutes", flush=True)
# Bowling throw Analysis

MediaPipe-based biomechanical analysis of bowling. Detects deliveries, extracts joint angles per phase, and generates LLM coaching reports.

---

## Setup

```bash
# 1. Create env
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 2. Install
pip install -r requirements.txt
pip install groq                # for Groq LLM
pip install google-genai        # for Gemini LLM

# 3. API keys — create .env in project root
GROQ_API_KEY=your_key          # console.groq.com  (free: 14,400 req/day)
GEMINI_API_KEY=your_key        # aistudio.google.com (free tier)

# 4. Add videos to data/videos/
```

---

## The Two Pipelines

### `analyze_pipeline.py` — Two-pass

**Pass 1** warms up `BowlerTracker` (background subtraction) on every 2nd frame, collects the wrist signal, then runs `find_peaks` on the full signal in batch to locate all deliveries upfront. Threshold is percentile-based (`p50 + 0.5*(p95-p50)`) — robust to noisy non-delivery frames.

**Pass 2** runs full pose estimation with the now-warm tracker, collects all frames into memory, then slices windows around the delivery timestamps from Pass 1.

Output → `outputs/results/<n>_analysis.json` + `_analysis.txt`

---

### `two_pass_pipeline.py` — Two-step

`video_processor.py` → `detector.py`

Saves a lean pose JSON after Step 1. Use `--skip-existing` to reuse it and re-run detection without re-running MediaPipe.

> `summarizer.py` is a separate standalone script — run it directly on any pose JSON:
> `python summarizer.py --json outputs/pose_json/<n>_pose.json --txt`

Output → `outputs/summaries/<n>_deliveries.json` + `_deliveries.txt`

---

## Commands

### `analyze_pipeline.py`

```bash
python analyze_pipeline.py --video "data/videos/bumrah.mp4"   (Example video path)
python analyze_pipeline.py --video "data/videos/bumrah.mp4" --no-video
python analyze_pipeline.py --video "data/videos/bumrah.mp4" --llm groq
python analyze_pipeline.py --video "data/videos/bumrah.mp4" --llm gemini

# Skip re-processing, run LLM on existing txt (use when quota resets)
python analyze_pipeline.py --video "data/videos/archive/videos/steyln.mp4" --llm-only --llm groq
python analyze_pipeline.py --video "data/videos/archive/videos/steyln.mp4" --llm-only --llm gemini
```

### `two_pass_pipeline.py`

```bash
python two_pass_pipeline.py --video "data/videos/bumrah.mp4"  (Example video path)
python two_pass_pipeline.py --video "data/videos/bumrah.mp4" --no-video
python two_pass_pipeline.py --video "data/videos/bumrah.mp4" --llm groq
python two_pass_pipeline.py --all
python two_pass_pipeline.py --all --no-video --skip-existing --llm groq

# LLM only on existing summaries
python two_pass_pipeline.py --video "data/videos/bumrah.mp4" --llm-only --llm groq
```

---

## Outputs

`outputs/results/` — JSON and TXT from `analyze_pipeline.py`

`outputs/summaries/` — JSON and TXT from `two_pass_pipeline.py`

`outputs/reports/` — LLM coaching report from either pipeline

`outputs/annotated_videos/` — skeleton overlay video

`outputs/pose_json/` — raw lean frames (two_pass_pipeline.py intermediate)

---

## LLM Providers

Both pipelines use automatic model fallback — if a model hits quota, the next is tried.

| Provider | Free Tier       | Models (in order)                                                   |
| -------- | --------------- | ------------------------------------------------------------------- |
| Groq     | 14,400 req/day  | `llama-3.3-70b-versatile` → `llama-3.1-8b-instant` → `mixtral-8x7b` |
| Gemini   | Daily per model | `gemini-2.0-flash-lite` → `gemini-2.0-flash` → `gemini-1.5-flash`   |

If all models are exhausted, use `--llm-only` the next day.

# BowlingBiomech

MediaPipe-based bowling biomechanics analyzer - detects deliveries, extracts joint angles per phase, and generates LLM coaching reports.

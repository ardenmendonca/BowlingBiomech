"""
visualizer.py - Plot joint angle timelines, phase breakdowns, and comparison charts

Run standalone:
    python visualizer.py --json outputs/pose_json/my_video_pose.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import PLOTS_DIR

# Colors per phase
PHASE_COLORS = {
    "run_up": "#4A90D9",
    "load": "#F5A623",
    "delivery_stride": "#7ED321",
    "release": "#D0021B",
    "follow_through": "#9B59B6",
    "unknown": "#AAAAAA",
    "no_detection": "#DDDDDD",
}

KEY_ANGLES = [
    "right_elbow",
    "right_shoulder",
    "right_knee",
    "left_knee",
    "trunk_lean",
]


def load_json(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def plot_angle_timelines(data: dict, save: bool = True, show: bool = True):
    """
    Plot joint angle values over time, colour-coded by bowling phase.
    """
    frames = data.get("frames", [])
    if not frames:
        print("No frame data found.")
        return

    frame_ids = [f["frame_idx"] for f in frames]
    phases = [f.get("phase", "unknown") for f in frames]

    fig, axes = plt.subplots(len(KEY_ANGLES), 1, figsize=(14, 3 * len(KEY_ANGLES)),
                             sharex=True)
    fig.suptitle(f"Joint Angle Timeline\n{data.get('video', '')}", fontsize=13, y=1.01)

    for ax, angle_name in zip(axes, KEY_ANGLES):
        values = [f.get("joint_angles", {}).get(angle_name, None) for f in frames]

        # Plot phase background bands
        prev_phase = None
        band_start = 0
        for i, phase in enumerate(phases):
            if phase != prev_phase or i == len(phases) - 1:
                if prev_phase is not None:
                    ax.axvspan(frame_ids[band_start], frame_ids[i - 1],
                               alpha=0.15, color=PHASE_COLORS.get(prev_phase, "#CCCCCC"))
                band_start = i
                prev_phase = phase

        # Plot angle curve
        valid_x = [frame_ids[i] for i, v in enumerate(values) if v and v > 0]
        valid_y = [v for v in values if v and v > 0]

        if valid_x:
            ax.plot(valid_x, valid_y, color="#222222", linewidth=1.5, label=angle_name)
            ax.set_ylabel(f"{angle_name}\n(degrees)", fontsize=9)
            ax.set_ylim(0, 200)
            ax.axhline(90, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.axhline(180, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Frame Index")

    # Legend for phases
    legend_patches = [
        mpatches.Patch(color=color, alpha=0.4, label=phase)
        for phase, color in PHASE_COLORS.items()
        if phase not in ("no_detection",)
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8,
               title="Phase", bbox_to_anchor=(1.12, 0.98))

    plt.tight_layout()

    if save:
        basename = os.path.splitext(os.path.basename(data.get("video", "output")))[0]
        path = os.path.join(PLOTS_DIR, f"{basename}_angles.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"📊  Saved angle timeline: {path}")

    if show:
        plt.show()

    plt.close()


def plot_phase_summary(data: dict, save: bool = True, show: bool = True):
    """
    Bar chart showing average angles per phase for key joints.
    """
    phase_stats = data.get("metrics", {}).get("phase_angle_stats", {})
    if not phase_stats:
        print("No phase stats found.")
        return

    phases = list(phase_stats.keys())
    angles_to_plot = ["right_elbow", "right_shoulder", "right_knee", "left_knee"]

    x = np.arange(len(phases))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, angle_name in enumerate(angles_to_plot):
        means = [phase_stats[p].get(angle_name, {}).get("mean", 0) for p in phases]
        stds = [phase_stats[p].get(angle_name, {}).get("std", 0) for p in phases]
        bars = ax.bar(x + i * width, means, width, label=angle_name,
                      yerr=stds, capsize=3, alpha=0.8)

    ax.set_xticks(x + width * (len(angles_to_plot) - 1) / 2)
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(f"Mean Joint Angles by Bowling Phase\n{data.get('video', '')}")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="90°")
    ax.set_ylim(0, 200)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        basename = os.path.splitext(os.path.basename(data.get("video", "output")))[0]
        path = os.path.join(PLOTS_DIR, f"{basename}_phase_summary.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"📊  Saved phase summary: {path}")

    if show:
        plt.show()

    plt.close()


def plot_findings_report(data: dict, save: bool = True, show: bool = True):
    """
    Text-based findings summary as a matplotlib figure (easy to screenshot / share).
    """
    findings = data.get("findings", [])
    metrics = data.get("metrics", {})
    summary = metrics.get("summary", {})

    fig, ax = plt.subplots(figsize=(10, max(4, len(findings) * 1.2 + 3)))
    ax.axis("off")

    text_lines = [
        f"Cricket Bowling Analysis: {data.get('video', '')}",
        f"Frames processed: {summary.get('detected_frames', '?')} / "
        f"{summary.get('total_frames', '?')}  "
        f"(detection rate: {summary.get('detection_rate_pct', '?')}%)",
        "",
        "BIOMECHANICAL FINDINGS",
        "─" * 60,
    ]

    for finding in findings:
        icon = {"ok": "✅", "info": "ℹ️", "warning": "⚠️"}.get(finding["severity"], "•")
        text_lines.append(f"{icon}  [{finding['category'].upper()}]  {finding['finding']}")

    release_angles = data.get("metrics", {}).get("release_frame", {}).get("joint_angles", {})
    if release_angles:
        text_lines += ["", "ANGLES AT RELEASE FRAME", "─" * 60]
        for k, v in release_angles.items():
            text_lines.append(f"  {k:<25} {v:.1f}°")

    ax.text(0.02, 0.98, "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F8F8F8", alpha=0.8))

    plt.tight_layout()

    if save:
        basename = os.path.splitext(os.path.basename(data.get("video", "output")))[0]
        path = os.path.join(PLOTS_DIR, f"{basename}_findings.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"📊  Saved findings report: {path}")

    if show:
        plt.show()

    plt.close()


def visualize_from_json(json_path: str, show: bool = True):
    data = load_json(json_path)
    plot_angle_timelines(data, save=True, show=show)
    plot_phase_summary(data, save=True, show=show)
    plot_findings_report(data, save=True, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pose analysis results")
    parser.add_argument("--json", required=True, help="Path to pose JSON file")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots without displaying them")
    args = parser.parse_args()

    visualize_from_json(args.json, show=not args.no_show)

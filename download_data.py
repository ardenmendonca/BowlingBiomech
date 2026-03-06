"""
download_data.py - Download cricket datasets from Kaggle

Usage:
    python download_data.py --dataset bowl_release
    python download_data.py --dataset cricket_videos
    python download_data.py --all

Requirements:
    - Kaggle API token at ~/.kaggle/kaggle.json
    - Get it from: https://www.kaggle.com/settings > API > Create New Token
"""

import os
import argparse
import zipfile
import shutil
from config import KAGGLE_DATASETS, VIDEO_DIR, DATA_DIR


def check_kaggle_credentials():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print(" Kaggle credentials not found!")
        print(f"   Expected: {kaggle_json}")
        print()
        print("To set up:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. API section → Create New Token")
        print("  3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac)")
        return False
    return True


def download_dataset(dataset_key: str, dest_dir: str = DATA_DIR):
    """Download and extract a Kaggle dataset."""
    if not check_kaggle_credentials():
        return False

    dataset_id = KAGGLE_DATASETS[dataset_key]
    print(f" Downloading: {dataset_id}")

    extract_dir = os.path.join(dest_dir, dataset_key)
    os.makedirs(extract_dir, exist_ok=True)

    # Use kaggle CLI
    cmd = f"kaggle datasets download -d {dataset_id} -p {extract_dir} --unzip"
    ret = os.system(cmd)

    if ret != 0:
        print(f" Download failed for {dataset_id}")
        return False

    print(f"  Downloaded to: {extract_dir}")

    # Move any video files to VIDEO_DIR for easy access
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    moved = 0
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in video_extensions:
                src = os.path.join(root, fname)
                dst = os.path.join(VIDEO_DIR, f"{dataset_key}_{fname}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    moved += 1

    if moved:
        print(f"   Copied {moved} video file(s) to {VIDEO_DIR}")

    return True


def list_videos():
    """List all videos available for processing."""
    print(f"\n Videos in {VIDEO_DIR}:")
    files = [f for f in os.listdir(VIDEO_DIR)
             if os.path.splitext(f)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    if not files:
        print("   (none — run download_data.py first or copy videos manually)")
    for f in sorted(files):
        size_mb = os.path.getsize(os.path.join(VIDEO_DIR, f)) / 1e6
        print(f"   {f}  ({size_mb:.1f} MB)")
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download cricket Kaggle datasets")
    parser.add_argument("--dataset", choices=list(KAGGLE_DATASETS.keys()),
                        help="Which dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List downloaded videos")
    args = parser.parse_args()

    if args.list:
        list_videos()
    elif args.all:
        for key in KAGGLE_DATASETS:
            download_dataset(key)
        list_videos()
    elif args.dataset:
        download_dataset(args.dataset)
        list_videos()
    else:
        parser.print_help()
        print()
        list_videos()

#!/usr/bin/env python3
"""Tạo val_tracks.json với số lượng validation tracks mong muốn.

Usage:
    python scripts/create_val_split.py --val-size 2000
    python scripts/create_val_split.py --val-size 2000 --output data/val_tracks.json
"""
import argparse
import glob
import json
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Tạo val_tracks.json")
    parser.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="Số tracks cho validation (default: 2000)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root data (default: Data/train)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file output (default: Data/val_tracks.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--scenario-b-only",
        action="store_true",
        help="Chỉ lấy val từ Scenario-B",
    )
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(PROJECT_ROOT, "Data", "train")
    # Mặc định dùng Data/val_tracks.json để khớp config.VAL_SPLIT_FILE
    output_path = args.output or os.path.join(PROJECT_ROOT, "Data", "val_tracks.json")

    if not os.path.isdir(data_root):
        print(f"❌ Data root không tồn tại: {data_root}")
        return 1

    search_path = os.path.join(os.path.abspath(data_root), "**", "track_*")
    all_tracks = sorted(glob.glob(search_path, recursive=True))

    if not all_tracks:
        print(f"❌ Không tìm thấy track nào trong {data_root}")
        return 1

    # Lọc theo Scenario-B nếu cần
    if args.scenario_b_only:
        candidates = [t for t in all_tracks if "Scenario-B" in t]
        if not candidates:
            print("⚠️ Không có Scenario-B, dùng tất cả tracks")
            candidates = all_tracks
    else:
        candidates = all_tracks

    random.Random(args.seed).shuffle(candidates)
    val_size = min(args.val_size, len(candidates))
    val_tracks = candidates[:val_size]
    val_ids = [os.path.basename(t) for t in val_tracks]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)

    print(f"✅ Đã tạo {output_path}")
    print(f"   Val tracks: {len(val_ids)}")
    print(f"   Tổng tracks có sẵn: {len(candidates)}")
    if args.scenario_b_only:
        print(f"   (Chỉ từ Scenario-B)")
    return 0


if __name__ == "__main__":
    exit(main())

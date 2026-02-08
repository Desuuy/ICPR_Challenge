#!/usr/bin/env python3
"""Kiểm tra: khi đánh giá trên val_tracks.json, model có lấy được label không.

Truy vết luồng: val_tracks.json -> track paths -> annotations.json -> label
"""
import json
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Paths (khớp config)
    val_split_file = os.path.join(PROJECT_ROOT, "Data", "val_tracks.json")
    if not os.path.exists(val_split_file):
        val_split_file = os.path.join(PROJECT_ROOT, "data", "val_tracks.json")
    data_root = os.path.join(PROJECT_ROOT, "Data", "train")
    if not os.path.isdir(data_root):
        data_root = os.path.join(PROJECT_ROOT, "data", "train")

    print("=" * 60)
    print("KIEM TRA: Val co lay duoc label de danh gia khong?")
    print("=" * 60)
    print(f"VAL_SPLIT_FILE: {val_split_file}")
    print(f"DATA_ROOT:      {data_root}")
    print()

    # 1. Load val_tracks.json
    if not os.path.exists(val_split_file):
        print(f"❌ Không tìm thấy: {val_split_file}")
        return 1
    with open(val_split_file, "r") as f:
        val_ids = json.load(f)
    print(f"val_tracks.json: {len(val_ids)} track IDs")

    # 2. Scan all tracks trong DATA_ROOT
    search_path = os.path.join(os.path.abspath(data_root), "**", "track_*")
    all_tracks = sorted(glob.glob(search_path, recursive=True))
    track_by_basename = {os.path.basename(t): t for t in all_tracks}
    print(f"Tong tracks trong DATA_ROOT: {len(all_tracks)}")

    # 3. Match val_ids với track paths
    val_tracks = []
    missing = []
    for vid in val_ids:
        if vid in track_by_basename:
            val_tracks.append(track_by_basename[vid])
        else:
            missing.append(vid)

    if missing:
        print(f"[!] Val IDs khong tim thay track folder: {len(missing)}")
        if len(missing) <= 5:
            print(f"   Ví dụ: {missing}")
        else:
            print(f"   Ví dụ: {missing[:5]}...")
    print(f"[OK] Val tracks match: {len(val_tracks)}/{len(val_ids)}")
    print()

    # 4. Đọc annotations.json cho từng val track
    have_label = 0
    no_label = 0
    no_annot = 0
    no_lr = 0
    samples = []

    for track_path in val_tracks[:100]:  # Check 100 đầu
        json_path = os.path.join(track_path, "annotations.json")
        if not os.path.exists(json_path):
            no_annot += 1
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = data[0]
            label = data.get("plate_text", data.get("license_plate", data.get("text", "")))
            if not label:
                no_label += 1
                continue
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png"))
                + glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            if not lr_files:
                no_lr += 1
                continue
            have_label += 1
            if len(samples) < 5:
                samples.append((os.path.basename(track_path), label, len(lr_files)))
        except Exception as e:
            no_annot += 1

    # Full count (không giới hạn 100)
    for track_path in val_tracks[100:]:
        json_path = os.path.join(track_path, "annotations.json")
        if not os.path.exists(json_path):
            no_annot += 1
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = data[0]
            label = data.get("plate_text", data.get("license_plate", data.get("text", "")))
            if not label:
                no_label += 1
                continue
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png"))
                + glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            if not lr_files:
                no_lr += 1
                continue
            have_label += 1
        except Exception:
            no_annot += 1

    print("Ket qua doc label tu annotations.json:")
    print(f"   [OK] Co label + anh LR: {have_label}")
    print(f"   [X] Khong co annotations.json: {no_annot}")
    print(f"   [X] Label rong: {no_label}")
    print(f"   [X] Khong co lr-*.jpg/png: {no_lr}")
    print()
    if samples:
        print("Mau (track_id, label, num_frames):")
        for tid, lbl, nf in samples:
            print(f"   {tid}: label='{lbl}', frames={nf}")
    print()

    if have_label == len(val_tracks):
        print("[OK] KET LUAN: Model CO LAY DUOC label de danh gia val.")
    elif have_label > 0:
        print(f"[!] KET LUAN: Chi {have_label}/{len(val_tracks)} tracks co label. Kiem tra annotations.json.")
    else:
        print("[X] KET LUAN: Khong co track nao co label. Val se khong danh gia dung.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

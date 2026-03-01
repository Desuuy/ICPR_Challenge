import os
import json
import random
import shutil
from collections import defaultdict

"""
Tạo danh sách ảnh + nhãn cho OpenOCR, tách rõ:
    - 80% track cho train
    - 20% track cho valid
và KHÔNG trùng lặp track giữa 2 bộ.

Kết quả:
    OUT_DIR/
        train_images/  (ảnh copy cho train)
        val_images/    (ảnh copy cho val)
        train_labels.txt  (relative_path \t label)
        val_labels.txt    (relative_path \t label)
"""

# Thư mục gốc chứa data MultiFrame-LPR (train)
ROOT = r"C:\Users\anhhu\MultiFrame-LPR\Data\train"

# Thư mục output cho dữ liệu chuẩn bị OpenOCR
OUT_DIR = r"C:\Users\anhhu\lmdb_for_openocr"
TRAIN_IMG_DIR = os.path.join(OUT_DIR, "train_images")
VAL_IMG_DIR = os.path.join(OUT_DIR, "val_images")
TRAIN_LABEL_FILE = os.path.join(OUT_DIR, "train_labels.txt")
VAL_LABEL_FILE = os.path.join(OUT_DIR, "val_labels.txt")

# Tỉ lệ chia train/val theo track
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)


def collect_tracks(root_dir: str):
    """
    Dò toàn bộ thư mục, tìm mỗi track có annotations.json,
    gom lại:
        tracks[track_id] = {
            "label": plate_text,
            "files": [(abs_img_path, new_flat_name), ...]
        }
    """
    tracks = {}

    for root, _, files in os.walk(root_dir):
        if "annotations.json" not in files:
            continue

        jpath = os.path.join(root, "annotations.json")
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                data = data[0]

        label = data.get("plate_text") or data.get(
            "license_plate") or data.get("text")
        if not label:
            continue

        track_id = os.path.basename(root)
        file_list = []

        for fname in files:
            if not fname.lower().endswith(".png"):
                continue
            if not (fname.startswith("hr-") or fname.startswith("lr-")):
                continue

            ipath = os.path.join(root, fname)
            if not os.path.exists(ipath):
                continue

            # Đặt tên flattend: trackId_fname để tránh trùng
            new_name = f"{track_id}_{fname}"
            file_list.append((ipath, new_name))

        if file_list:
            tracks[track_id] = {
                "label": label,
                "files": file_list,
            }

    return tracks


def split_tracks(tracks_dict, train_ratio: float, seed: int = 42):
    """
    Chia danh sách track_id thành train/val theo tỉ lệ train_ratio,
    không trùng lặp.
    """
    track_ids = sorted(tracks_dict.keys())
    random.Random(seed).shuffle(track_ids)

    n = len(track_ids)
    if n == 0:
        return [], []

    split_idx = int(n * train_ratio)
    # Đảm bảo cả train và val đều không rỗng nếu có >= 2 track
    if split_idx <= 0 and n > 1:
        split_idx = 1
    if split_idx >= n and n > 1:
        split_idx = n - 1

    train_ids = track_ids[:split_idx]
    val_ids = track_ids[split_idx:]
    return train_ids, val_ids


def export_split(tracks_dict, track_ids, img_dir, label_path):
    """
    Copy ảnh của các track trong track_ids vào img_dir,
    và ghi file label dạng: relative_path \t label
    """
    lines = []
    for tid in track_ids:
        info = tracks_dict[tid]
        label = info["label"]
        for src_path, new_name in info["files"]:
            dst_path = os.path.join(img_dir, new_name)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            rel_path = os.path.relpath(dst_path, img_dir)
            lines.append(f"{rel_path}\t{label}\n")

    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return len(lines)


def main():
    print(f"Scanning tracks under: {ROOT}")
    tracks = collect_tracks(ROOT)
    print(f"Found {len(tracks)} tracks with labels")

    train_ids, val_ids = split_tracks(tracks, TRAIN_RATIO, RANDOM_SEED)
    print(f"Train tracks: {len(train_ids)}, Val tracks: {len(val_ids)}")

    n_train = export_split(tracks, train_ids, TRAIN_IMG_DIR, TRAIN_LABEL_FILE)
    n_val = export_split(tracks, val_ids, VAL_IMG_DIR, VAL_LABEL_FILE)

    print(f"Saved {n_train} samples to {TRAIN_LABEL_FILE}")
    print(f"Saved {n_val} samples to {VAL_LABEL_FILE}")


if __name__ == "__main__":
    main()

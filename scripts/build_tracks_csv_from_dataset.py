#!/usr/bin/env python3
"""
Build a tracks.csv file from the existing ICPR-style dataset structure.

Dataset layout (ví dụ):
    data/train/Scenario-A/Brazilian/track_00001/lr-001.png
                                                  lr-002.png
                                                  ...

Script này sẽ:
    - Quét tất cả folder `track_*` dưới một root (train hoặc public_test).
    - Với mỗi track:
        * Lấy danh sách ảnh LR (lr-*.png/jpg), đã sort sẵn.
        * Gán frame_id = 1..N theo thứ tự.
        * Ghi ra CSV:
              track_id,frame_id,image_path
              track_00001,1,data/train/Scenario-A/Brazilian/track_00001/lr-001.png
              ...

Bạn có thể dùng CSV này cho:
    - `prepare_openocr_input.py` → tạo list ảnh cho OpenOCR.
    - `make_submission.py` → fusion theo track_id.

Đường dẫn: truyền khi chạy (--root-dir, --output, --rel-to).
Hoặc sửa 3 hằng số dưới đây rồi chạy không tham số: python scripts/build_tracks_csv_from_dataset.py
"""

from __future__ import annotations

# --- Sửa 3 dòng này nếu muốn mặc định, rồi chạy script không cần truyền tham số ---
DEFAULT_ROOT_DIR = r"C:\Users\anhhu\MultiFrame-LPR\data\blind_test"
DEFAULT_OUTPUT_CSV = r"C:\Users\anhhu\MultiFrame-LPR\data\tracks.csv"
DEFAULT_REL_TO = r"C:\Users\anhhu\MultiFrame-LPR"

import argparse
import csv
import glob
import os
from typing import List


def find_track_dirs(root_dir: str) -> List[str]:
    """
    Tìm tất cả thư mục `track_*` đệ quy dưới root_dir.
    """
    abs_root = os.path.abspath(root_dir)
    pattern = os.path.join(abs_root, "**", "track_*")
    return sorted(
        d for d in glob.glob(pattern, recursive=True) if os.path.isdir(d)
    )


def build_tracks_csv(
    root_dir: str,
    output_csv: str,
    rel_to: str | None = None,
) -> None:
    """
    Quét dataset và ghi tracks.csv.

    Args:
        root_dir: Thư mục gốc chứa các `track_*` (ví dụ: data/train hoặc data/public_test).
        output_csv: Đường dẫn file CSV output.
        rel_to: Nếu set, image_path trong CSV sẽ là đường dẫn tương đối so với `rel_to`.
                Nếu None, sẽ dùng đường dẫn tương đối với CWD hiện tại.
    """
    abs_root = os.path.abspath(root_dir)
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(f"Root dir not found: {abs_root}")

    if rel_to is None:
        rel_to = os.getcwd()
    rel_to = os.path.abspath(rel_to)

    track_dirs = find_track_dirs(abs_root)
    if not track_dirs:
        print(f"⚠️ No track_* folders found under: {abs_root}")
        return

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    num_rows = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "frame_id", "image_path"])

        for track_path in track_dirs:
            track_id = os.path.basename(track_path)

            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png"))
                + glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            if not lr_files:
                continue

            for idx, img_path in enumerate(lr_files, start=1):
                # Đường dẫn ảnh tương đối so với rel_to để dễ dùng trong các script khác.
                rel_img_path = os.path.relpath(
                    os.path.abspath(img_path), start=rel_to
                )
                writer.writerow([track_id, idx, rel_img_path])
                num_rows += 1

    print(f"📂 Scanned root: {abs_root}")
    print(f"📁 Found {len(track_dirs)} track folders")
    print(f"📝 Wrote {num_rows} rows to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tracks.csv from ICPR-style MultiFrame-LPR dataset."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=DEFAULT_ROOT_DIR,
        help="Root directory containing track_* folders (default: DEFAULT_ROOT_DIR in script).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (default: DEFAULT_OUTPUT_CSV in script).",
    )
    parser.add_argument(
        "--rel-to",
        type=str,
        default=DEFAULT_REL_TO,
        help="Base dir for relative image_path in CSV (default: DEFAULT_REL_TO in script).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_tracks_csv(
        root_dir=args.root_dir,
        output_csv=args.output,
        rel_to=args.rel_to,
    )


if __name__ == "__main__":
    main()


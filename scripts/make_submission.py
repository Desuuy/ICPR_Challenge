#!/usr/bin/env python3
"""
Fusion script: combine OpenOCR per-image predictions into per-track predictions.

Expected inputs:
1) tracks.csv  (output từ pipeline detection + tracking + crop)
   - Format (header bắt buộc):
       track_id,frame_id,image_path
     Ví dụ:
       track_00001,000001,data/crops/track_00001_f000001.jpg
       track_00001,000002,data/crops/track_00001_f000002.jpg
       track_00002,000010,data/crops/track_00002_f000010.jpg

2) openocr_results.txt  (output inference từ OpenOCR)
   - Mặc định script này giả định mỗi dòng:
       image_path<TAB>text<TAB>confidence
     Ví dụ:
       data/crops/track_00001_f000001.jpg\tABC1234\t0.94
       data/crops/track_00001_f000002.jpg\tABC1234\t0.97
       data/crops/track_00002_f000010.jpg\tXYZ5678\t0.88

3) Output:
   - predictions.txt đúng format nộp bài:
       track_00001,ABC1234;0.9876
       track_00002,XYZ5678;0.9123

Nếu format file OpenOCR khác (ví dụ JSON), bạn chỉ cần chỉnh lại hàm
`load_openocr_results`.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def _norm_path(path: str, base_dir: str | None = None) -> str:
    """
    Chuẩn hoá đường dẫn để so khớp giữa tracks.csv và kết quả OpenOCR.

    - Nếu path là absolute: dùng luôn.
    - Nếu path là relative: join với base_dir (hoặc CWD).
    - Sau đó áp dụng abspath + normpath + normcase.
    """
    if base_dir is None:
        base_dir = os.getcwd()
    path = path.strip()
    if not path:
        return path
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def load_tracks(tracks_csv: str) -> Dict[str, List[str]]:
    """
    Đọc tracks.csv -> map track_id -> list image_path (giữ nguyên thứ tự frame_id nếu có).
    """
    if not os.path.exists(tracks_csv):
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv}")

    tracks: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    base_dir = os.path.dirname(os.path.abspath(tracks_csv))

    with open(tracks_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"track_id", "frame_id", "image_path"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"tracks.csv must contain columns: {required_cols}, "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            track_id = row["track_id"]
            frame_id_str = row["frame_id"]
            raw_image_path = row["image_path"]
            image_path = _norm_path(raw_image_path, base_dir=base_dir)

            try:
                frame_id = int(frame_id_str)
            except ValueError:
                # Nếu frame_id là string không phải số, vẫn giữ lại nhưng sort theo string
                # (ít quan trọng vì fusion không phụ thuộc thứ tự).
                try:
                    frame_id = int(frame_id_str.lstrip("0") or "0")
                except Exception:
                    frame_id = 0

            tracks[track_id].append((frame_id, image_path))

    # Sort theo frame_id để giữ thứ tự thời gian
    ordered_tracks: Dict[str, List[str]] = {}
    for tid, items in tracks.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        ordered_tracks[tid] = [p for _, p in items_sorted]

    return ordered_tracks


def load_openocr_results(results_path: str) -> Dict[str, Tuple[str, float]]:
    """
    Đọc output của OpenOCR:
        image_path<TAB>text<TAB>confidence

    Trả về: map image_path -> (text, confidence)
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"OpenOCR results file not found: {results_path}")

    mapping: Dict[str, Tuple[str, float]] = {}

    base_dir = os.path.dirname(os.path.abspath(results_path))

    with open(results_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid format at line {line_num} in {results_path}: {line!r}"
                )

            raw_image_path = parts[0]
            image_path = _norm_path(raw_image_path, base_dir=base_dir)
            text = parts[1]
            conf = 1.0

            if len(parts) >= 3:
                try:
                    conf = float(parts[2])
                except ValueError:
                    # Nếu confidence không parse được, giữ mặc định 1.0
                    conf = 1.0

            mapping[image_path] = (text, conf)

    return mapping


def fuse_track_predictions(
    image_paths: List[str],
    image_pred_map: Dict[str, Tuple[str, float]],
    min_conf: float = 0.0,
) -> Tuple[str, float]:
    """
    Multi-frame fusion cho 1 track:
    - Gom tất cả (text, conf) của các frame thuộc track.
    - Đếm tần suất mỗi text, tính tổng/mean confidence.
    - Chọn text:
        1) frequency cao nhất,
        2) nếu tie -> text có tổng confidence cao hơn.

    Trả về: (best_text, fused_confidence)
        fused_confidence = mean confidence của best_text trên các frame của nó.
    """
    candidates: List[Tuple[str, float]] = []
    for img in image_paths:
        if img not in image_pred_map:
            # Nếu thiếu kết quả cho 1 frame, bỏ qua frame đó
            continue
        text, conf = image_pred_map[img]
        if conf < min_conf:
            # Bỏ các frame quá tự tin thấp nếu muốn
            continue
        candidates.append((text, conf))

    if not candidates:
        # Không có frame nào hợp lệ -> trả về chuỗi rỗng với conf 0
        return "", 0.0

    # Đếm frequency của mỗi text
    freq = Counter(t for t, _ in candidates)

    # Tính tổng và mean confidence cho mỗi text
    sum_conf: Dict[str, float] = defaultdict(float)
    count_conf: Dict[str, int] = defaultdict(int)
    for t, c in candidates:
        sum_conf[t] += c
        count_conf[t] += 1

    # Chọn text tốt nhất: ưu tiên frequency, sau đó tổng confidence
    best_text = None
    best_key = None  # (freq, sum_conf)
    for t in freq.keys():
        key = (freq[t], sum_conf[t])
        if best_key is None or key > best_key:
            best_key = key
            best_text = t

    if best_text is None:
        return "", 0.0

    fused_conf = sum_conf[best_text] / max(count_conf[best_text], 1)
    return best_text, fused_conf


def make_submission(
    tracks_csv: str,
    openocr_results: str,
    output_path: str,
    min_conf: float = 0.0,
) -> None:
    """
    Glue chính:
    - Đọc tracks.csv (track_id -> list image_path).
    - Đọc kết quả OpenOCR (image_path -> (text, conf)).
    - Multi-frame fusion cho từng track.
    - Ghi predictions.txt: track_id,plate;confidence
    """
    print(f"📂 Loading tracks from: {tracks_csv}")
    tracks = load_tracks(tracks_csv)
    print(f"   -> {len(tracks)} tracks")

    print(f"📂 Loading OpenOCR results from: {openocr_results}")
    image_pred_map = load_openocr_results(openocr_results)
    print(f"   -> {len(image_pred_map)} image predictions")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    num_empty = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for track_id, image_paths in tracks.items():
            text, fused_conf = fuse_track_predictions(
                image_paths=image_paths,
                image_pred_map=image_pred_map,
                min_conf=min_conf,
            )

            if not text:
                num_empty += 1

            # Định dạng: track_id,PLATE;CONF
            # CONF giữ 4 chữ số thập phân
            line = f"{track_id},{text};{fused_conf:.4f}\n"
            f.write(line)

    print(f"✅ Saved submission to: {output_path}")
    print(f"   Tracks without any valid prediction: {num_empty}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse OpenOCR per-image predictions into per-track predictions."
    )
    parser.add_argument(
        "--tracks-csv",
        type=str,
        required=True,
        help="Path to tracks.csv (track_id,frame_id,image_path).",
    )
    parser.add_argument(
        "--openocr-results",
        type=str,
        required=True,
        help="Path to OpenOCR results file (image_path<TAB>text<TAB>confidence).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.txt",
        help="Output predictions file path (default: predictions.txt).",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Optional: minimum confidence per frame to keep in fusion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_submission(
        tracks_csv=args.tracks_csv,
        openocr_results=args.openocr_results,
        output_path=args.output,
        min_conf=args.min_conf,
    )


if __name__ == "__main__":
    main()


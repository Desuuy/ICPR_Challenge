#!/usr/bin/env python3
"""
Utility script to convert `tracks.csv` into an image list file for OpenOCR.

Goal:
- You already have detection+tracking+crop pipeline that writes `tracks.csv`:
    track_id,frame_id,image_path
- This script reads that CSV and produces a flat list of image paths that
  OpenOCR can use for inference (e.g., via a `Global.infer_list`-style option).

Usage example:
    python scripts/prepare_openocr_input.py \
        --tracks-csv data/tracks.csv \
        --output-list data/openocr_infer_list.txt \
        --images-root .

The output list will contain one image path per line. If `--images-root` is
provided and a path in `tracks.csv` is relative, it will be joined with
`images-root`. Absolute paths are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable, Set


def _normalize_path(path: str, images_root: str | None) -> str:
    """
    Make image path consistent with where OpenOCR will run.

    - If `path` is absolute -> return as-is.
    - If `path` is relative and `images_root` is given -> join(root, path).
    - Otherwise -> return path unchanged.
    """
    path = path.strip()
    if not path:
        return path

    if os.path.isabs(path):
        return path

    if images_root:
        return os.path.normpath(os.path.join(images_root, path))

    return os.path.normpath(path)


def build_image_list(
    tracks_csv: str,
    output_list: str,
    images_root: str | None = None,
    unique: bool = True,
) -> None:
    """
    Read tracks.csv and write a flat list of image paths for OpenOCR.

    Args:
        tracks_csv: Path to CSV with header `track_id,frame_id,image_path`.
        output_list: Destination text file; each line is an image path.
        images_root: Optional root to prepend to relative paths.
        unique: If True, deduplicate image paths.
    """
    if not os.path.exists(tracks_csv):
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv}")

    os.makedirs(os.path.dirname(output_list) or ".", exist_ok=True)

    seen: Set[str] = set()

    with open(tracks_csv, "r", encoding="utf-8") as f_in, open(
        output_list, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        required_cols = {"track_id", "frame_id", "image_path"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"tracks.csv must contain columns: {required_cols}, "
                f"got {reader.fieldnames}"
            )

        num_rows = 0
        num_written = 0

        for row in reader:
            num_rows += 1
            raw_path = row["image_path"]
            img_path = _normalize_path(raw_path, images_root)
            if not img_path:
                continue

            if unique:
                if img_path in seen:
                    continue
                seen.add(img_path)

            f_out.write(img_path + "\n")
            num_written += 1

    print(f"📂 Loaded {num_rows} rows from {tracks_csv}")
    print(f"📝 Wrote {num_written} image paths to {output_list}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenOCR image list from tracks.csv."
    )
    parser.add_argument(
        "--tracks-csv",
        type=str,
        required=True,
        help="Path to tracks.csv (with track_id,frame_id,image_path).",
    )
    parser.add_argument(
        "--output-list",
        type=str,
        required=True,
        help="Output text file path; each line is an image path for OpenOCR.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help=(
            "Optional base directory to prepend to relative image paths from tracks.csv. "
            "Absolute paths are left unchanged."
        ),
    )
    parser.add_argument(
        "--no-unique",
        action="store_true",
        help="Disable deduplication; keep duplicate image paths if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_image_list(
        tracks_csv=args.tracks_csv,
        output_list=args.output_list,
        images_root=args.images_root,
        unique=not args.no_unique,
    )


if __name__ == "__main__":
    main()


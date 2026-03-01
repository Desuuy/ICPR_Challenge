#!/usr/bin/env python3
"""
Chạy pipeline Inference + nộp bài (các bước trong MultiFrame-LPR).

Bước 1: Tạo tracks.csv từ dataset (build_tracks_csv_from_dataset).
Bước 2: Tạo list ảnh cho OpenOCR (prepare_openocr_input).
Bước 3: Bạn chạy OpenOCR inference bên ngoài → openocr_results.txt.
Bước 4: Fusion (make_submission) → predictions.txt.

Cách dùng:
    python scripts/run_inference_pipeline.py --root-dir Data/public_test
    # Sau khi có openocr_results.txt:
    python scripts/run_inference_pipeline.py --root-dir Data/public_test --openocr-results results/openocr_results.txt --fusion-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str], cwd: str | None = None) -> None:
    cwd = cwd or os.getcwd()
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference pipeline (tracks.csv -> OpenOCR -> predictions.txt)")
    parser.add_argument("--root-dir", type=str, default="Data/public_test", help="Root chứa track_* folders")
    parser.add_argument("--repo", type=str, default=None, help="Đường dẫn repo MultiFrame-LPR (mặc định: CWD)")
    parser.add_argument("--tracks-csv", type=str, default="data/tracks.csv", help="Output tracks.csv")
    parser.add_argument("--openocr-list", type=str, default="data/openocr_infer_list.txt", help="Output list ảnh OpenOCR")
    parser.add_argument("--openocr-results", type=str, default="results/openocr_results.txt", help="File kết quả OpenOCR (input cho fusion)")
    parser.add_argument("--output", type=str, default="predictions.txt", help="File predictions output")
    parser.add_argument("--fusion-only", action="store_true", help="Chỉ chạy bước 4 (fusion), cần --tracks-csv và --openocr-results đã có sẵn")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo or os.getcwd())
    root_dir = os.path.abspath(args.root_dir) if not os.path.isabs(args.root_dir) else args.root_dir
    tracks_csv = args.tracks_csv if os.path.isabs(args.tracks_csv) else os.path.join(repo, args.tracks_csv)
    openocr_list = args.openocr_list if os.path.isabs(args.openocr_list) else os.path.join(repo, args.openocr_list)
    openocr_results = args.openocr_results if os.path.isabs(args.openocr_results) else os.path.join(repo, args.openocr_results)
    output = args.output if os.path.isabs(args.output) else os.path.join(repo, args.output)

    os.makedirs(os.path.dirname(tracks_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(openocr_list) or ".", exist_ok=True)

    if not args.fusion_only:
        # --- Bước 1: tracks.csv ---
        print("\n=== Bước 1: Tạo tracks.csv ===")
        run([
            sys.executable,
            os.path.join(repo, "scripts", "build_tracks_csv_from_dataset.py"),
            "--root-dir", root_dir,
            "--output", tracks_csv,
            "--rel-to", repo,
        ], cwd=repo)

        # --- Bước 2: list ảnh OpenOCR ---
        print("\n=== Bước 2: Tạo list ảnh cho OpenOCR ===")
        run([
            sys.executable,
            os.path.join(repo, "scripts", "prepare_openocr_input.py"),
            "--tracks-csv", tracks_csv,
            "--output-list", openocr_list,
            "--images-root", repo,
        ], cwd=repo)

        # --- Bước 3: nhắc chạy OpenOCR ---
        print("\n=== Bước 3: Chạy OpenOCR inference (bên ngoài repo OpenOCR) ===")
        print("  Ví dụ:")
        print('  cd C:\\Users\\anhhu\\OpenOCR')
        print("  python tools/infer_rec.py -c configs/rec/lp_svtrv2_gtc.yml \\")
        print(f'    -o Global.infer_img="..." Global.save_res_path="{openocr_results}"')
        print(f"  Sau khi có file: {openocr_results}")
        print("  Format mỗi dòng: image_path\\ttext\\tconfidence\n")

        if not os.path.isfile(openocr_results):
            print("  Chưa thấy openocr_results.txt. Chạy OpenOCR xong rồi chạy fusion:")
            print(f"  python scripts/run_inference_pipeline.py --fusion-only --tracks-csv {tracks_csv} --openocr-results {openocr_results} --output {output}")
            return

    # --- Bước 4: Fusion ---
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    print("\n=== Bước 4: Fusion -> predictions.txt ===")
    run([
        sys.executable,
        os.path.join(repo, "scripts", "make_submission.py"),
        "--tracks-csv", tracks_csv,
        "--openocr-results", openocr_results,
        "--output", output,
        "--min-conf", "0.0",
    ], cwd=repo)

    print(f"\n✅ Xong. File nộp bài: {output}")


if __name__ == "__main__":
    main()

"""
test_and_analyze.py

Script đánh giá model MultiFrameSVTRv2 trên val set với:
- Decode CTC bằng decode_with_confidence
- Lưu CSV: ground_truth, prediction, confidence, track_id
- Thống kê Exact Match Accuracy và các cặp ký tự hay nhầm lẫn
"""

import os
from collections import Counter
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.mf_svtrv2 import MultiFrameSVTRv2
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


def build_val_loader(config: Config) -> DataLoader | None:
    """Khởi tạo DataLoader cho val set, dùng cùng Dataset pipeline như train_optimized."""
    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        augmentation_level=config.AUGMENTATION_LEVEL,
        same_aug_per_sample=getattr(config, "SAME_AUG_PER_SAMPLE", True),
        use_msr=getattr(config, "USE_MSR", False),
        msr_width_min=getattr(config, "MSR_WIDTH_MIN", 64),
        msr_width_max=getattr(config, "MSR_WIDTH_MAX", 128),
    )

    if len(val_ds) == 0:
        print("⚠️ Validation dataset rỗng, không thể test.")
        return None

    loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn,
    )
    return loader


def build_model(config: Config) -> MultiFrameSVTRv2:
    """Tạo model MultiFrameSVTRv2 và load checkpoint tốt nhất nếu có."""
    model = MultiFrameSVTRv2(
        num_classes=config.NUM_CLASSES,
        use_stn=config.USE_STN,
        dropout=config.DROPOUT,
        use_temp_scaling=config.USE_TEMP_SCALING,
    ).to(config.DEVICE)

    exp_name = config.EXPERIMENT_NAME
    best_ckpt = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")

    if os.path.exists(best_ckpt):
        print(f"📦 Loading best checkpoint: {best_ckpt}")
        state = torch.load(best_ckpt, map_location=config.DEVICE)
        model.load_state_dict(state)
    else:
        print(f"⚠️ Không tìm thấy {best_ckpt}, dùng PRETRAINED_PATH: {config.PRETRAINED_PATH}")
        if os.path.exists(config.PRETRAINED_PATH):
            state = torch.load(config.PRETRAINED_PATH, map_location=config.DEVICE)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        else:
            print("❌ Không tìm thấy PRETRAINED_PATH, model sẽ chạy với random weights.")

    model.eval()
    return model


def test_and_analyze_ctc(
    model: MultiFrameSVTRv2,
    loader: DataLoader,
    config: Config,
    save_csv_path: str = "plate_recognition_analysis.csv",
) -> pd.DataFrame:
    """
    Đánh giá model CTC (MultiFrameSVTRv2) trên loader:
    - Decode bằng decode_with_confidence
    - Lưu CSV
    - In Exact Match Accuracy + thống kê lỗi ký tự
    """
    device = config.DEVICE
    idx2char = config.IDX2CHAR
    beam_width = getattr(config, "CTC_BEAM_WIDTH", 1)

    model.eval()
    all_rows: List[dict] = []
    confusion_counter: Counter = Counter()

    exact_match = 0
    total_samples = 0

    ds = loader.dataset
    print(f"\n🚀 Bắt đầu đánh giá chi tiết trên {len(ds)} mẫu (val set)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            (
                images,
                _targets,
                _tlen,
                labels_text,
                track_ids,
                img_paths_batch,
                country_ids,
            ) = batch

            images = images.to(device)  # [B, T, C, H, W]
            country_ids = country_ids.to(device)

            logits = model(images, country_ids)  # [B, W', C]
            log_probs = logits.log_softmax(2)

            decoded_list = decode_with_confidence(
                log_probs, idx2char, beam_width=beam_width
            )

            for i, (pred_text, conf) in enumerate(decoded_list):
                gt_text = labels_text[i]
                track_id = track_ids[i]

                all_rows.append(
                    {
                        "track_id": track_id,
                        "ground_truth": gt_text,
                        "prediction": pred_text,
                        "confidence": conf,
                    }
                )

                total_samples += 1
                if gt_text == pred_text:
                    exact_match += 1

                max_len = max(len(gt_text), len(pred_text))
                for pos in range(max_len):
                    char_gt = gt_text[pos] if pos < len(gt_text) else "[EMPTY]"
                    char_pred = pred_text[pos] if pos < len(pred_text) else "[EMPTY]"
                    if char_gt != char_pred:
                        confusion_counter[(char_gt, char_pred, pos)] += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu kết quả chi tiết vào: {save_csv_path}")

    acc = (exact_match / total_samples) * 100 if total_samples > 0 else 0.0
    print("\n" + "=" * 50)
    print("🎯 TỔNG KẾT ĐÁNH GIÁ (OVERALL ACCURACY)")
    print("=" * 50)
    print(f"   • Tổng số mẫu:          {total_samples}")
    print(f"   • Số mẫu đoán đúng 100%: {exact_match}")
    print(f"   • Exact Match Accuracy:  {acc:.2f}%")

    print("\n" + "=" * 50)
    print("📊 THỐNG KÊ CÁC CẶP KÝ TỰ HAY NHẦM LẪN NHẤT")
    print(f"{'GT':<8} | {'Dự đoán':<8} | {'Vị trí':<7} | {'Số lần'}")
    print("-" * 45)

    for (char_gt, char_pred, pos), count in confusion_counter.most_common(20):
        gt_disp = char_gt if char_gt != " " else "[SPACE]"
        pred_disp = char_pred if char_pred != " " else "[SPACE]"
        print(f"{gt_disp:<8} | {pred_disp:<8} | {pos:<7} | {count}")

    pos_counts = Counter([k[2] for k in confusion_counter.keys()])
    common_pos = pos_counts.most_common(3)
    if common_pos:
        print(
            "💡 Gợi ý post-processing: Các vị trí hay sai nhất là index: "
            + ", ".join(str(p[0]) for p in common_pos)
        )

    print("=" * 50)
    return df


def main():
    config = Config()
    seed_everything(config.SEED)

    print(f"🚀 TEST & ANALYZE | Device: {config.DEVICE}")

    val_loader = build_val_loader(config)
    if val_loader is None:
        return

    model = build_model(config)

    _ = test_and_analyze_ctc(
        model=model,
        loader=val_loader,
        config=config,
        save_csv_path="plate_recognition_analysis.csv",
    )


if __name__ == "__main__":
    main()
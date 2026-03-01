"""Quick end-to-end sanity check for the current CTC + MSR MultiFrame pipeline.

Chạy script này để kiểm tra:
1) Dataset + collate_fn: shapes, target/target_lengths có hợp lệ không.
2) Model MultiFrameSVTRv2: forward OK, không lỗi shape.
3) CTC loss: tính được loss hữu hạn (không NaN/Inf).
4) Decode: in ra vài (GT, Pred, confidence) để xem model có đang "nói chuyện"
   cùng vocab và độ dài label đúng chưa.

Lệnh chạy gợi ý:
    python debug_pipeline_ctc.py

Script này KHÔNG train, chỉ lấy 1–2 batch để test logic.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from configs.config import get_default_config
from src.data.dataset import MultiFrameDataset
from src.mf_svtrv2 import MultiFrameSVTRv2
from src.utils.postprocess import decode_with_confidence


@torch.no_grad()
def run_single_batch_check(
    config,
    loader: DataLoader,
    model: nn.Module,
) -> None:
    """Lấy 1 batch từ loader, chạy qua model + CTC loss + decode để debug."""

    device = config.DEVICE
    model.eval()

    batch = next(iter(loader))
    (
        images,          # [B, T, C, H, W]
        targets,         # concat targets, 1D LongTensor
        target_lengths,  # [B]
        labels_text,     # tuple[str]
        track_ids,       # tuple[str]
        img_paths_batch,  # tuple[tuple[str]]
        country_ids,     # [B]
    ) = batch

    print("=== BATCH SHAPES ===")
    print(f"images:         {tuple(images.shape)}")
    print(
        f"targets:        {tuple(targets.shape)} (sum(target_lengths)={int(target_lengths.sum())})")
    print(f"target_lengths: {tuple(target_lengths.shape)}")
    print(f"country_ids:    {tuple(country_ids.shape)}")
    print(f"examples labels: {list(labels_text)[:3]}")
    print("====================\n")

    images = images.to(device)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)
    country_ids = country_ids.to(device)

    # Forward model
    logits = model(images, country_ids=country_ids)  # [B, T, C]
    print(f"model logits shape: {tuple(logits.shape)}")

    # Chuẩn hoá cho CTC: log_softmax trên dim lớp
    logits = logits.log_softmax(2)

    # lengths theo thời gian T (seq_len)
    batch_size, seq_len, num_classes = logits.size()
    input_lengths = torch.full(
        size=(batch_size,),
        fill_value=seq_len,
        dtype=torch.long,
        device=logits.device,
    )

    # CTC loss tiêu chuẩn để debug (không dính focal/GTC)
    ctc = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )

    loss = ctc(
        logits.permute(1, 0, 2),  # [T, B, C]
        targets,
        input_lengths,
        target_lengths,
    )

    print(f"CTC loss (debug): {float(loss):.4f}")
    if not torch.isfinite(loss):
        print("⚠️ Loss không hữu hạn (NaN/Inf) – cần xem lại pipeline.")

    # Decode để xem mô hình đang dự đoán gì (dù là random hoặc pretrained)
    decoded = decode_with_confidence(
        logits, config.IDX2CHAR, beam_width=config.CTC_BEAM_WIDTH)

    print("\n=== VÍ DỤ PREDICTIONS (tối đa 5 mẫu) ===")
    for i, (pred_text, conf) in enumerate(decoded[:5]):
        gt = labels_text[i]
        tid = track_ids[i]
        print(f"[{i}] track_id={tid}")
        print(f"    GT : {gt}")
        print(f"    Pred: {pred_text} (conf={conf:.4f})")
        print("---")
    print("====================\n")


def main():
    config = get_default_config()

    # Đảm bảo dùng batch nhỏ + num_workers=0 cho debug nhanh, tránh lỗi dataloader.
    debug_batch_size = min(4, config.BATCH_SIZE)
    debug_num_workers = 0

    print("=== DEBUG CTC PIPELINE ===")
    print(f"DATA_ROOT : {config.DATA_ROOT}")
    print(f"DEVICE    : {config.DEVICE}")
    print(f"BATCH_SIZE: {debug_batch_size}")
    print(f"WORKERS   : {debug_num_workers}")
    print("==========================\n")

    if not os.path.exists(config.DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT không tồn tại: {config.DATA_ROOT}")

    # Dataset train (dùng cùng config với train_optimized, nhưng lấy subset nhỏ để debug)
    train_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode="train",
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        augmentation_level=config.AUGMENTATION_LEVEL,
        is_test=False,
        full_train=False,
        same_aug_per_sample=getattr(config, "SAME_AUG_PER_SAMPLE", True),
        sr_enhancer=None,
        use_msr=getattr(config, "USE_MSR", False),
        msr_width_min=getattr(config, "MSR_WIDTH_MIN", 64),
        msr_width_max=getattr(config, "MSR_WIDTH_MAX", 256),
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "Dataset train rỗng – kiểm tra DATA_ROOT / cấu trúc track_*.")

    # Lấy tối đa 8 sample đầu để debug nhanh
    subset_indices = list(range(min(8, len(train_ds))))
    debug_ds = Subset(train_ds, subset_indices)

    train_loader = DataLoader(
        debug_ds,
        batch_size=debug_batch_size,
        shuffle=False,
        num_workers=debug_num_workers,
        pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn,
    )

    # Khởi tạo model MultiFrameSVTRv2 giống train_optimized
    model = MultiFrameSVTRv2(
        num_classes=config.NUM_CLASSES,
        use_stn=config.USE_STN,
        dropout=getattr(config, "DROPOUT", 0.0),
    ).to(config.DEVICE)

    # Nếu có checkpoint đã train trước đó, load vào để debug chất lượng sau finetune.
    # Đường dẫn gợi ý: weights/mf_svtrv2_best.pth (như bạn nêu).
    ckpt_candidates = [
        os.path.join(os.path.dirname(__file__),
                     "weights", "mf_svtrv2_best.pth"),
        os.path.join(os.path.dirname(__file__),
                     "weights", "ck-mf_svtrv2_best.pth"),
    ]
    loaded_ckpt = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            loaded_ckpt = p
            break

    if loaded_ckpt is not None:
        print(f"📦 Loading checkpoint for debug: {loaded_ckpt}")
        state = torch.load(loaded_ckpt, map_location=config.DEVICE)
        # Hỗ trợ cả dạng thuần state_dict hoặc dict có key 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Xử lý riêng pos_embed nếu checkpoint dùng max_sz nhỏ hơn (VD: 32x128 -> 32x256).
        # Trong backbone: backbone.pope.patch_embed.2.pos_embed có shape [1, C, H, W_feat].
        model_state = model.state_dict()
        pe_key = "backbone.pope.patch_embed.2.pos_embed"
        if pe_key in state and pe_key in model_state:
            v = state[pe_key]
            target = model_state[pe_key]
            if v.shape != target.shape:
                print(
                    f"   🔄 Interpolating pos_embed: {tuple(v.shape)} -> {tuple(target.shape)}")
                # Nội suy theo H,W về đúng kích thước backbone hiện tại
                v_interp = F.interpolate(
                    v,
                    size=target.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
                state[pe_key] = v_interp

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"   -> Loaded with strict=False")
        print(f"      Missing keys   : {len(missing)}")
        print(f"      Unexpected keys: {len(unexpected)}")
    else:
        print("ℹ️  Không tìm thấy mf_svtrv2_best.pth, dùng weights ngẫu nhiên cho debug.")

    print("\nModel đã khởi tạo (đã load checkpoint nếu có), chạy batch debug...\n")
    run_single_batch_check(config, train_loader, model)

    print("✅ Debug CTC pipeline hoàn tất (xem log ở trên để đánh giá).")


if __name__ == "__main__":
    main()

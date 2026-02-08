#!/usr/bin/env python3
"""Debug: Tai sao Val Acc = 0%? Kiem tra output model."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.config import Config
from src.mf_svtrv2 import MultiFrameSVTRv2
from src.data.dataset import MultiFrameDataset
from torch.utils.data import DataLoader


def main():
    config = Config()
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for vf in [
        os.path.join(proj, "Data", "val_tracks.json"),
        os.path.join(proj, "data", "val_tracks.json"),
    ]:
        if os.path.exists(vf):
            config.VAL_SPLIT_FILE = vf
            break

    print("=" * 60)
    print("DEBUG: Val Acc = 0% - Kiem tra model output")
    print("=" * 60)

    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='val',
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
    )
    if len(val_ds) == 0:
        print("Val dataset rong!")
        return 1

    loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=MultiFrameDataset.collate_fn)
    images, targets, target_lengths, labels_text, track_ids, _ = next(iter(loader))

    model = MultiFrameSVTRv2(num_classes=config.NUM_CLASSES, use_stn=config.USE_STN).to(config.DEVICE)

    # Load pretrained
    ckpt = os.path.join(proj, "weights", "best.pth")
    if os.path.exists(ckpt):
        model.load_weights(ckpt)
    else:
        print("Khong co best.pth, dung random init")

    model.eval()
    with torch.no_grad():
        preds = model(images.to(config.DEVICE))

    # Check preds
    print("\n1. Model output shape:", preds.shape)
    print("   [batch, time_steps, num_classes]")

    has_nan = torch.isnan(preds).any().item()
    has_inf = torch.isinf(preds).any().item()
    print(f"\n2. Output co NaN: {has_nan}")
    print(f"   Output co Inf: {has_inf}")

    probs = preds.exp()
    max_prob, indices = probs.max(dim=2)
    print(f"\n3. Argmax indices (time step 0): {indices[0, :8].tolist()}")
    print(f"   So luong blank (0): {(indices == 0).sum().item()} / {indices.numel()}")

    # Decode thủ công
    from src.utils.postprocess import decode_with_confidence
    decoded = decode_with_confidence(preds, config.IDX2CHAR)
    print(f"\n4. Du doan vs Ground truth (4 mau dau):")
    for i in range(min(4, len(decoded))):
        pred_str, conf = decoded[i]
        gt = labels_text[i] if i < len(labels_text) else "?"
        print(f"   GT: '{gt}' | Pred: '{pred_str}' | conf: {conf:.4f}")

    if (indices == 0).float().mean() > 0.95:
        print("\n>>> NGUYEN NHAN: Model output toan BLANK (index 0)")
        print("    -> Loss NaN nen khong train duoc, hoac pretrained khong hop")
        print("    -> Thu: python train.py --no-pretrained --epochs 3")
    elif has_nan or has_inf:
        print("\n>>> NGUYEN NHAN: Model output NaN/Inf")
        print("    -> Thu: --no-pretrained, giam LEARNING_RATE")
    else:
        print("\n>>> Model output OK nhung pred sai -> can train them")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

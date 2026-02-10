#!/usr/bin/env python3
"""Test beam search decoding với checkpoint hiện có (79.38%)."""
import torch
import os
import sys
from torch.utils.data import DataLoader
from configs.config import Config
from src.mf_svtrv2 import MultiFrameSVTRv2
from src.data.dataset import MultiFrameDataset
from src.training.trainer import Trainer


def main():
    print("="*70)
    print("🔍 TESTING BEAM SEARCH DECODING")
    print("="*70)
    print("Mục đích: Kiểm tra xem beam search có cải thiện accuracy không")
    print("Không cần train lại - chỉ test với checkpoint hiện có")
    print("="*70 + "\n")

    # ============================================================
    # NHẬP ĐƯỜNG DẪN CHECKPOINT
    # ============================================================
    print("📍 NHẬP ĐƯỜNG DẪN CHECKPOINT:")
    print("-" * 70)

    # Mặc định tìm trong results/
    default_paths = [
        'results/mf_svtrv2_best.pth',
        'results/mf_svtrv2_final.pth',
        'weights/best.pth',
    ]

    print("Các đường dẫn phổ biến:")
    for i, path in enumerate(default_paths, 1):
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {i}. {path} {exists}")

    print("\nVí dụ đường dẫn:")
    print("  - results/mf_svtrv2_best.pth")
    print("  - /workspace/MultiFrame-LPR/results/mf_svtrv2_best.pth")
    print("  - /path/to/your/checkpoint.pth")
    print("-" * 70)

    # Nhập từ command line argument hoặc input
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"📦 Using checkpoint from argument: {checkpoint_path}")
    else:
        checkpoint_path = input("\n👉 Nhập đường dẫn checkpoint (hoặc Enter để dùng default): ").strip()
        
        # Nếu để trống, thử tìm file tồn tại
        if not checkpoint_path:
            for path in default_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    print(f"✅ Auto-detected: {checkpoint_path}")
                    break
            
            if not checkpoint_path:
                print("❌ ERROR: Không tìm thấy checkpoint nào!")
                print("   Vui lòng nhập đường dẫn thủ công.")
                sys.exit(1)

    # Kiểm tra file tồn tại
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ ERROR: File không tồn tại: {checkpoint_path}")
        print("   Kiểm tra lại đường dẫn và thử lại.")
        sys.exit(1)

    print(f"\n✅ Checkpoint path: {checkpoint_path}")
    print(f"   File size: {os.path.getsize(checkpoint_path) / (1024**2):.2f} MB")
    print()

    # ============================================================
    # LOAD DATA & MODEL
    # ============================================================
    config = Config()

    # Load validation dataset
    print("📂 Loading validation dataset...")
    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='val',
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=0,  # Windows multiprocessing issue - dùng 0 cho test script
        pin_memory=True
    )

    print(f"✅ Loaded {len(val_ds)} validation samples\n")

    # Initialize model
    print("🔧 Initializing model...")
    model = MultiFrameSVTRv2(
        num_classes=config.NUM_CLASSES,
        use_stn=False,
        dropout=0.0
    ).to(config.DEVICE)

    # Load checkpoint
    print(f"📦 Loading checkpoint...")
    try:
        state = torch.load(checkpoint_path, map_location=config.DEVICE)
        # Cho phép bỏ qua các tham số mới (vd: temp_scaling.temperature)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"   ℹ️ Missing keys (ignored): {missing}")
        if unexpected:
            print(f"   ℹ️ Unexpected keys (ignored): {unexpected}")
        print("✅ Checkpoint loaded successfully!\n")
    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test different beam widths
    print("="*70)
    print("🧪 TESTING DIFFERENT BEAM WIDTHS")
    print("="*70)
    print(f"{'Beam':>6} | {'Accuracy':>9} | {'CER':>7} | {'Correct':>8} | {'Time Est.'}")
    print("-"*70)

    import time

    results = []
    for beam_width in [1, 3, 5, 10]:
        config.CTC_BEAM_WIDTH = beam_width
        
        # Create trainer (chỉ để validate) - dùng val_loader làm train_loader giả
        trainer = Trainer(
            model=model,
            train_loader=val_loader,  # cần khác None để OneCycleLR/scheduler không lỗi
            val_loader=val_loader,
            config=config,
            idx2char=config.IDX2CHAR
        )
        
        # Validate
        start_time = time.time()
        metrics, _, _ = trainer.validate()
        elapsed = time.time() - start_time
        
        acc = metrics['acc']
        cer = metrics['cer']
        correct = metrics['correct']
        total = metrics['total']
        
        results.append({
            'beam': beam_width,
            'acc': acc,
            'cer': cer,
            'correct': correct,
            'total': total,
            'time': elapsed
        })
        
        print(f"{beam_width:6d} | {acc:8.2f}% | {cer:7.4f} | {correct:4d}/{total:4d} | {elapsed:6.1f}s")

    print("="*70 + "\n")

    # Summary & Recommendation
    print("="*70)
    print("📊 SUMMARY & RECOMMENDATION")
    print("="*70)

    baseline = results[0]  # beam=1
    best = max(results, key=lambda x: x['acc'])

    print(f"\n🎯 Baseline (Beam=1):")
    print(f"   Accuracy: {baseline['acc']:.2f}%")
    print(f"   CER: {baseline['cer']:.4f}")

    print(f"\n✨ Best Result (Beam={best['beam']}):")
    print(f"   Accuracy: {best['acc']:.2f}%")
    print(f"   CER: {best['cer']:.4f}")
    print(f"   Improvement: +{best['acc'] - baseline['acc']:.2f}%")
    print(f"   Inference time: {best['time']:.1f}s (vs {baseline['time']:.1f}s baseline)")

    if best['acc'] - baseline['acc'] > 0.5:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   ✅ Beam={best['beam']} improves accuracy by +{best['acc'] - baseline['acc']:.2f}%!")
        print(f"   ✅ Set CTC_BEAM_WIDTH={best['beam']} in config.py")
        print(f"   ✅ This is FREE improvement without retraining!")
    else:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   ℹ️  Beam search gives minimal improvement (<0.5%)")
        print(f"   ℹ️  Keep CTC_BEAM_WIDTH=1 for faster inference")

    # Next steps
    print(f"\n📋 NEXT STEPS:")
    if best['acc'] < 85:
        print(f"   1. Current best: {best['acc']:.2f}% (need 90% target)")
        print(f"   2. Gap to target: {90 - best['acc']:.2f}%")
        print(f"   3. Recommended: Fine-tune 50 epochs + Add STN")
        print(f"      → Run: python train_optimized.py --epochs 50 --lr 0.0001")
    else:
        print(f"   1. Current: {best['acc']:.2f}% - Very close to 90% target!")
        print(f"   2. Recommended: Light fine-tuning or add STN")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
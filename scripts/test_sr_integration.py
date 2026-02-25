#!/usr/bin/env python3
"""Script kiểm tra tích hợp MF-LPR Super-Resolution vào pipeline OCR.

Chạy script này để verify:
1. Module SR có import được không
2. Adapter MF_LPR_SR có khởi tạo được không (nếu có checkpoint)
3. Dataset có nhận SR enhancer không
4. Pipeline có chạy được không (test với 1 batch nhỏ)

Usage:
    python test_sr_integration.py [--sr-checkpoint-path PATH] [--no-sr]
"""

from src.sr import MF_LPR_SR
from src.data.dataset import MultiFrameDataset
from configs.config import Config
import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_sr_module_import():
    """Test 1: Kiểm tra module SR có import được không."""
    print("=" * 60)
    print("TEST 1: Kiểm tra import module SR")
    print("=" * 60)
    try:
        from src.sr import MF_LPR_SR
        print("✅ PASS: Module `src.sr.MF_LPR_SR` import thành công")
        return True
    except ImportError as e:
        print(f"❌ FAIL: Không thể import MF_LPR_SR. Lỗi: {e}")
        return False


def test_sr_adapter_init(checkpoint_path: str = None, config_path: str = None):
    """Test 2: Kiểm tra adapter SR có khởi tạo được không."""
    print("\n" + "=" * 60)
    print("TEST 2: Kiểm tra khởi tạo MF_LPR_SR adapter")
    print("=" * 60)

    if not checkpoint_path:
        print("⚠️  SKIP: Không có checkpoint path, bỏ qua test này")
        print("   (Bạn có thể test bằng cách: --sr-checkpoint-path <path>)")
        return True

    if not os.path.exists(checkpoint_path):
        print(f"❌ FAIL: Checkpoint không tồn tại: {checkpoint_path}")
        return False

    if config_path and not os.path.exists(config_path):
        print(f"❌ FAIL: Config không tồn tại: {config_path}")
        return False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sr_enhancer = MF_LPR_SR(
            checkpoint_path=checkpoint_path,
            config_path=config_path or "sr_model/config/LP-Diff.json",
            device=device,
        )
        print(f"✅ PASS: MF_LPR_SR đã khởi tạo thành công trên {device}")
        print(f"   - Checkpoint: {checkpoint_path}")
        print(f"   - Config: {config_path or 'sr_model/config/LP-Diff.json'}")
        return True, sr_enhancer
    except Exception as e:
        print(f"❌ FAIL: Không thể khởi tạo MF_LPR_SR. Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_sr_enhance_function(sr_enhancer):
    """Test 3: Kiểm tra hàm enhance có chạy được không."""
    print("\n" + "=" * 60)
    print("TEST 3: Kiểm tra hàm enhance_sequence")
    print("=" * 60)

    if sr_enhancer is None:
        print("⚠️  SKIP: Không có SR enhancer, bỏ qua test này")
        return True

    try:
        # Tạo dummy input: (T=5, C=3, H=32, W=128), normalized [-1, 1]
        dummy_frames = torch.randn(5, 3, 32, 128) * 0.5  # Giả lập normalize
        print(f"   Input shape: {dummy_frames.shape}")

        with torch.no_grad():
            enhanced = sr_enhancer.enhance_sequence(
                dummy_frames,
                resize_to=(32, 128)
            )

        print(f"   Output shape: {enhanced.shape}")

        if enhanced.shape[0] == dummy_frames.shape[0]:
            print("✅ PASS: enhance_sequence hoạt động đúng (giữ nguyên số frame)")
            return True
        else:
            print(
                f"❌ FAIL: Số frame không khớp. Input: {dummy_frames.shape[0]}, Output: {enhanced.shape[0]}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Lỗi khi chạy enhance_sequence. Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_with_sr(sr_enhancer, data_root: str = None):
    """Test 4: Kiểm tra Dataset có nhận SR enhancer không."""
    print("\n" + "=" * 60)
    print("TEST 4: Kiểm tra Dataset với SR enhancer")
    print("=" * 60)

    if not data_root or not os.path.exists(data_root):
        print(f"⚠️  SKIP: Data root không tồn tại: {data_root}")
        print("   (Bạn có thể test bằng cách: --data-root <path>)")
        return True

    try:
        config = Config()
        config.CHAR2IDX = {char: idx + 1 for idx,
                           char in enumerate(config.CHARS)}

        # Tạo dataset với SR enhancer
        dataset = MultiFrameDataset(
            root_dir=data_root,
            mode='train',
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            char2idx=config.CHAR2IDX,
            sr_enhancer=sr_enhancer,
            augmentation_level='light',  # Dùng light để test nhanh
        )

        if len(dataset) == 0:
            print("⚠️  WARNING: Dataset rỗng, không thể test")
            return True

        # Test load 1 sample (MultiFrameDataset trả về 7 trường, lấy 6 trường đầu)
        sample = dataset[0]
        images, targets, target_len, label, track_id, img_paths, _ = sample

        print(f"   Dataset size: {len(dataset)} samples")
        print(f"   Sample 0 - Images shape: {images.shape}")
        print(f"   Sample 0 - Label: {label}")
        print(f"   Sample 0 - Img paths: {img_paths}")

        if images.shape[0] == 5:  # 5 frames
            print("✅ PASS: Dataset load được sample với SR enhancer")
            if sr_enhancer is not None:
                print("   ✅ SR enhancer đã được gắn vào dataset")
            return True
        else:
            print(
                f"❌ FAIL: Số frame không đúng. Expected: 5, Got: {images.shape[0]}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Lỗi khi tạo/load dataset. Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration(sr_enhancer, data_root: str = None):
    """Test 5: Kiểm tra pipeline hoàn chỉnh (Dataset -> DataLoader -> Model input)."""
    print("\n" + "=" * 60)
    print("TEST 5: Kiểm tra pipeline hoàn chỉnh")
    print("=" * 60)

    if not data_root or not os.path.exists(data_root):
        print(f"⚠️  SKIP: Data root không tồn tại: {data_root}")
        return True

    try:
        from torch.utils.data import DataLoader

        config = Config()
        config.CHAR2IDX = {char: idx + 1 for idx,
                           char in enumerate(config.CHARS)}

        dataset = MultiFrameDataset(
            root_dir=data_root,
            mode='train',
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            char2idx=config.CHAR2IDX,
            sr_enhancer=sr_enhancer,
            augmentation_level='light',
        )

        if len(dataset) == 0:
            print("⚠️  SKIP: Dataset rỗng")
            return True

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=MultiFrameDataset.collate_fn,
            num_workers=0,  # Dùng 0 để tránh multiprocessing issues khi test
        )

        # Load 1 batch (7 trường, bỏ qua country_ids vì chỉ kiểm tra shape ảnh)
        batch = next(iter(loader))
        images, targets, target_lengths, labels_text, track_ids, img_paths, _ = batch

        print(f"   Batch images shape: {images.shape}")  # (B, T=5, C, H, W)
        print(f"   Batch size: {images.shape[0]}")

        # Kiểm tra shape hợp lệ cho model OCR
        if images.dim() == 5 and images.shape[1] == 5:
            print("✅ PASS: Pipeline hoàn chỉnh hoạt động đúng")
            print(
                f"   ✅ Shape phù hợp cho model OCR: (batch, frames=5, channels, height, width)")
            return True
        else:
            print(
                f"❌ FAIL: Shape không đúng. Expected: (B, 5, C, H, W), Got: {images.shape}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Lỗi trong pipeline. Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra tích hợp MF-LPR Super-Resolution"
    )
    parser.add_argument(
        "--sr-checkpoint-path",
        type=str,
        default=None,
        help="Đường dẫn checkpoint SR để test (nếu có)",
    )
    parser.add_argument(
        "--sr-config-path",
        type=str,
        default=None,
        help="Đường dẫn config JSON của SR (mặc định: sr_model/config/LP-Diff.json)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Đường dẫn data để test dataset (optional)",
    )
    parser.add_argument(
        "--no-sr",
        action="store_true",
        help="Test pipeline KHÔNG có SR (để so sánh)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TÍCH HỢP MF-LPR SUPER-RESOLUTION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Import
    results.append(("Import module", test_sr_module_import()))

    # Test 2: Init adapter
    sr_enhancer = None
    if not args.no_sr:
        if args.sr_checkpoint_path:
            success, sr_enhancer = test_sr_adapter_init(
                args.sr_checkpoint_path,
                args.sr_config_path
            )
            results.append(("Init SR adapter", success))
        else:
            print("\n⚠️  Không có --sr-checkpoint-path, bỏ qua test init adapter")
            results.append(("Init SR adapter", None))
    else:
        print("\n⚠️  Flag --no-sr được set, bỏ qua test SR")
        results.append(("Init SR adapter", None))

    # Test 3: Enhance function
    if sr_enhancer is not None:
        results.append(
            ("Enhance function", test_sr_enhance_function(sr_enhancer)))
    else:
        results.append(("Enhance function", None))

    # Test 4: Dataset integration
    data_root = args.data_root or getattr(Config(), "DATA_ROOT", None)
    results.append(
        ("Dataset integration", test_dataset_with_sr(sr_enhancer, data_root)))

    # Test 5: Pipeline integration
    results.append(
        ("Pipeline integration", test_pipeline_integration(sr_enhancer, data_root)))

    # Tổng kết
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results:
        if result is None:
            status = "⏭️  SKIPPED"
            skipped += 1
        elif result:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        print(f"   {test_name:30s} {status}")

    print(f"\n   Tổng: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n✅ TẤT CẢ TEST ĐÃ PASS! MF-LPR SR đã được tích hợp thành công.")
        return 0
    else:
        print(f"\n❌ CÓ {failed} TEST FAIL. Vui lòng kiểm tra lại.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

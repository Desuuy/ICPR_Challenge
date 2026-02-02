#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
from src.mf_svtrv2 import MultiFrameSVTRv2
from src.utils.common import seed_everything, clear_cuda_cache_and_report, print_model_memory_requirement
from src.training.trainer import Trainer
from src.models.restran import ResTranOCR
from src.models.crnn import MultiFrameCRNN
from src.data.dataset import MultiFrameDataset
from configs.config import Config
from torch.utils.data import DataLoader
import torch
import argparse
import os
import sys

# Giảm phân mảnh CUDA (tránh OOM do fragmentation)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/submission files (default: from config)"
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran", "mf_svtrv2"], default=None,
        help="Model architecture: 'crnn', 'restran' or 'mf_svtrv2' (default: from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="LSTM hidden size for CRNN (default: from config)"
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=None,
        help="Number of transformer attention heads (default: from config)"
    )
    parser.add_argument(
        "--transformer-layers", type=int, default=None,
        help="Number of transformer encoder layers (default: from config)"
    )
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level for training data (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save checkpoints and submission files (default: results/)",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable Spatial Transformer Network (STN) alignment",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on full dataset and generate submission file for test data",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Initialize config with CLI overrides
    config = Config()

    # Map CLI arguments to config attributes
    arg_to_config = {
        'experiment_name': 'EXPERIMENT_NAME',
        'model': 'MODEL_TYPE',
        'epochs': 'EPOCHS',
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'data_root': 'DATA_ROOT',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'hidden_size': 'HIDDEN_SIZE',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
        'svtr_dims': 'SVTR_DIMS',
        'svtr_depths': 'SVTR_DEPTHS',
        'svtr_heads': 'SVTR_HEADS',
    }

    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    # Special cases
    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level

    if args.no_stn:
        config.USE_STN = False

    # Output directory
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    seed_everything(config.SEED)

    # Làm trống cache CUDA và in bộ nhớ GPU mỗi lần chạy
    if config.DEVICE.type == "cuda":
        clear_cuda_cache_and_report()

    print(f"🚀 Configuration:")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")
    print(f"   SUBMISSION_MODE: {args.submission_mode}")

    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # Common dataset parameters
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
        'same_aug_per_sample': getattr(config, 'SAME_AUG_PER_SAMPLE', True),
    }

    # Create datasets based on mode
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        print("   - Training on FULL dataset (no validation split)")
        print("   - Will generate predictions for test data after training\n")

        # Create training dataset with full_train=True
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            full_train=True,
            **common_ds_params
        )

        # Create test dataset if test data exists
        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode='val',
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
                char2idx=config.CHAR2IDX,
                seed=config.SEED,
                is_test=True,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print(
                f"⚠️ WARNING: Test data not found at {config.TEST_DATA_ROOT}")

        val_loader = None
    else:
        # Normal training/validation split mode
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            **common_ds_params
        )

        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            **common_ds_params
        )

        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print("⚠️ WARNING: Validation dataset is empty.")

        test_loader = None

    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    # Create training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # For mf_svtrv2, restran and crnn
    # Initialize model based on config
    if config.MODEL_TYPE == "mf_svtrv2":
        model = MultiFrameSVTRv2(
            num_classes=config.NUM_CLASSES,
            use_stn=config.USE_STN,
            dropout=getattr(config, 'DROPOUT', 0.0),
        ).to(config.DEVICE)

        # Xác nhận architecture
        arch_info = model.verify_architecture()
        print(f"\n✅ Model đã được khởi tạo với:")
        print(f"   - STN: {'✅' if arch_info['has_stn'] else '❌'}")
        print(
            f"   - Backbone: {arch_info['backbone_type']} {'✅' if arch_info['has_backbone'] else '❌'}")
        print(
            f"   - Fusion: {arch_info['fusion_type']} {'✅' if arch_info['has_fusion'] else '❌'}")
        print(
            f"   - Head: {arch_info['head_type']} {'✅' if arch_info['has_head'] else '❌'}")

        # Nạp trọng số Pretrained UniRec40M
        pretrained_loaded = False
        if hasattr(config, 'PRETRAINED_PATH') and config.PRETRAINED_PATH:
            if os.path.exists(config.PRETRAINED_PATH):
                print(
                    f"\n🔄 Loading Pretrained Weights: {config.PRETRAINED_PATH}")
                model.load_weights(config.PRETRAINED_PATH)
                pretrained_loaded = True
            else:
                print(
                    f"\n⚠️ Pretrained path không tồn tại: {config.PRETRAINED_PATH}")
                print(f"   Model sẽ được train từ đầu (random initialization)")
        else:
            print(f"\nℹ️ Không có PRETRAINED_PATH trong config")
            print(f"   Model sẽ được train từ đầu (random initialization)")
    elif config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)

    # In chi tiết về model architecture và số lượng tham số
    print("\n" + "="*60)
    print("📋 MODEL ARCHITECTURE & PARAMETERS")
    print("="*60)

    if config.MODEL_TYPE == "mf_svtrv2":
        print(f"   Type: MultiFrameSVTRv2")
        print(f"   STN: {'✅ ENABLED' if config.USE_STN else '❌ DISABLED'}")
        print(f"   Backbone: SVTRv2LNConvTwo33")
        print(f"   Decoder: RCTCDecoder (CTC)")
        print(f"   Fusion: AttentionFusion (5 frames)")

        # Đếm params từng component
        if hasattr(model, 'stn') and config.USE_STN:
            stn_params = sum(p.numel() for p in model.stn.parameters())
            print(
                f"   STN params: {stn_params:,} ({stn_params*4/(1024**2):.2f} MB)")
        if hasattr(model, 'backbone'):
            backbone_params = sum(p.numel()
                                  for p in model.backbone.parameters())
            print(
                f"   Backbone params: {backbone_params:,} ({backbone_params*4/(1024**2):.2f} MB)")
        if hasattr(model, 'fusion'):
            fusion_params = sum(p.numel() for p in model.fusion.parameters())
            print(
                f"   Fusion params: {fusion_params:,} ({fusion_params*4/(1024**2):.2f} MB)")
        if hasattr(model, 'head'):
            head_params = sum(p.numel() for p in model.head.parameters())
            print(
                f"   Head params: {head_params:,} ({head_params*4/(1024**2):.2f} MB)")

        # Hiển thị trạng thái pretrained
        pretrained_status = "✅ LOADED" if pretrained_loaded else "❌ NOT LOADED (random init)"
        print(f"\n   Pretrained Weights: {pretrained_status}")
        if hasattr(config, 'PRETRAINED_PATH') and config.PRETRAINED_PATH:
            print(f"   Path: {config.PRETRAINED_PATH}")
    elif config.MODEL_TYPE == "restran":
        print(f"   Type: ResTranOCR")
        print(f"   STN: {'✅ ENABLED' if config.USE_STN else '❌ DISABLED'}")
    else:
        print(f"   Type: MultiFrameCRNN")
        print(f"   STN: {'✅ ENABLED' if config.USE_STN else '❌ DISABLED'}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes

    print(f"\n   📊 Total params: {total_params:,} ({param_size_mb:.2f} MB)")
    print(f"   📊 Trainable: {trainable_params:,}")
    print(f"   📊 Non-trainable: {total_params - trainable_params:,}")
    print("="*60 + "\n")

    # In số lượng bộ nhớ cần để chạy model
    print_model_memory_requirement(model, config.BATCH_SIZE, config.DEVICE)

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
    )

    trainer.fit()

    # Run test inference in submission mode
    if args.submission_mode and test_loader is not None:
        print("\n" + "="*60)
        print("📝 GENERATING SUBMISSION FILE")
        print("="*60)

        # Load best checkpoint if it exists
        exp_name = config.EXPERIMENT_NAME
        best_model_path = os.path.join(
            config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(
                best_model_path, map_location=config.DEVICE))
        else:
            print("⚠️ No best checkpoint found, using final model weights")

        # Run inference on test data
        trainer.predict_test(
            test_loader, output_filename=f"submission_{exp_name}_final.txt")


if __name__ == "__main__":
    main()

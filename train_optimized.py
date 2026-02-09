#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
from src.mf_svtrv2 import MultiFrameSVTRv2
from src.utils.common import seed_everything, clear_cuda_cache_and_report, print_model_memory_requirement
from src.training.trainer import Trainer
from src.models.restran import ResTranOCR
from src.models.crnn import MultiFrameCRNN
from src.data.dataset import MultiFrameDataset
from src.sr import MF_LPR_SR
from configs.config import Config
from torch.utils.data import DataLoader
import torch
import argparse
import os
import sys
from datetime import datetime


class Tee:
    """Ghi đồng thời ra stdout và file."""
    def __init__(self, filepath: str):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data: str):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout

# Giảm phân mảnh CUDA (tránh OOM do fragmentation)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
    parser.add_argument(
        "--test-data-root",
        type=str,
        default=None,
        help="Root directory for test data (default: config.TEST_DATA_ROOT = data/public_test)",
    )
    parser.add_argument(
        "--use-sr",
        action="store_true",
        help="Enable MF-LPR super-resolution on input frames (MF_LPR_SR)",
    )
    parser.add_argument(
        "--sr-checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint GEN path for MF-LPR SR model (e.g. I80000_E41_gen_best_psnr.pth)",
    )
    parser.add_argument(
        "--sr-config-path",
        type=str,
        default=None,
        help="Config JSON path for MF-LPR SR model (default: sr_model/config/LP-Diff.json)",
    )
    parser.add_argument(
        "--sr-n-timestep",
        type=int,
        default=None,
        help="Override n_timestep cho SR inference (10=nhanh, 100=mặc định, 1000=chất lượng cao; None=theo LP-Diff.json)",
    )
    
    # ============================================================
    # 2 FLAGS RIÊNG BIỆT CHO 2 LOẠI WEIGHTS
    # ============================================================
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Không load pretrained UniRec weights (weights/best.pth) - train từ random init",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Không load checkpoint từ previous training (results/mf_svtrv2_best.pth) - train from scratch",
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
        'test_data_root': 'TEST_DATA_ROOT',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'hidden_size': 'HIDDEN_SIZE',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
        'svtr_dims': 'SVTR_DIMS',
        'svtr_depths': 'SVTR_DEPTHS',
        'svtr_heads': 'SVTR_HEADS',
        'use_sr': 'USE_SR',
        'sr_checkpoint_path': 'SR_CHECKPOINT_PATH',
        'sr_config_path': 'SR_CONFIG_PATH',
        'sr_n_timestep': 'SR_N_TIMESTEP',
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

    if args.use_sr:
        sr_model_path = os.path.join(os.path.dirname(__file__), "sr_model")
        if not os.path.isdir(sr_model_path):
            print("❌ ERROR: --use-sr requires sr_model/ folder (MF-LPR SR).")
            print("   SR integration is on branch feat/Restoration_Module.")
            print("   Use: git checkout feat/Restoration_Module")
            sys.exit(1)
        config.USE_SR = True

    # Output directory
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Ghi log terminal vào file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.OUTPUT_DIR, f"train_log_{timestamp}.txt")
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"📋 Logging to {log_path}")

    try:
        _run_training(args, config)
    finally:
        tee.close()
        print(f"📋 Log đã lưu: {log_path}")


def _run_training(args, config):

    # Làm trống cache CUDA và in bộ nhớ GPU
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
    print(f"   USE_SR: {getattr(config, 'USE_SR', False)}")
    
    # Hiển thị scheduler type
    scheduler_type = getattr(config, 'SCHEDULER_TYPE', 'onecycle')
    print(f"   SCHEDULER: {scheduler_type}")
    
    # Hiển thị loading flags
    print(f"   LOAD_PRETRAINED: {not args.no_pretrained}")
    print(f"   LOAD_CHECKPOINT: {not args.no_checkpoint}")
    
    use_focal = getattr(config, 'USE_FOCAL_CTC', False)
    print(f"   USE_FOCAL_CTC: {use_focal}  ->  LOSS: {'Focal CTC' if use_focal else 'CTC'}")
    print(f"   SUBMISSION_MODE: {args.submission_mode}")
    if args.submission_mode:
        print(f"   TEST_DATA_ROOT: {config.TEST_DATA_ROOT}")

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

    # Optional: initialize super-resolution enhancer
    sr_enhancer = None
    if getattr(config, "USE_SR", False):
        print("\n" + "="*60)
        print("🔍 KIỂM TRA TÍCH HỢP MF-LPR SUPER-RESOLUTION")
        print("="*60)
        if not getattr(config, "SR_CHECKPOINT_PATH", ""):
            print("⚠️ USE_SR=True nhưng SR_CHECKPOINT_PATH đang rỗng -> SR sẽ không được dùng.")
            print("   💡 Để bật SR, hãy set SR_CHECKPOINT_PATH trong config hoặc dùng --sr-checkpoint-path")
        else:
            try:
                print(f"📦 Đang load checkpoint SR: {config.SR_CHECKPOINT_PATH}")
                sr_enhancer = MF_LPR_SR(
                    checkpoint_path=config.SR_CHECKPOINT_PATH,
                    config_path=getattr(config, "SR_CONFIG_PATH", "sr_model/config/LP-Diff.json"),
                    device=config.DEVICE,
                    n_timestep_override=getattr(config, "SR_N_TIMESTEP", None),
                )
                print("✅ MF-LPR Super-Resolution đã được khởi tạo thành công!")
                print(f"   - Device: {config.DEVICE}")
                print(f"   - Checkpoint: {config.SR_CHECKPOINT_PATH}")
                print(f"   - Config: {getattr(config, 'SR_CONFIG_PATH', 'sr_model/config/LP-Diff.json')}")
                sr_nt = getattr(config, 'SR_N_TIMESTEP', None)
                print(f"   - n_timestep: {sr_nt if sr_nt is not None else 'theo LP-Diff.json'}")
                print("   - Status: SR sẽ được áp dụng cho TẤT CẢ frames trong dataset (train/val/test)")
                print("="*60 + "\n")
            except Exception as e:
                print(f"❌ Không thể khởi tạo MF_LPR_SR, sẽ tắt SR. Lý do: {e}")
                import traceback
                traceback.print_exc()
                sr_enhancer = None
                print("="*60 + "\n")
    else:
        print(f"\nℹ️  USE_SR=False -> Pipeline chạy KHÔNG có Super-Resolution\n")

    # Khi dùng SR: num_workers=0 để tránh CUDA fork error
    num_workers = 0 if sr_enhancer is not None else config.NUM_WORKERS
    if sr_enhancer is not None:
        print(f"⚠️  USE_SR=True -> num_workers=0 (tránh lỗi CUDA fork)\n")

    # Create datasets based on mode
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        print("   - Training on FULL dataset (no validation split)")
        print("   - Will generate predictions for test data after training\n")

        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            full_train=True,
            sr_enhancer=sr_enhancer,
            **common_ds_params
        )

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
                sr_enhancer=sr_enhancer,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            print(f"⚠️ WARNING: Test data not found at {config.TEST_DATA_ROOT}")

        val_loader = None
    else:
        # Normal training/validation split mode
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            sr_enhancer=sr_enhancer,
            **common_ds_params
        )

        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            sr_enhancer=sr_enhancer,
            **common_ds_params
        )

        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=num_workers,
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
        num_workers=num_workers,
        pin_memory=True
    )

    # ============================================================
    # INITIALIZE MODEL
    # ============================================================
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
        print(f"   - Backbone: {arch_info['backbone_type']} {'✅' if arch_info['has_backbone'] else '❌'}")
        print(f"   - Fusion: {arch_info['fusion_type']} {'✅' if arch_info['has_fusion'] else '❌'}")
        print(f"   - Head: {arch_info['head_type']} {'✅' if arch_info['has_head'] else '❌'}")

        # ============================================================
        # STEP 1: LOAD PRETRAINED UNIREC WEIGHTS (weights/best.pth)
        # ============================================================
        pretrained_loaded = False
        if not args.no_pretrained and hasattr(config, 'PRETRAINED_PATH') and config.PRETRAINED_PATH:
            if os.path.exists(config.PRETRAINED_PATH):
                print(f"\n" + "="*60)
                print(f"📦 STEP 1: LOADING PRETRAINED UNIREC WEIGHTS")
                print("="*60)
                print(f"   Path: {config.PRETRAINED_PATH}")
                try:
                    model.load_weights(config.PRETRAINED_PATH)
                    pretrained_loaded = True
                    print(f"   ✅ Successfully loaded pretrained UniRec backbone!")
                except Exception as e:
                    print(f"   ❌ Failed to load pretrained weights: {e}")
                print("="*60 + "\n")
            else:
                print(f"\n⚠️ Pretrained path không tồn tại: {config.PRETRAINED_PATH}\n")
        else:
            if args.no_pretrained:
                print(f"\n" + "="*60)
                print(f"⚙️  STEP 1: SKIP PRETRAINED WEIGHTS (--no-pretrained flag)")
                print("="*60)
                print(f"   Model will use random initialization")
                print("="*60 + "\n")
            else:
                print(f"\nℹ️  No PRETRAINED_PATH in config\n")

    elif config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)
        pretrained_loaded = False
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)
        pretrained_loaded = False

    # ============================================================
    # PRINT MODEL ARCHITECTURE & PARAMETERS
    # ============================================================
    print("="*60)
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
            print(f"   STN params: {stn_params:,} ({stn_params*4/(1024**2):.2f} MB)")
        if hasattr(model, 'backbone'):
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            print(f"   Backbone params: {backbone_params:,} ({backbone_params*4/(1024**2):.2f} MB)")
        if hasattr(model, 'fusion'):
            fusion_params = sum(p.numel() for p in model.fusion.parameters())
            print(f"   Fusion params: {fusion_params:,} ({fusion_params*4/(1024**2):.2f} MB)")
        if hasattr(model, 'head'):
            head_params = sum(p.numel() for p in model.head.parameters())
            print(f"   Head params: {head_params:,} ({head_params*4/(1024**2):.2f} MB)")

    elif config.MODEL_TYPE == "restran":
        print(f"   Type: ResTranOCR")
        print(f"   STN: {'✅ ENABLED' if config.USE_STN else '❌ DISABLED'}")
    else:
        print(f"   Type: MultiFrameCRNN")
        print(f"   STN: {'✅ ENABLED' if config.USE_STN else '❌ DISABLED'}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 ** 2)

    print(f"\n   📊 Total params: {total_params:,} ({param_size_mb:.2f} MB)")
    print(f"   📊 Trainable: {trainable_params:,}")
    print(f"   📊 Non-trainable: {total_params - trainable_params:,}")
    print("="*60 + "\n")

    # Print memory requirement
    print_model_memory_requirement(model, config.BATCH_SIZE, config.DEVICE)

    # ============================================================
    # STEP 2: LOAD CHECKPOINT FROM PREVIOUS TRAINING
    # (results/mf_svtrv2_best.pth) - OVERRIDES PRETRAINED
    # ============================================================
    checkpoint_loaded = False
    if not args.no_checkpoint:
        checkpoint_path = os.path.join(config.OUTPUT_DIR, f"{config.EXPERIMENT_NAME}_best.pth")
        if os.path.exists(checkpoint_path):
            print("="*60)
            print("🔄 STEP 2: LOADING CHECKPOINT FROM PREVIOUS TRAINING")
            print("="*60)
            print(f"   Path: {checkpoint_path}")
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
                checkpoint_loaded = True
                print(f"   ✅ Successfully loaded checkpoint!")
                print(f"   📊 This OVERRIDES pretrained UniRec weights")
                
                # Hiển thị scheduler warning
                scheduler_type = getattr(config, 'SCHEDULER_TYPE', 'onecycle')
                print(f"   📊 Scheduler: {scheduler_type}")
                
                if scheduler_type == 'cosine':
                    print(f"   ✅ Cosine scheduler: Fine-tuning from checkpoint")
                elif scheduler_type == 'onecycle':
                    print(f"   ⚠️  WARNING: OneCycleLR will RESTART LR curve!")
                    print(f"       Consider using SCHEDULER_TYPE='cosine' for fine-tuning")
                
            except Exception as e:
                print(f"   ❌ Failed to load checkpoint: {e}")
                print(f"   Will use {'pretrained' if pretrained_loaded else 'random'} weights")
            
            print("="*60 + "\n")
        else:
            print("="*60)
            print("ℹ️  STEP 2: NO CHECKPOINT FOUND")
            print("="*60)
            print(f"   Path checked: {checkpoint_path}")
            print(f"   Starting from {'pretrained UniRec' if pretrained_loaded else 'random'} weights")
            print("="*60 + "\n")
    else:
        print("="*60)
        print("⚙️  STEP 2: SKIP CHECKPOINT (--no-checkpoint flag)")
        print("="*60)
        print(f"   Training from {'pretrained UniRec' if pretrained_loaded else 'random'} weights")
        print("="*60 + "\n")

    # ============================================================
    # SUMMARY: WHAT WEIGHTS ARE LOADED
    # ============================================================
    print("="*60)
    print("📊 WEIGHTS LOADING SUMMARY")
    print("="*60)
    if checkpoint_loaded:
        print(f"   ✅ USING: Checkpoint from previous training (79.38%)")
        print(f"      Path: {checkpoint_path}")
        print(f"   ℹ️  Pretrained UniRec was loaded but OVERRIDDEN by checkpoint")
    elif pretrained_loaded:
        print(f"   ✅ USING: Pretrained UniRec backbone")
        print(f"      Path: {config.PRETRAINED_PATH}")
        print(f"   ℹ️  No checkpoint found - training from pretrained")
    else:
        print(f"   ⚠️  USING: Random initialization")
        print(f"   ℹ️  Both --no-pretrained and --no-checkpoint flags set")
    print("="*60 + "\n")

    # ============================================================
    # INITIALIZE TRAINER AND START TRAINING
    # ============================================================
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

        exp_name = config.EXPERIMENT_NAME
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        else:
            print("⚠️ No best checkpoint found, using final model weights")

        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_final.txt")


if __name__ == "__main__":
    main()
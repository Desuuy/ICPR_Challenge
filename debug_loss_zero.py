#!/usr/bin/env python3
"""Debug script for Loss = 0 issue."""
import torch
import sys
import os

print("="*70)
print("🔍 DEBUGGING LOSS = 0 ISSUE")
print("="*70)

# ============================================================
# TEST 1: Model Forward Output
# ============================================================
print("\n1️⃣ Testing model forward output...")
try:
    from src.mf_svtrv2 import MultiFrameSVTRv2
    
    model = MultiFrameSVTRv2(num_classes=37, use_stn=False)
    x = torch.randn(2, 5, 3, 32, 128)
    c = torch.tensor([0, 1])
    
    with torch.no_grad():
        out = model(x, c)
    
    print(f"   Output shape: {out.shape}")
    print(f"   Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"   Contains NaN: {torch.isnan(out).any()}")
    print(f"   Contains Inf: {torch.isinf(out).any()}")
    print(f"   Mean: {out.mean():.4f}")
    print(f"   Std: {out.std():.4f}")
    
    # Check log_softmax values
    if out.min() < -100:
        print("   ⚠️ WARNING: Very negative values (underflow risk)")
    if out.max() > 0:
        print("   ⚠️ WARNING: Positive log_softmax values (not normalized)")
    
    print("   ✅ Model forward OK")
except Exception as e:
    print(f"   ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# TEST 2: CTC Loss Computation
# ============================================================
print("\n2️⃣ Testing CTC loss computation...")
try:
    # Giả lập preds và targets
    batch_size = 2
    seq_len = 32
    num_classes = 37
    
    # Preds: [Seq, Batch, Classes] (CTC expects time-first)
    preds = torch.randn(seq_len, batch_size, num_classes).log_softmax(2)
    
    # Targets
    targets = torch.tensor([1, 2, 3, 4, 5, 6, 7])  # "ABC1234"
    target_lengths = torch.tensor([7, 7])
    input_lengths = torch.tensor([seq_len, seq_len])
    
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(preds, targets, input_lengths, target_lengths)
    
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Loss is zero: {loss.item() == 0.0}")
    print(f"   Loss is NaN: {torch.isnan(loss)}")
    print(f"   Loss is Inf: {torch.isinf(loss)}")
    
    if loss.item() == 0.0:
        print("   ⚠️ CRITICAL: Loss is exactly 0!")
        print("   → Model might be predicting blank for everything")
    elif loss.item() < 0.01:
        print("   ⚠️ WARNING: Loss too small (near-perfect prediction?)")
    elif loss.item() > 100:
        print("   ⚠️ WARNING: Loss too large (divergence?)")
    else:
        print("   ✅ CTC loss OK")
        
except Exception as e:
    print(f"   ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# TEST 3: Dataset Loading
# ============================================================
print("\n3️⃣ Testing dataset loading...")
try:
    from configs.config import Config
    from src.data.dataset import MultiFrameDataset
    
    config = Config()
    
    ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='train',
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
    )
    
    print(f"   Dataset size: {len(ds)}")
    
    if len(ds) > 0:
        sample = ds[0]
        images, targets, target_len, label, track_id, paths, country_id = sample
        
        print(f"   Images shape: {images.shape}")
        print(f"   Targets: {targets}")
        print(f"   Target length: {target_len}")
        print(f"   Label: '{label}'")
        print(f"   Country ID: {country_id}")
        
        if target_len == 0:
            print("   ⚠️ CRITICAL: Target length is 0!")
        if len(targets) == 0:
            print("   ⚠️ CRITICAL: Empty targets!")
        
        print("   ✅ Dataset OK")
    else:
        print("   ❌ FAIL: Dataset is empty!")
        
except Exception as e:
    print(f"   ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# TEST 4: Config Values
# ============================================================
print("\n4️⃣ Checking config values...")
try:
    from configs.config import Config
    config = Config()
    
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   GRAD_CLIP: {config.GRAD_CLIP}")
    print(f"   DROPOUT: {getattr(config, 'DROPOUT', 'NOT SET')}")
    print(f"   LABEL_SMOOTHING: {getattr(config, 'LABEL_SMOOTHING', 'NOT SET')}")
    print(f"   USE_FOCAL_CTC: {getattr(config, 'USE_FOCAL_CTC', False)}")
    print(f"   SCHEDULER_TYPE: {getattr(config, 'SCHEDULER_TYPE', 'onecycle')}")
    
    # Check problematic values
    if config.LEARNING_RATE == 0:
        print("   ❌ CRITICAL: Learning rate is 0!")
    elif config.LEARNING_RATE > 0.01:
        print("   ⚠️ WARNING: Learning rate too high")
    
    if config.GRAD_CLIP == 0:
        print("   ⚠️ WARNING: Gradient clipping disabled")
    
    print("   ✅ Config OK")
    
except Exception as e:
    print(f"   ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# TEST 5: Full Forward Pass with Real Data
# ============================================================
print("\n5️⃣ Testing full forward pass with real data...")
try:
    from configs.config import Config
    from src.mf_svtrv2 import MultiFrameSVTRv2
    from src.data.dataset import MultiFrameDataset
    from torch.utils.data import DataLoader
    
    config = Config()
    model = MultiFrameSVTRv2(num_classes=37, use_stn=False).to(config.DEVICE)
    
    ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='train',
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
    )
    
    loader = DataLoader(ds, batch_size=2, collate_fn=MultiFrameDataset.collate_fn)
    
    # Get one batch
    batch = next(iter(loader))
    images, targets, target_lengths, _, _, _, country_ids = batch
    
    images = images.to(config.DEVICE)
    targets = targets.to(config.DEVICE)
    country_ids = country_ids.to(config.DEVICE)
    
    print(f"   Batch images: {images.shape}")
    print(f"   Batch targets: {targets.shape}")
    print(f"   Target lengths: {target_lengths}")
    print(f"   Country IDs: {country_ids}")
    
    # Forward pass
    with torch.no_grad():
        preds = model(images, country_ids)
    
    print(f"   Preds shape: {preds.shape}")
    print(f"   Preds range: [{preds.min():.4f}, {preds.max():.4f}]")
    
    # Compute loss
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    input_lengths = torch.full((images.size(0),), preds.size(1), dtype=torch.long)
    
    loss = criterion(preds.permute(1, 0, 2), targets, input_lengths, target_lengths)
    
    print(f"   Loss: {loss.item():.6f}")
    
    if loss.item() == 0.0:
        print("   ❌ CRITICAL: Loss is 0 with real data!")
        print("   → Checking predictions...")
        
        # Decode predictions
        preds_np = preds.argmax(dim=2).cpu().numpy()
        print(f"   Sample predictions: {preds_np[0][:20]}")
        
        # Count blank predictions
        blank_count = (preds_np == 0).sum()
        total = preds_np.size
        print(f"   Blank predictions: {blank_count}/{total} ({100*blank_count/total:.1f}%)")
        
        if blank_count > total * 0.9:
            print("   ❌ Model predicting mostly blanks!")
    else:
        print("   ✅ Loss computation working!")
        
except Exception as e:
    print(f"   ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print("If all tests passed, the issue might be in:")
print("  1. Gradient accumulation logic")
print("  2. Optimizer step conditions")
print("  3. Loss scaling/normalization")
print("  4. Label encoding mismatch")
print("\nPlease share:")
print("  - configs/config.py")
print("  - Latest training log (first 30 lines)")
print("  - This debug output")
print("="*70)
"""Common utility functions."""
import os
import random

import numpy as np
import torch


def clear_cuda_cache_and_report() -> None:
    """Làm trống cache CUDA và in ra bộ nhớ GPU (total, free, allocated, reserved)."""
    if not torch.cuda.is_available():
        print("📌 CUDA không khả dụng, bỏ qua clear GPU.")
        return
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = total - reserved  # gần đúng vì reserved có thể > allocated
    def _mb(x):
        return x / (1024 ** 2)
    print("🔄 Đã xóa cache CUDA. Bộ nhớ GPU:")
    print(f"   Total:     {_mb(total):.2f} MiB ({_mb(total) / 1024:.2f} GiB)")
    print(f"   Allocated: {_mb(allocated):.2f} MiB")
    print(f"   Reserved:  {_mb(reserved):.2f} MiB")
    print(f"   Free:      {_mb(max(0, total - reserved)):.2f} MiB")


def print_model_memory_requirement(model: torch.nn.Module, batch_size: int, device: torch.device) -> None:
    """In ra số lượng bộ nhớ cần để chạy model (params + optimizer ước tính + gợi ý batch)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # float32: 4 bytes/param; AdamW: 2 states (momentum, variance) ~ 8 bytes/param
    param_mb = total_params * 4 / (1024 ** 2)
    optimizer_mb = trainable_params * 8 / (1024 ** 2)  # AdamW
    # Ước tính activation cho 1 batch: B*5*C*H*W + các feature maps trung gian (thô: ~3–5x input)
    # Input: batch_size * 5 * 3 * 32 * 128 * 4 bytes
    input_mb = batch_size * 5 * 3 * 32 * 128 * 4 / (1024 ** 2)
    activations_estimate_mb = input_mb * 4  # heuristic
    total_estimate_mb = param_mb + optimizer_mb + activations_estimate_mb
    total_estimate_gib = total_estimate_mb / 1024

    print("📊 Bộ nhớ cần để chạy model (ước tính):")
    print(f"   Tham số:        {total_params:,} (~{param_mb:.2f} MiB)")
    print(f"   Trainable:      {trainable_params:,}")
    print(f"   Optimizer (AdamW): ~{optimizer_mb:.2f} MiB")
    print(f"   Batch size:     {batch_size} (input ~{input_mb:.2f} MiB)")
    print(f"   Activations (ước tính): ~{activations_estimate_mb:.2f} MiB")
    print(f"   Tổng ước tính: ~{total_estimate_mb:.2f} MiB (~{total_estimate_gib:.2f} GiB)")
    if torch.cuda.is_available():
        free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024 ** 2)
        if free_mb < total_estimate_mb:
            suggest = max(1, batch_size // 2)
            print(f"   ⚠️ GPU free ~{free_mb:.0f} MiB < ước tính ~{total_estimate_mb:.0f} MiB. Thử giảm --batch-size (vd: --batch-size {suggest}).")
        else:
            print(f"   ✓ GPU free ~{free_mb:.0f} MiB, đủ cho ước tính ~{total_estimate_mb:.0f} MiB.")


def seed_everything(seed: int = 42, benchmark: bool = False) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
        benchmark: If True, enables CUDNN benchmark for speed; disables deterministic mode.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if benchmark:
        print(f"⚡ Benchmark mode ENABLED (Speed optimized). Deterministic mode DISABLED.")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print(f"🔒 Deterministic mode ENABLED (Reproducibility optimized). Benchmark mode DISABLED.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

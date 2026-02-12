"""Snapshot of configuration that achieved best validation accuracy (~79.38%).

This file is READ-ONLY reference. It does NOT affect training unless you
explicitly import and use it in your scripts.

Original run characteristics (from train_log_20260208_175154.txt):
    - MODEL: mf_svtrv2
    - USE_STN: False
    - USE_SR: False
    - USE_FOCAL_CTC: False  (CTC loss)
    - EPOCHS: 30
    - BATCH_SIZE: 64
    - LEARNING_RATE: 0.00065  (base LR, with OneCycle-style schedule)
    - IMG_SIZE: 3 x 32 x 128
    - CTC_BEAM_WIDTH: 5
    - DROPOUT: 0.3 (STN/Fusion)
    - DATA_ROOT: Data/train
    - VAL_SPLIT_FILE: Data/val_tracks.json
    - CHARSET: 0–9, A–Z  (36 chars + blank)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import os
import torch


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class BestAccConfig:
    """Config snapshot for the 79.38% Val Acc run."""

    # Core training setup
    MODEL_TYPE: str = "mf_svtrv2"
    EXPERIMENT_NAME: str = "mf_svtrv2"

    # Scheduler used in the best run
    SCHEDULER_TYPE: str = "onecycle"

    # Training hyperparameters (base values)
    LEARNING_RATE: float = 6.5e-4
    EPOCHS: int = 30
    BATCH_SIZE: int = 64
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 0.1
    GRAD_CLIP: float = 1.0

    # Multi-frame / decoding
    SPLIT_RATIO: float = 0.9
    CTC_BEAM_WIDTH: int = 5
    SAME_AUG_PER_SAMPLE: bool = True
    DROPOUT: float = 0.3

    # Modules toggles during best run
    USE_STN: bool = False
    USE_SR: bool = False
    USE_FOCAL_CTC: bool = False
    USE_TEMP_SCALING: bool = False

    # Data & paths (relative to project root)
    DATA_ROOT: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "Data", "train")
    )
    TEST_DATA_ROOT: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "Data", "Pa7a3Hin-test-public")
    )
    VAL_SPLIT_FILE: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "Data", "val_tracks.json")
    )
    OUTPUT_DIR: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "results")
    )
    SUBMISSION_FILE: str = "submission.txt"

    # Image & charset
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Wrong prediction logging
    SAVE_WRONG_PREDICTIONS: bool = True
    SAVE_WRONG_IMAGES: bool = True

    # Pretrained backbone path
    PRETRAINED_PATH: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "weights", "best.pth")
    )

    # SVTRv2 architecture (matches UniRec config)
    SVTR_DIMS: list[int] = field(default_factory=lambda: [128, 256, 384])
    SVTR_DEPTHS: list[int] = field(default_factory=lambda: [6, 6, 6])
    SVTR_HEADS: list[int] = field(default_factory=lambda: [4, 8, 12])

    # Device
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Derived attributes
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Compute derived attributes after initialization."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for CTC blank


def get_best_acc_config() -> BestAccConfig:
    """Helper to get the best-acc config snapshot."""
    return BestAccConfig()


"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict
import torch


@dataclass
class Config:
    """Training configuration with all hyperparameters."""

    # Experiment tracking
    MODEL_TYPE: str = "mf_svtrv2"  # "crnn" or "restran" or "mf_svtrv2"
    EXPERIMENT_NAME: str = MODEL_TYPE
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True  # Enable Spatial Transformer Network

    # Data paths
    DATA_ROOT: str = "data/train"
    TEST_DATA_ROOT: str = "data/test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"

    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128

    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."

    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.000325
    EPOCHS: int = 1
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 0.05
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False

    # Accuracy improvements (see docs/IMPROVEMENT_PROPOSALS.md)
    # Focus on hard samples (sample-level weighting)
    USE_FOCAL_CTC: bool = True
    CTC_BEAM_WIDTH: int = 1      # 1 = greedy decode; 5–10 = beam search
    SAME_AUG_PER_SAMPLE: bool = True  # Same augmentation for all 5 frames
    DROPOUT: float = 0.1         # Dropout in STN/Fusion (0 = disabled)
    # Save wrong predictions for analysis
    SAVE_WRONG_PREDICTIONS: bool = True
    # Copy wrong-prediction images to results/wrong_images_*/ for inspection
    SAVE_WRONG_IMAGES: bool = True
    # Super-Resolution (MF-LPR SR) - requires sr_model/ (LP-Diff or similar)
    USE_SR: bool = True
    SR_CHECKPOINT_PATH: str = "weights/gen_best_psnr.pth" 
    SR_CONFIG_PATH: str = "sr_model/config/LP-Diff.json"

    # Pretrained path
    PRETRAINED_PATH: str = r"weights/best.pth"

    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25

    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1

    # SVTRv2-Small model hyperparameters
    SVTR_DIMS: list = field(default_factory=lambda: [
                            128, 256, 384])  # Khớp dims
    SVTR_DEPTHS: list = field(default_factory=lambda: [
                              6, 6, 6])     # Khớp depths
    SVTR_HEADS: list = field(default_factory=lambda: [
                             4, 8, 12])    # Khớp num_heads


    DEVICE: torch.device = field(default_factory=lambda: torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    OUTPUT_DIR: str = "results"

    # Derived attributes (computed in __post_init__)
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self):
        """Compute derived attributes after initialization."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()

"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import torch

# Project root (thư mục chứa train.py) - không phụ thuộc cwd
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    """Training configuration with all hyperparameters."""

    # File config nếu train tiếp 
    SCHEDULER_TYPE: str = "cosine"
    LEARNING_RATE: float = 3e-5  # Thấp hơn 0.00065
    EPOCHS: int = 50
    CTC_BEAM_WIDTH: int = 5
    LABEL_SMOOTHING: float = 0.1


    # Focus on hard samples (sample-level weighting)
    USE_FOCAL_CTC: bool = False 
    # Enable Spatial Transformer Network (False to avoid NaN when STN chưa học)
    USE_STN: bool = False  
    # Super-Resolution (MF-LPR SR) - requires sr_model/ (LP-Diff or similar)
    USE_SR: bool = False

    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.00003 # Giảm từ 0.00325 để tránh gradient explosion/NaN
    EPOCHS: int = 1
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 0.1
    GRAD_CLIP: float = 1.0  # Giảm từ 5.0 để ổn định gradient, tránh NaN
    SPLIT_RATIO: float = 0.9
    # 1 = greedy decode; 5–10 = beam search
    CTC_BEAM_WIDTH: int = 5 # thay từ 1 thành 5
    # Same augmentation for all 5 frames 
    SAME_AUG_PER_SAMPLE: bool = True  
    # Dropout in STN/Fusion (0 = disabled)
    DROPOUT: float = 0.3   
    USE_CUDNN_BENCHMARK: bool = False

    USE_TEMP_SCALING: bool = False
    # Experiment tracking
    MODEL_TYPE: str = "mf_svtrv2"  # "crnn" or "restran" or "mf_svtrv2"
    EXPERIMENT_NAME: str = MODEL_TYPE
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    
    # Data paths (tương đối project root, không phụ thuộc cwd)
    DATA_ROOT: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "Data", "train"))
    TEST_DATA_ROOT: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "Data", "Pa7a3Hin-test-public"))
    VAL_SPLIT_FILE: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "Data", "val_tracks.json"))
    SUBMISSION_FILE: str = "submission.txt"

    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128  # Backbone hardcode max_sz=[32,128]; tăng cần sửa mf_svtrv2.py

    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
    # Save wrong predictions for analysis
    SAVE_WRONG_PREDICTIONS: bool = True
    # Copy wrong-prediction images to results/wrong_images_*/ for inspection
    SAVE_WRONG_IMAGES: bool = True
    
    SR_CHECKPOINT_PATH: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "weights", "gen_best_psnr.pth"))
    SR_CONFIG_PATH: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "sr_model", "config", "LP-Diff.json"))
    # Override n_timestep cho SR inference (None = dùng từ LP-Diff.json; 10/100/1000 = nhanh/chất lượng)
    SR_N_TIMESTEP: Optional[int] = None

    # Pretrained path
    PRETRAINED_PATH: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "weights", "best.pth"))

    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25

    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1

    # SVTRv2-Base model hyperparameters
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

"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import torch

# Project root (thư mục chứa train.py) - không phụ thuộc cwd
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:

    # File config nếu train tiếp
    SCHEDULER_TYPE: str = "cosine"  # "onecycle" or "cosine"
    LEARNING_RATE: float = 0.000325
    EPOCHS: int = 1
    CTC_BEAM_WIDTH: int = 5
    # Label smoothing cho CTC: tạm thời đặt 0.0 để tránh làm sai phân phối log-prob của CTC
    LABEL_SMOOTHING: float = 0.0
    # Bật chế độ submission: train full data + tạo file submission cho test (có thể bật bằng config hoặc --submission-mode)
    SUBMISSION_MODE: bool = False

    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 256

    # Multi-Scale Resize (MSR)
    # Để đơn giản pipeline và debug dễ hơn, mặc định TẮT MSR;
    # khi đã ổn định có thể bật lại để tăng robustness.
    USE_MSR: bool = False
    MSR_WIDTH_MIN: int = 64
    # MAX width cho MSR phải khớp backbone SVTRv2 (max_sz=[32, 128])
    MSR_WIDTH_MAX: int = 256  # Chỉnh từ 128 -> 256

    DROPOUT: float = 0.05

    # Focus on hard samples (sample-level weighting)
    # Bật Focal Loss CTC / nhánh GTC khi đã debug xong.
    USE_FOCAL_CTC: bool = False
    # Nhánh GTC/SMTR (GTCLoss + GTCDecoder) – giữ False để pipeline CTC cơ bản ổn định.
    USE_GTC: bool = False
    # Bật STN
    USE_STN: bool = False

    # Super-Resolution (MF-LPR SR) - requires sr_model/ (LP-Diff or similar)
    USE_SR: bool = False

    # Save wrong predictions for analysis
    SAVE_WRONG_PREDICTIONS: bool = True

    # Copy wrong-prediction images to results/wrong_images_*/ for inspection
    SAVE_WRONG_IMAGES: bool = True

    # Training hyperparameters
    BATCH_SIZE: int = 32
    SEED: int = 42
    NUM_WORKERS: int = 0  # 0 tránh lỗi "paging file too small" khi spawn nhiều worker load CUDA trên Windows
    WEIGHT_DECAY: float = 0.05
    GRAD_CLIP: float = 1.0
    # Gradient accumulation steps (Effective Batch = BATCH_SIZE * ACCUM_STEPS)
    ACCUM_STEPS: int = 8
    SPLIT_RATIO: float = 0.9

    # Same augmentation for all 5 frames
    SAME_AUG_PER_SAMPLE: bool = True
    # Dropout in STN/Fusion (0 = disabled)

    USE_CUDNN_BENCHMARK: bool = False

    USE_TEMP_SCALING: bool = False

    # Overfit subset nhỏ để debug (0 = tắt)
    OVERFIT_NUM_TRAIN_SAMPLES: int = 0
    OVERFIT_NUM_VAL_SAMPLES: int = 0

    # Experiment tracking
    # "crnn" or "restran" or "mf_svtrv2"
    MODEL_TYPE: str = "mf_svtrv2"
    EXPERIMENT_NAME: str = MODEL_TYPE
    AUGMENTATION_LEVEL: str = "light"  # "full" or "light"

    # Character set cho CTC/GTC (khớp EN_symbol_dict.txt)
    # Nếu sau này bạn đổi file vocab, chỉ cần sửa CHARS cho trùng là được.
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Độ dài tối đa chuỗi text dùng cho encoder GTC (SMTR) khi ta bật nhánh GTC
    # (config.yml gốc dùng 25; với biển số 7 ký tự thì 10–12 cũng đủ, nhưng 25 an toàn hơn).
    MAX_TEXT_LENGTH: int = 25

    # Đường dẫn file vocab theo format OpenRec (mỗi ký tự 1 dòng).
    # Hiện đang trỏ tới EN_symbol_dict.txt mà bạn đã tạo.
    CHAR_DICT_PATH: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "EN_symbol_dict.txt"))

    # Data paths
    DATA_ROOT: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "Data", "train"))
    TEST_DATA_ROOT: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "Data", "Pa7a3Hin-test-public"))
    VAL_SPLIT_FILE: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "Data", "val_tracks.json"))
    SUBMISSION_FILE: str = "submission.txt"

    SR_CHECKPOINT_PATH: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "weights", "gen_best_psnr.pth"))
    SR_CONFIG_PATH: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "sr_model", "config", "LP-Diff.json"))
    # Override n_timestep cho SR inference (None = dùng từ LP-Diff.json; 10/100/1000 = nhanh/chất lượng)
    SR_N_TIMESTEP: Optional[int] = None

    # Pretrained path
    PRETRAINED_PATH: str = field(default_factory=lambda: os.path.join(
        _PROJECT_ROOT, "weights", "ocr_best.pth"))

    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.1

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

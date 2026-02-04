## MultiFrame-LPR

Multi-frame OCR solution for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition**.

This repository implements a **5-frame license plate recognizer** with:
- Modern **SVTRv2-based backbone** (`MultiFrameSVTRv2`)
- Optional **Spatial Transformer Network (STN)**
- Optional **Super-Resolution (MF‑LPR / LP‑Diff)** pre-processing

🔗 **Challenge:** `https://icpr26lrlpr.github.io/`

---

## Quick Start

### 1. Environment

```bash
# Activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies (uv)
uv sync
```

### 2. Train baseline OCR (không SR, dùng MultiFrameSVTRv2 + STN)

```bash
python train.py \
  --model mf_svtrv2 \
  --experiment-name mfsvtrv2_baseline
```

> Mặc định `MODEL_TYPE = "mf_svtrv2"` trong `configs/config.py`, nên có thể chỉ cần:
> ```bash
> python train.py
> ```

### 3. Bật Super-Resolution (MF‑LPR / LP‑Diff)

Giả sử bạn đã có checkpoint SR:

```python
# configs/config.py
USE_SR: bool = True
SR_CHECKPOINT_PATH: str = r"weights/sr/mf_lpr_sr_best.pth"
SR_CONFIG_PATH: str = "sr_model/config/LP-Diff.json"
```

Chạy:

```bash
python train.py --batch-size 32
```

Hoặc override từ CLI:

```bash
python train.py \
  --use-sr \
  --sr-checkpoint-path "weights/sr/mf_lpr_sr_best.pth"
```

### 4. Tạo submission cho test set

```bash
python train.py \
  --submission-mode \
  --experiment-name mfsvtrv2_submit
```

Kết quả: `results/submission_mfsvtrv2_submit_final.txt`

---

## Key Features

- **Multi-Frame Fusion (5 frames)**: Input shape `(B, 5, 3, 32, 128)` với attention fusion.
- **MultiFrameSVTRv2 (mặc định)**:
  - Backbone SVTRv2-LNConvTwo33
  - Attention fusion cho 5 frame
  - CTC head cho sequence decoding.
- **STN (tùy chọn)**: Căn chỉnh biển số trước khi đưa vào backbone.
- **Super-Resolution (MF‑LPR / LP‑Diff, tùy chọn)**:
  - Pretrained diffusion SR model, chạy **trước** OCR.
  - Tích hợp qua adapter `src/sr/mf_lpr_sr.py`.
- **Scenario-aware splitting**:
  - Split train/val ưu tiên track từ Scenario-B, lưu vào `data/val_tracks.json`.
- **Training utilities**:
  - Mixed precision (`torch.amp`), gradient clipping, OneCycleLR, focal CTC loss (tùy chọn).

---

## Model Architectures

### 1. MultiFrameSVTRv2 (mặc định, tốt nhất)

**Pipeline:**  
`5× LR Frames → (optional STN) → SVTRv2 Backbone → Attention Fusion → CTC Head`

- Định nghĩa trong `src/mf_svtrv2.py`
- Sử dụng cấu hình từ `configs/config.py` (`SVTR_DIMS`, `SVTR_DEPTHS`, `SVTR_HEADS`).
- Pretrained weights (ví dụ UniRec) load từ `Config.PRETRAINED_PATH`.

### 2. ResTranOCR

**Pipeline:**  
`5× Frames → (optional STN) → ResNet34 → Attention Fusion → Transformer Encoder → CTC`

- Định nghĩa trong `src/models/restran.py`.
- Chọn bằng `--model restran`.

### 3. MultiFrameCRNN

**Pipeline:**  
`5× Frames → (optional STN) → CNN → Attention Fusion → BiLSTM → CTC`

- Định nghĩa trong `src/models/crnn.py`.
- Chọn bằng `--model crnn`.

---

## Installation

**Yêu cầu:**
- Python 3.11+
- GPU hỗ trợ CUDA (khuyến nghị)

### Bằng uv (khuyến nghị)

```bash
git clone <repo_url>
cd MultiFrame-LPR
uv sync
```

### Bằng pip (nếu không dùng uv)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python matplotlib numpy pandas tqdm
```

---

## Data Preparation

Cấu trúc thư mục:

```text
data/train/
├── Scenario-A/
│   └── Brazilian/
│       ├── track_00001/
│       │   ├── lr-0.png
│       │   ├── lr-1.png
│       │   ├── ...
│       │   ├── hr-0.png (tùy chọn, dùng cho synthetic LR)
│       │   └── annotations.json
│       └── track_00002/
│           └── ...
└── Scenario-B/
    └── ...
```

`annotations.json` (tối thiểu):

```json
{"plate_text": "ABC1234"}
```

- Train/val split được tạo và lưu vào `data/val_tracks.json` (Scenario-B aware).
- Test public (nếu có) đặt trong `data/public_test/` cùng cấu trúc track nhưng **không có annotations**.

---

## Training Usage

### Basic training (MultiFrameSVTRv2 + STN, không SR)

```bash
python train.py \
  --model mf_svtrv2 \
  --experiment-name mfsvtrv2_baseline
```

### Training với cấu hình tuỳ chỉnh

```bash
python train.py \
  --model mf_svtrv2 \
  --experiment-name my_exp \
  --data-root data/train \
  --batch-size 64 \
  --epochs 30 \
  --lr 3.25e-4 \
  --aug-level full
```

### Tắt STN

```bash
python train.py --no-stn
```

### Bật Super-Resolution (nếu đã có checkpoint SR)

```bash
python train.py \
  --use-sr \
  --sr-checkpoint-path "weights/sr/mf_lpr_sr_best.pth"
```

### Các tham số CLI quan trọng

- `-m, --model`: `crnn`, `restran`, `mf_svtrv2` (default: `mf_svtrv2`)
- `-n, --experiment-name`: tên thí nghiệm, dùng để đặt tên file output
- `--data-root`: thư mục train (default: `data/train`)
- `--batch-size`: batch size (default: `64`)
- `--epochs`: số epoch (default: từ `Config.EPOCHS`)
- `--lr, --learning-rate`: learning rate (default: `Config.LEARNING_RATE`)
- `--aug-level`: `full` hoặc `light`
- `--no-stn`: tắt STN
- `--submission-mode`: train trên full data và tạo submission cho test
- `--output-dir`: thư mục lưu kết quả (default: `results/`)
- `--use-sr`: bật Super-Resolution MF‑LPR
- `--sr-checkpoint-path`: đường dẫn checkpoint GEN của SR
- `--sr-config-path`: đường dẫn JSON config SR (default: `sr_model/config/LP-Diff.json`)

---

## Super-Resolution Integration (MF‑LPR / LP‑Diff)

### Adapter: `src/sr/mf_lpr_sr.py`

```python
from src.sr import MF_LPR_SR

sr = MF_LPR_SR(
    checkpoint_path="weights/sr/mf_lpr_sr_best.pth",
    config_path="sr_model/config/LP-Diff.json",
    device=config.DEVICE,
)

sr_frames = sr.enhance_sequence(frames, resize_to=(32, 128))
```

- Adapter wrap lại code gốc trong `sr_model/` (UNet + GaussianDiffusion + MTA).
- Chỉ dùng cho **inference**; SR model được pretrained và đóng băng, **không train cùng OCR**.

### Bật/tắt SR trong pipeline

Trong `configs/config.py`:

```python
USE_SR: bool = True  # hoặc False
SR_CHECKPOINT_PATH: str = r"weights/sr/mf_lpr_sr_best.pth"
SR_CONFIG_PATH: str = "sr_model/config/LP-Diff.json"
```

Trong `train.py`, nếu `USE_SR=True` và `SR_CHECKPOINT_PATH` hợp lệ:
- Khởi tạo `MF_LPR_SR`.
- Truyền `sr_enhancer` vào `MultiFrameDataset`.
- Mỗi sample (5 frame) được SR trước khi vào model OCR.

⚠️ **Lưu ý runtime:**  
Diffusion SR rất nặng (1000 bước / lần). Để train thực tế:
- Nên giảm `n_timestep` trong config SR (ví dụ 50–100).
- Hoặc chỉ SR frame giữa.
- Hoặc precompute ảnh SR offline rồi train OCR trên ảnh SR đã lưu.

---

## Outputs

Tất cả file output đều được lưu trong `OUTPUT_DIR` (default: `results/`).

### Checkpoints

- **`{EXPERIMENT_NAME}_best.pth`**
  - Model tốt nhất theo **Val Accuracy** (luôn có).
- **`{EXPERIMENT_NAME}_final.pth`**
  - Trọng số model **epoch cuối** (chỉ khi có validation; không có trong `--submission-mode`).

### Submission files

- **`submission_{EXPERIMENT_NAME}.txt`**
  - Dự đoán trên **validation set** mỗi khi có best mới.
  - Format: `track_id,pred_text;confidence`.
- **`submission_{EXPERIMENT_NAME}_final.txt`**
  - Chỉ khi `--submission-mode` và có test data.
  - Dự đoán cho **test set** để nộp bài.

### Wrong predictions

- **`wrong_predictions_{EXPERIMENT_NAME}.txt`**
  - Sinh ra khi:
    - Có validation
    - `SAVE_WRONG_PREDICTIONS=True` (default)
    - Có ít nhất 1 sample sai
  - Format:
    ```text
    track_id    ground_truth    prediction    confidence    img_paths
    track_00042 ABC1234         ABC1235      0.8234        data/.../track_00042/lr-0.png;...;lr-4.png
    ```
  - `img_paths` giúp bạn mở nhanh đúng 5 ảnh của sample bị sai.

---

## Configuration (tóm tắt)

Các hyperparameter chính trong `configs/config.py`:

```python
MODEL_TYPE: str = "mf_svtrv2"   # "crnn", "restran", "mf_svtrv2"
EXPERIMENT_NAME: str = MODEL_TYPE
AUGMENTATION_LEVEL: str = "full"   # "full" hoặc "light"
USE_STN: bool = True

DATA_ROOT: str = "data/train"
TEST_DATA_ROOT: str = "data/public_test"

BATCH_SIZE: int = 64
LEARNING_RATE: float = 3.25e-4
EPOCHS: int = 1      # chỉnh trong config hoặc override bằng CLI

USE_FOCAL_CTC: bool = True
CTC_BEAM_WIDTH: int = 1

PRETRAINED_PATH: str = r"weights/best.pth"   # cho mf_svtrv2 / restran / crnn

USE_SR: bool = False                         # bật/tắt SR
SR_CHECKPOINT_PATH: str = ""                 # cần set nếu USE_SR=True
SR_CONFIG_PATH: str = "sr_model/config/LP-Diff.json"
```

Tất cả các field có thể override qua CLI (`arg_to_config` trong `train.py`).

---

## Project Structure

```text
.
├── configs/
│   └── config.py                 # Dataclass cấu hình
├── src/
│   ├── data/
│   │   ├── dataset.py            # MultiFrameDataset (5 frames, scenario-aware split, optional SR)
│   │   └── transforms.py         # Augmentation pipelines
│   ├── models/
│   │   ├── crnn.py               # Multi-frame CRNN
│   │   ├── restran.py            # ResTranOCR
│   │   └── components.py         # STN, AttentionFusion, etc.
│   ├── sr/
│   │   ├── mf_lpr_sr.py          # MF-LPR / LP-Diff adapter (SR)
│   │   └── __init__.py
│   ├── training/
│   │   └── trainer.py            # Training, validation, saving outputs
│   └── utils/
│       ├── common.py             # seed, CUDA utils, memory estimate
│       └── postprocess.py        # CTC decoding, CER
├── sr_model/                     # Original LP-Diff SR implementation
│   ├── config/LP-Diff.json       # SR config
│   ├── model/                    # UNet + GaussianDiffusion + MTA
│   └── ...
├── train.py                      # Main training / submission script
├── run_ablation.py               # Ablation study (không bắt buộc)
├── docs/
│   ├── add_sr.md                 # Hướng dẫn tích hợp SR (chi tiết)
│   ├── CHECKPOINT_PATHS.md       # Hướng dẫn đặt checkpoint
│   ├── HOW_TO_VERIFY_SR.md       # Cách kiểm tra SR đã tích hợp
│   └── OUTPUTS_AFTER_RUN.md      # Giải thích file output
└── pyproject.toml                # Dependencies
```

---

## Notes & Recommendations

- **Khi debug / thử nghiệm nhanh**, nên:
  - Đặt `USE_SR=False` để training nhanh.
  - Dùng subset data + `EPOCHS=1` để kiểm tra pipeline.
- **Khi bật SR để train nghiệm chỉnh**, hãy:
  - Đảm bảo checkpoint SR chạy ổn với `test_sr_integration.py`.
  - Cân nhắc giảm `n_timestep` hoặc chỉ SR frame trung tâm để tiết kiệm thời gian.

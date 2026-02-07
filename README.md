# MultiFrame-LPR

Multi-frame OCR solution for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition (LRLPR)**. This implementation fuses temporal information from 5 video frames using attention-based fusion to achieve robust recognition on low-resolution license plates.

🔗 **Challenge:** [ICPR 2026 LRLPR](https://icpr26lrlpr.github.io/)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Super-Resolution (Optional)](#super-resolution-optional)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [License](#license)

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Desuuy/ICPR_Challenge.git
cd ICPR_Challenge

# Install dependencies (uv recommended)
uv sync

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Train with default model (MF-SVTRv2 + STN)
python train.py

# Train with Super-Resolution
python train.py --use-sr --sr-checkpoint-path weights/I80000_E41_gen_best_psnr.pth

# Generate submission file
python train.py --submission-mode --test-data-root data/public_test
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Frame Fusion** | Processes 5-frame sequences with attention-based fusion |
| **Spatial Transformer Network (STN)** | Optional automatic image alignment for warped plates |
| **Three Architectures** | CRNN (baseline), ResTranOCR, MF-SVTRv2 (default) |
| **Super-Resolution** | Optional LP-Diff integration for degraded input enhancement |
| **Focal CTC** | Sample-level weighting for hard sample focus |
| **Scenario-B Aware Split** | Validation prioritizes challenging scenarios |
| **Mixed Precision** | FP16 training with gradient scaling |
| **Wrong Prediction Analysis** | Saves wrong predictions and copies images for inspection |

---

## Model Architectures

### MF-SVTRv2 (Default)

**Pipeline:** Multi-frame Input → STN (optional) → SVTRv2LNConvTwo33 Backbone → Attention Fusion → RCTC Decoder → CTC

- **Backbone:** SVTRv2-Small (dims=[128,256,384], depths=[6,6,6])
- **Fusion:** Attention-based temporal fusion
- **Decoder:** RCTC (CTC head)
- **Pretrained:** Supports UniRec/OpenRec checkpoint loading

### ResTranOCR

**Pipeline:** Multi-frame Input → STN (optional) → ResNet34 → Attention Fusion → Transformer Encoder → CTC

- **Backbone:** ResNet34 feature extractor
- **Sequence:** Transformer with positional encoding

### CRNN (Baseline)

**Pipeline:** Multi-frame Input → STN (optional) → CNN → Attention Fusion → BiLSTM → CTC

- **Backbone:** Lightweight CNN
- **Sequence:** Bidirectional LSTM

**All models accept input shape:** `(Batch, 5, 3, 32, 128)` and output character sequences via CTC decoding.

---

## Installation

**Requirements:**
- Python 3.11+
- CUDA-enabled GPU (recommended)
- Windows / Linux / macOS

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install albumentations opencv-python matplotlib numpy pandas tqdm
```

---

## Data Preparation

Organize your dataset with the following structure:

```
data/train/
├── track_001/
│   ├── lr-001.png
│   ├── lr-002.png
│   ├── lr-003.png
│   ├── lr-004.png
│   ├── lr-005.png
│   ├── hr-001.png        # Optional: for synthetic LR generation
│   └── annotations.json
└── track_002/
    └── ...
```

**annotations.json format:**
```json
{"plate_text": "ABC1234"}
```
or
```json
{"license_plate": "ABC1234"}
```

**Test data** (for submission): Same structure under `data/public_test/` (or `--test-data-root`).

---

## Usage

### Basic Training

```bash
# Default: MF-SVTRv2, full augmentation, STN enabled
python train.py

# CRNN baseline
python train.py --model crnn --experiment-name crnn_baseline

# ResTranOCR
python train.py --model restran --experiment-name restran_exp
```

### Custom Configuration

```bash
python train.py \
    --model mf_svtrv2 \
    --experiment-name my_exp \
    --data-root /path/to/dataset \
    --batch-size 32 \
    --epochs 30 \
    --lr 0.000325 \
    --aug-level full
```

### Submission Mode

Train on full dataset (no validation split) and generate predictions for test data:

```bash
python train.py --submission-mode --test-data-root data/public_test
```

### Disable STN

```bash
python train.py --no-stn
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-m, --model` | Model type: `crnn`, `restran`, `mf_svtrv2` | `mf_svtrv2` |
| `-n, --experiment-name` | Experiment identifier | from config |
| `--data-root` | Training data path | `data/train` |
| `--test-data-root` | Test data path (submission mode) | `data/public_test` |
| `--batch-size` | Batch size | 64 |
| `--epochs` | Training epochs | from config |
| `--lr` | Learning rate | 0.000325 |
| `--aug-level` | `full` or `light` | `full` |
| `--no-stn` | Disable STN | False |
| `--submission-mode` | Full train + test inference | False |
| `--use-sr` | Enable Super-Resolution | False |
| `--sr-checkpoint-path` | SR checkpoint (GEN) path | from config |
| `--sr-config-path` | SR config JSON path | `sr_model/config/LP-Diff.json` |
| `--output-dir` | Output directory | `results/` |

---

## Super-Resolution (Optional)

Optional integration with **LP-Diff** (MF-LPR SR) to enhance low-resolution frames before OCR:

1. Place LP-Diff code in `sr_model/` (see [LP-Diff](https://github.com/haoyGONG/LP-Diff))
2. Download GEN checkpoint (e.g. `I80000_E41_gen_best_psnr.pth`)
3. Run with SR:

```bash
python train.py --use-sr --sr-checkpoint-path weights/I80000_E41_gen_best_psnr.pth
```

SR is applied per-sample in the dataset before augmentation/transform.

---

## Configuration

Key hyperparameters in `configs/config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL_TYPE` | `crnn` / `restran` / `mf_svtrv2` | `mf_svtrv2` |
| `USE_STN` | Enable STN | `True` |
| `USE_FOCAL_CTC` | Focal-style CTC loss | `True` |
| `SAVE_WRONG_PREDICTIONS` | Save wrong predictions file | `True` |
| `SAVE_WRONG_IMAGES` | Copy wrong images to results | `True` |
| `USE_SR` | Enable Super-Resolution | `False` |
| `IMG_HEIGHT`, `IMG_WIDTH` | Input size | 32, 128 |
| `PRETRAINED_PATH` | OCR pretrained checkpoint | `weights/best.pth` |

All parameters can be overridden via CLI.

---

## Outputs

After training, outputs are saved in `results/` (or `--output-dir`):

| File | Description |
|------|-------------|
| `{exp}_best.pth` | Best model checkpoint (by val accuracy) |
| `{exp}_final.pth` | Final epoch checkpoint (normal mode) |
| `submission_{exp}.txt` | Predictions: `track_id,pred;confidence` |
| `wrong_predictions_{exp}.txt` | Wrong predictions with gt, pred, img_paths |
| `wrong_images_{exp}/` | Copied images for wrong predictions |

---

## Project Structure

```
.
├── configs/
│   └── config.py              # Configuration
├── src/
│   ├── data/
│   │   ├── dataset.py         # MultiFrameDataset
│   │   └── transforms.py     # Augmentation
│   ├── models/
│   │   ├── crnn.py
│   │   ├── restran.py
│   │   └── components.py     # STN, AttentionFusion
│   ├── sr/
│   │   └── mf_lpr_sr.py      # SR adapter (LP-Diff)
│   ├── training/
│   │   └── trainer.py
│   ├── utils/
│   └── mf_svtrv2.py          # MF-SVTRv2 model
├── sr_model/                  # LP-Diff (optional)
├── train.py
├── run_ablation.py
└── pyproject.toml
```

---

## License

See repository license.

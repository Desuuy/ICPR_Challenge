## MultiFrame-LPR

MultiFrame-LPR is a multi-frame OCR system for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition**.  
It processes short sequences of low-resolution license plate frames and fuses temporal information to produce robust text predictions.

The current pipeline consists of:
- A 5-frame OCR backbone based on **SVTRv2** (`MultiFrameSVTRv2`)
- Optional **Spatial Transformer Network (STN)** for alignment
- Optional **Super-Resolution (MF-LPR / LP-Diff)** as a pre-processing stage

Challenge website: `https://icpr26lrlpr.github.io/`

---

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Training Usage](#training-usage)
- [Super-Resolution Integration (MF-LPR / LP-Diff)](#super-resolution-integration-mf-lpr--lp-diff)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Practical Notes](#practical-notes)

---

## Installation

### Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)

### Using `uv` (recommended)

```bash
git clone <this-repo-url>
cd MultiFrame-LPR
uv sync
```

### Using `pip`

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python matplotlib numpy pandas tqdm
```

---

## Data Preparation

The training data is expected under `data/train` with the following structure (simplified):

```text
data/train/
├── Scenario-A/
│   └── Brazilian/
│       ├── track_00001/
│       │   ├── lr-0.png
│       │   ├── lr-1.png
│       │   ├── ...
│       │   ├── hr-0.png              # optional, for synthetic LR generation
│       │   └── annotations.json
│       └── track_00002/
│           └── ...
└── Scenario-B/
    └── ...
```

The minimal `annotations.json` format is:

```json
{"plate_text": "ABC1234"}
```

Notes:
- Train/validation splitting is scenario-aware and stored in `data/val_tracks.json`.
- If a public test set is provided, it should follow the same `track_*` structure under `data/public_test/` but without annotations.

---

## Model Architectures

### 1. MultiFrameSVTRv2 (default and recommended)

Pipeline:

```text
5 × LR frames → (optional STN) → SVTRv2 backbone → attention-based temporal fusion → CTC head
```

- Implemented in `src/mf_svtrv2.py`.
- Controlled via `configs/config.py` (`SVTR_DIMS`, `SVTR_DEPTHS`, `SVTR_HEADS`).
- Can load pretrained weights via `Config.PRETRAINED_PATH` (e.g., UniRec-based checkpoints).

### 2. ResTranOCR

Pipeline:

```text
5 × frames → (optional STN) → ResNet34 backbone → attention fusion → Transformer encoder → CTC
```

- Implemented in `src/models/restran.py`.
- Select via `--model restran`.

### 3. MultiFrameCRNN

Pipeline:

```text
5 × frames → (optional STN) → CNN backbone → attention fusion → BiLSTM → CTC
```

- Implemented in `src/models/crnn.py`.
- Select via `--model crnn`.

All models operate on input tensors of shape `(batch, 5, 3, 32, 128)` and produce sequences decoded by CTC.

---

## Training Usage

The main entry point is `train.py`.

### Basic training (MultiFrameSVTRv2 + STN, no SR)

```bash
python train.py \
  --model mf_svtrv2 \
  --experiment-name mfsvtrv2_baseline
```

Since `MODEL_TYPE = "mf_svtrv2"` in `configs/config.py`, the simplest case is:

```bash
python train.py
```

### Custom configuration

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

### Disable STN

```bash
python train.py --no-stn
```

### Generate submission for test set

```bash
python train.py \
  --submission-mode \
  --experiment-name mfsvtrv2_submit
```

This will create: `results/submission_mfsvtrv2_submit_final.txt`.

### Key command-line arguments

- `-m, --model`: model type, one of `crnn`, `restran`, `mf_svtrv2` (default: `mf_svtrv2`)
- `-n, --experiment-name`: experiment identifier used in output file names
- `--data-root`: path to training data (default: `data/train`)
- `--batch-size`: batch size (default: `Config.BATCH_SIZE`)
- `--epochs`: number of epochs (default: `Config.EPOCHS`)
- `--lr, --learning-rate`: learning rate (default: `Config.LEARNING_RATE`)
- `--aug-level`: `"full"` or `"light"`
- `--no-stn`: disable STN
- `--submission-mode`: train on the full dataset and then run inference on the test set
- `--output-dir`: directory for outputs (default: `results/`)
- `--use-sr`: enable MF-LPR Super-Resolution in the data pipeline
- `--sr-checkpoint-path`: path to the MF-LPR generator checkpoint (`*_gen*.pth`)
- `--sr-config-path`: path to the SR JSON config (default: `sr_model/config/LP-Diff.json`)

---

## Super-Resolution Integration (MF-LPR / LP-Diff)

Super-resolution is integrated as an **optional pre-processing step** before the OCR backbone.

### Adapter: `src/sr/mf_lpr_sr.py`

The adapter wraps the original LP-Diff implementation from `sr_model/` and exposes a simple API:

```python
from src.sr import MF_LPR_SR
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sr = MF_LPR_SR(
    checkpoint_path="weights/sr/mf_lpr_sr_best.pth",
    config_path="sr_model/config/LP-Diff.json",
    device=device,
)

# frames: (T, C, H, W) normalized to [-1, 1]
sr_frames = sr.enhance_sequence(frames, resize_to=(32, 128))
```

Key points:
- Uses the original UNet + GaussianDiffusion + MTA from `sr_model/`.
- Intended for **inference only**; SR weights are pretrained and frozen (not updated during OCR training).

### Enabling/disabling SR

In `configs/config.py`:

```python
USE_SR: bool = True  # or False
SR_CHECKPOINT_PATH: str = r"weights/sr/mf_lpr_sr_best.pth"
SR_CONFIG_PATH: str = "sr_model/config/LP-Diff.json"
```

In `train.py`, when `USE_SR=True` and `SR_CHECKPOINT_PATH` is valid:
- An `MF_LPR_SR` instance is created.
- This instance is passed into `MultiFrameDataset` as `sr_enhancer`.
- Each 5-frame sample is optionally super-resolved before feeding into the OCR model.

### Runtime considerations

LP-Diff is a diffusion-based SR model with `n_timestep = 1000` by default.  
This is computationally expensive:
- Each SR call runs a 1000-step sampling loop over the batch.
- In the naive setup, this can make training prohibitively slow.

For practical use:
- Consider reducing `n_timestep` in the SR config (e.g., 50–100 for inference).
- Consider applying SR only to the central frame of the sequence rather than all 5 frames.
- For large-scale training, precomputing SR images offline and training OCR on those images is often the most efficient approach.

---

## Outputs

All outputs are written under `OUTPUT_DIR` (default: `results/`).

### Checkpoints

- `{EXPERIMENT_NAME}_best.pth`
  - Best model checkpoint according to validation accuracy (always saved).
- `{EXPERIMENT_NAME}_final.pth`
  - Final model weights at the end of training (only when validation is used; not in `--submission-mode`).

### Submission files

- `submission_{EXPERIMENT_NAME}.txt`
  - Predictions on the validation set whenever a new best model is found.
  - Format: `track_id,predicted_text;confidence`.
- `submission_{EXPERIMENT_NAME}_final.txt`
  - Only in `--submission-mode` when test data is available.
  - Predictions for the test set in the competition format.

### Wrong predictions

- `wrong_predictions_{EXPERIMENT_NAME}.txt`
  - Generated when:
    - Validation is enabled.
    - `SAVE_WRONG_PREDICTIONS = True` (default).
    - At least one misclassified sample exists.
  - Format (tab-separated):

    ```text
    track_id    ground_truth    prediction    confidence    img_paths
    track_00042 ABC1234         ABC1235      0.8234        data/.../track_00042/lr-0.png;...;lr-4.png
    ```

  - The `img_paths` field contains the list of 5 LR frame paths for that sample, which is convenient for manual inspection.

---

## Configuration

The main configuration is defined in `configs/config.py` as a dataclass `Config`.  
Important fields include:

```python
MODEL_TYPE: str = "mf_svtrv2"        # "crnn", "restran", or "mf_svtrv2"
EXPERIMENT_NAME: str = MODEL_TYPE
AUGMENTATION_LEVEL: str = "full"     # "full" or "light"
USE_STN: bool = True

DATA_ROOT: str = "data/train"
TEST_DATA_ROOT: str = "data/public_test"

BATCH_SIZE: int = 64
LEARNING_RATE: float = 3.25e-4
EPOCHS: int = 1

USE_FOCAL_CTC: bool = True
CTC_BEAM_WIDTH: int = 1

PRETRAINED_PATH: str = r"weights/best.pth"   # OCR pretrained weights (if available)

USE_SR: bool = False                         # enable/disable SR
SR_CHECKPOINT_PATH: str = ""                 # must be set if USE_SR=True
SR_CONFIG_PATH: str = "sr_model/config/LP-Diff.json"
```

Most of these fields can be overridden via CLI arguments in `train.py`  
(see `arg_to_config` mapping).

---

## Project Structure

```text
.
├── configs/
│   └── config.py                 # Configuration dataclass
├── src/
│   ├── data/
│   │   ├── dataset.py            # MultiFrameDataset (5 frames, scenario-aware split, optional SR)
│   │   └── transforms.py         # Augmentation pipelines
│   ├── models/
│   │   ├── crnn.py               # Multi-frame CRNN
│   │   ├── restran.py            # ResTranOCR
│   │   └── components.py         # STN, AttentionFusion, etc.
│   ├── sr/
│   │   ├── mf_lpr_sr.py          # MF-LPR / LP-Diff SR adapter
│   │   └── __init__.py
│   ├── training/
│   │   └── trainer.py            # Training loop, validation, saving outputs
│   └── utils/
│       ├── common.py             # Seeding, CUDA utilities, memory estimation
│       └── postprocess.py        # CTC decoding, CER computation
├── sr_model/                     # Original LP-Diff SR implementation
│   ├── config/LP-Diff.json       # SR configuration
│   ├── model/                    # UNet + GaussianDiffusion + MTA
│   └── ...
├── train.py                      # Main training / submission script
├── run_ablation.py               # Ablation study automation (optional)
├── docs/
│   ├── add_sr.md                 # Detailed SR integration guide
│   ├── CHECKPOINT_PATHS.md       # Checkpoint placement and configuration
│   ├── HOW_TO_VERIFY_SR.md       # How to verify SR integration
│   └── OUTPUTS_AFTER_RUN.md      # Explanation of output files
└── pyproject.toml                # Dependencies
```

---

## Practical Notes

- For quick debugging or hyperparameter tuning:
  - Set `USE_SR = False` to avoid SR overhead.
  - Use a small subset of the data and `EPOCHS = 1`.

- When enabling SR for serious experiments:
  - Verify the SR checkpoint with `test_sr_integration.py`.
  - Consider reducing diffusion steps or SR coverage (e.g., only the central frame) to keep runtime manageable.

Contributions, issues, and pull requests are welcome. Please open a GitHub issue if you encounter problems with the integration or documentation.

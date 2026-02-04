<p align="center">
  <a href="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/results_with_word.png">
    <img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/results_with_word.png" width="80%">
  </a>
</p>

<h1 align="center">LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate</h1>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.html"><img src="https://img.shields.io/badge/Paper-CVPR%202025-blue.svg"></a>
  <a href="https://icpr26lrlpr.github.io/"><img src="https://img.shields.io/badge/Challenge-ICPR%202026-green.svg"></a>
</p>

<p align="center">
  <b>Customized implementation for ICPR 2026 Challenge on Low-Resolution License Plate Recognition</b><br>
  Based on the original LP-Diff model by Haoyan Gong et al. (CVPR 2025)
</p>

---

## 📝 About This Repository

This is a **customized implementation** of the LP-Diff model adapted for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition**. While the original LP-Diff was trained on the MDLP dataset (11,006 groups), this version is configured for the ICPR 2026 competition dataset with modified training pipeline and baseline-compatible data splitting.

**Original Paper:** [LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate](https://openaccess.thecvf.com/content/CVPR2025/html/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.html) (CVPR 2025)

**Challenge Link:** [ICPR 2026 LRLPR](https://icpr26lrlpr.github.io/)

---

## 🔥 Highlights

- **[Customized for ICPR 2026]**: Adapted training pipeline for competition dataset format
- **[Baseline Split Compatible]**: Uses same train/val split as baseline OCR models for fair comparison
- **[Diffusion-based SR Model]**: Leverages LP-Diff's state-of-the-art architecture
- **[Multi-Frame Input]**: Processes 3 consecutive LR frames to generate 1 SR output
- **[Production Ready]**: Optimized training configs for different GPU memory constraints

---

## 🌟 Original LP-Diff Performance

**On MDLP dataset (original paper):**

|   Method    |  PSNR ↑   |  SSIM ↑   |  FID ↓   |  LPIPS ↓  |   NED ↓   |   ACC ↑   |
| :---------: | :-------: | :-------: | :------: | :-------: | :-------: | :-------: |
|    SRCNN    |   14.01   |   0.195   |  248.3   |   0.517   |   0.626   |   0.041   |
|     HAT     |   14.16   |   0.250   |  229.6   |   0.413   |   0.613   |   0.050   |
| Real-ESRGAN |   13.93   |   0.369   |   31.0   |   0.176   |   0.279   |   0.161   |
|   ResDiff   |   12.00   |   0.269   |   35.9   |   0.277   |   0.292   |   0.159   |
|  ResShift   |   12.53   |   0.321   |   89.1   |   0.288   |   0.332   |   0.099   |
| **LP-Diff** | **14.40** | **0.393** | **22.0** | **0.159** | **0.198** | **0.305** |

---

## 📚 Dataset

### ICPR 2026 Competition Dataset

This implementation is designed for the **ICPR 2026 Challenge dataset**, which includes:

- **Multi-frame sequences**: 5 LR frames per track
- **Real-world conditions**: Various distances, lighting, weather
- **Paired data structure**: LR frames + HR ground truth + annotations

**Dataset Structure:**
```
data/train/
├── Scenario-A/
│   ├── Brazilian/
│   │   ├── track_xxxxx/
│   │   │   ├── lr-001.png
│   │   │   ├── lr-002.png
│   │   │   ├── lr-003.png
│   │   │   ├── lr-004.png
│   │   │   ├── lr-005.png
│   │   │   ├── hr-001.png
│   │   │   ├── hr-002.png
│   │   │   ├── ...
│   │   │   └── annotations.json
│   │   └── ...
│   └── Mercosur/
│       └── ...
└── Scenario-B/
    └── ...
```

**Annotations Format:**
```json
{"plate_text": "ABC1234"}
```

### Original MDLP Dataset

The original LP-Diff model was trained on the **MDLP Dataset** (11,006 groups). If you want to use the original dataset:

- [Google Drive](https://drive.google.com/file/d/1UpECGcWcF92z-P6pJ9couzGTXb1TMHqk/view?usp=sharing)
- [Baidu Netdisk (access code: 1ebm)](https://pan.baidu.com/s/1Aphb_jIx_0tRR71BBbwVwA?pwd=1ebm)

---

## 🚀 Getting Started

### 1. Installation

**Requirements:**
- Python 3.10+
- CUDA-enabled GPU (16GB+ recommended)
- Docker (optional, for containerized training)

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Or using Docker:**
```bash
docker build -t lp-diff .
docker run --gpus all -it --rm \
  -v /path/to/dataset:/data/train \
  -v /path/to/code:/opt/program \
  lp-diff /bin/bash
```

---

## 🎯 Training

### Quick Start

**Training with default config:**
```bash
python run.py -p train -c ./config/LP-Diff.json
```

**Validation:**
```bash
python run.py -p val -c ./config/LP-Diff.json
```
---

## 🏗️ Model Architecture

LP-Diff consists of four key components:

- **ICAM**: Inter-frame Cross Attention Module (multi-frame fusion)
- **TEM**: Texture Enhancement Module (fine detail recovery)
- **DFM**: Dual-Pathway Fusion Module (channel/spatial selection)
- **RCDM**: Residual Condition Diffusion Module (diffusion process)

<img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/backbone_v2.png" width="100%"/>

**Input:** 3 consecutive LR frames (224×112)  
**Output:** 1 SR frame (224×112 or higher resolution)  
**Process:** Sliding window over 5-frame sequences generates 3 SR outputs per track

---

## 📂 Project Structure

```
LP-Diff/
│
├── config/              # Training and testing config files
├── data/                # Data loading scripts
├── experiments/         # Model checkpoints and logs
├── figs/                # Visualization images for README and paper
├── models/              # Model implementations
├── requirements.txt     # Python dependencies
└── run.py               # Main training/testing script
```

---

## 📖 Citation

### This Implementation

If you use this customized implementation, please cite both the original LP-Diff paper and acknowledge the ICPR 2026 challenge:

```bibtex
@inproceedings{gong2025lp,
  title={LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate},
  author={Gong, Haoyan and Zhang, Zhenrong and Feng, Yuzheng and Nguyen, Anh and Liu, Hongbin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17831--17840},
  year={2025}
}

@misc{icpr2026lrlpr,
  title={ICPR 2026 Challenge on Low-Resolution License Plate Recognition},
  howpublished={\url{https://icpr26lrlpr.github.io/}},
  year={2026}
}
```

---

## 🤝 Acknowledgements

- **Original LP-Diff:** [Haoyan Gong et al. (CVPR 2025)](https://github.com/haoyGONG/LP-Diff)
- **Base Framework:** [ResDiff](https://github.com/LYL1015/ResDiff/tree/master)
- **Competition:** [ICPR 2026 LRLPR Challenge](https://icpr26lrlpr.github.io/)

---

## 💬 Contact

For questions about this implementation:
- Open an issue in this repository
- For original LP-Diff questions: [m.g.haoyan@gmail.com](mailto:m.g.haoyan@gmail.com)

---

## 📝 License

This project follows the same license as the original LP-Diff implementation. Please refer to the original repository for license details.

---

## 🔗 Related Resources

- **Original LP-Diff GitHub:** [https://github.com/haoyGONG/LP-Diff](https://github.com/haoyGONG/LP-Diff)
- **ICPR 2026 Challenge:** [https://icpr26lrlpr.github.io/](https://icpr26lrlpr.github.io/)
- **Original Paper:** [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.html)

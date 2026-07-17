# PPE Detection for Warehouse CCTV

A computer vision system for detecting Personal Protective Equipment (PPE) on workers using YOLO11 models. Designed for deployment on edge hardware (Jetson, Raspberry Pi) for real-time CCTV monitoring.

## Overview

**Objective:** Detect PPE compliance on warehouse CCTV footage in real-time
- **Classes:** Hardhat, No Hardhat, Safety Vest, No Safety Vest, Safety Shoes, No Safety Shoes, Person
- **Target Hardware:** Edge devices (GPU-accelerated preferred)
- **Model Architecture:** YOLOv11 (nano to medium variants) (soon YOLOv12, YOLO26, RF-DETR)
- **Status:** Pre-training benchmark phase

---

## Project Structure

```
ppe-detection/
├── config/                          # Configuration files for different model experiments
│   ├── config.yaml                 # Base configuration (YOLOv11n)
│   ├── config_v2.yaml              # [Optional] Alternative configurations
│   └── ...
├── src/
│   └── train.py                    # Main training script (handles sequential model training)
├── data/                           # Dataset directory (auto-created by Roboflow)
│   └── PPR_EXT.v2-raw.yolov11/
│       ├── images/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       ├── labels/
│       └── data.yaml
├── runs/                           # Training outputs (auto-created)
│   └── detect/
│       ├── ppe_v1/
│       ├── ppe_v2/
│       └── ...
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- pip

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ppe-detection
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify GPU access:**
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## Dataset

**Source:** Roboflow (PPR_EXT.v2-raw)
**Link:** https://universe.roboflow.com/ppev6/ppr_ext/dataset/1

**Classes (7 total):**
| Class | Description |
|-------|-------------|
| person | Worker/human |
| hardhat | Wearing hardhat ✓ |
| no_hardhat | Not wearing hardhat ✗ |
| safety_vest | Wearing safety vest ✓ |
| no_safety_vest | Not wearing safety vest ✗ |
| safety_shoes | Wearing safety shoes ✓ |
| no_safety_shoes | Not wearing safety shoes ✗ |

**Data Split:**
- Train: ~70%
- Validation: ~20%
- Test: ~10%

---

## Configuration

All training hyperparameters are controlled via YAML config files in the `config/` folder.

### Key Settings

```yaml
# Model selection
model: yolo11n.pt          # Options: yolo11n/s/m/l, yolo12n/s/m/l, etc.
task: detect               # Detection task

# Dataset
data: data/PPR_EXT.v2-raw.yolov11/data.yaml

# Training
epochs: 20                 # Number of training epochs
batch: 16                  # Batch size (adjust based on VRAM)
imgsz: 640                 # Input image size (pixels)
patience: 3                # Early stopping patience (epochs)

# Optimizer
optimizer: AdamW           # Options: AdamW, SGD
lr0: 0.01                  # Initial learning rate
lrf: 0.01                  # Final LR as fraction of lr0

# Augmentation (toggle based on dataset pre-augmentation)
use_augmentation: false    # Set true for on-the-fly augmentation
hsv_h: 0.015              # Hue augmentation
hsv_s: 0.7                # Saturation augmentation
hsv_v: 0.4                # Value (brightness) augmentation
fliplr: 0.5               # Horizontal flip probability
scale: 0.2                # Random scale ±20%
mosaic: 1.0               # Mosaic augmentation (4 images combined)

# Output
project: ./runs            # Results output directory
name: ppe_v1               # Run name (creates ./runs/detect/ppe_v1/)
exist_ok: false            # Auto-version if run exists
```

**Quick Start:** Use the default `config/config.yaml` for baseline training.

---

## Training

### Single Model Training

```bash
cd src
python train.py
```

This trains the first model in the `MODELS` array (see Sequential Training below).

### Sequential Model Training (Benchmarking)

To train multiple models back-to-back for benchmarking:

1. **Edit `src/train.py` (lines 14-18):**
```python
MODELS = [
    "config/config.yaml",          # YOLOv11n (nano)
    "config/config_v11s.yaml",     # YOLOv11s (small)
    "config/config_v11m.yaml",     # YOLOv11m (medium)
    "config/config_v12n.yaml",     # YOLOv12n
]
```

2. **Run training:**
```bash
python src/train.py
```

3. **Monitor progress:**
```
================================================================================
              STARTING SEQUENTIAL TRAINING - 4 models
================================================================================

[1/4] Training: config/config.yaml
   Model     : yolo11n.pt
   Dataset   : data/PPR_EXT.v2-raw.yolov11/data.yaml
   ...
   Duration  : 0h 45m 30s ✓

[2/4] Training: config/config_v11s.yaml
   ...

[3/4] Training: config/config_v11m.yaml
   ...

[4/4] Training: config/config_v12n.yaml
   ...

================================================================================
                         TRAINING COMPLETE
================================================================================
Models trained: 4/4
Total time: 3h 12m 45s

  ✓ config/config.yaml (0h 45m 30s)
  ✓ config/config_v11s.yaml (0h 48m 15s)
  ✓ config/config_v11m.yaml (0h 51m 20s)
  ✓ config/config_v12n.yaml (0h 47m 40s)
```

**Optional:** Skip failed models and continue:
```python
CONTINUE_ON_ERROR = True  # Line 21 in train.py
```

---

## Model Benchmarking

Results are saved in `./runs/detect/` with automatic versioning:

```
runs/detect/
├── ppe_v1/                # YOLOv11n results
│   ├── weights/
│   │   ├── best.pt       # Best model (highest validation mAP)
│   │   └── last.pt       # Final epoch weights
│   ├── results.csv       # Per-epoch metrics
│   ├── confusion_matrix.png
│   └── plots/
│       ├── results.png   # Loss/mAP curves
│       └── ...
├── ppe_v2/                # YOLOv11s results
└── ...
```

### Evaluation Metrics

After training completes, evaluate on test set:

```bash
yolo val model=runs/detect/ppe_v1/weights/best.pt data=data/PPR_EXT.v2-raw.yolov11/data.yaml
```

**Benchmark Criteria:**
- **mAP50** — Accuracy (IoU=0.5 threshold)
- **mAP50-95** — Standard COCO metric
- **Precision/Recall** — Per-class performance
- **Inference Speed** — FPS on target hardware
- **Model Size** — MB (for edge deployment)
- **Memory Usage** — GB VRAM/RAM required

---

## Edge Deployment

### Export to ONNX (for broader hardware support)

```bash
yolo export model=runs/detect/ppe_v1/weights/best.pt format=onnx imgsz=640
```

### Export to TensorRT (for Jetson)

```bash
yolo export model=runs/detect/ppe_v1/weights/best.pt format=engine imgsz=640 device=0
```

### Inference on Video Stream

```bash
yolo predict model=runs/detect/ppe_v1/weights/best.pt source=video.mp4 conf=0.5 imgsz=640
```

---

## GPU Requirements

| Model | VRAM | Inference Speed (RTX 4050) | Recommended Hardware |
|-------|------|---------------------------|----------------------|
| YOLOv11n | 2GB | ~30 FPS | Jetson Nano, Pi 5 (with acceleration) |
| YOLOv11s | 4GB | ~20 FPS | Jetson Orin Nano |
| YOLOv11m | 6GB | ~12 FPS | Jetson Orin |
| YOLOv11l | 8GB | ~8 FPS | Jetson Orin 64GB |

---

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `batch` size in config (e.g., 16 → 8)
- Reduce `imgsz` (e.g., 640 → 512)
- Use smaller model (e.g., nano instead of medium)

**Training not improving:**
- Verify dataset path in config
- Check augmentation settings (enable if disabled)
- Increase `patience` to allow more epochs
- Verify GPU is being used (check VRAM usage)

**Config file not found:**
- Ensure config paths in `MODELS` array are relative to `src/` directory
- Check file exists: `ls config/config.yaml`

---

## Next Steps

1. **Run baseline training** with default config
2. **Evaluate results** on test set
3. **Create alternative configs** for model comparison
4. **Run sequential training** to benchmark different models
5. **Export best model** for edge deployment
6. **Test on real CCTV footage** for domain validation

---

## References

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/models/yolov11/)
- [Roboflow Dataset](https://universe.roboflow.com/ppev6/ppr_ext/dataset/1)
- [YOLO Export Formats](https://docs.ultralytics.com/modes/export/)

---

## License

This project uses PPE detection for safety compliance monitoring. Refer to `LICENSE` for details.

---

**Last Updated:** July 2026
**Project Status:** Active Development

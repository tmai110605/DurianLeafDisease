# 🌿 DurianLeafProject

A deep learning project for **multi-task classification** of durian leaf diseases — simultaneously predicting the **disease type** and **severity level** from a single leaf image using custom-built lightweight CNN architectures.

---

## 📌 Overview

This project trains and evaluates multiple CNN architectures on the **Durian Leaf Diseases Dataset (DLDD)**, supporting both:

- **Multi-task learning** — jointly predicting disease type + severity in one forward pass
- **Single-task learning** — training separate models for disease or severity only

It also supports **Grad-CAM visualization** to interpret model predictions.

---

## 🗂️ Project Structure

```
DurianLeafProject/
├── data/
│   └── Durian_Leaf_Diseases/
│       ├── train/          # Training images + metadata_train.csv
│       ├── val/            # Validation images + metadata_val.csv
│       └── test/           # Test images + metadata_test.csv
├── checkpoints/            # Saved model checkpoints (.pth) + training logs
│   └── v2batch16_best/
│       ├── best_mobilenetv3_multitask.pth
│       ├── training_log.json
│       ├── classification_report.txt
│       └── confusion_matrix_*.png
├── results/                # Evaluation outputs (metrics, confusion matrices)
├── Grad-CAM/               # Grad-CAM visualization outputs
└── src/
    ├── config.py               # Global paths and hyperparameter defaults
    ├── dataset.py              # Dataset loader (multi-task)
    ├── dataset_single.py       # Dataset loader (single-task)
    ├── MAFC.py                 # Multi-scale Attention Feature Calibration module
    ├── mobilenetv3_custom.py   # MobileNetV2 / MobileNetV3-Small / MobileNetV3-Large (multi-task)
    ├── mobilenetv3_single.py   # MobileNet variants (single-task)
    ├── resnet_custom.py        # ResNet18 / ResNet50 (multi-task)
    ├── resnet_single.py        # ResNet variants (single-task)
    ├── densenet_custom.py      # DenseNet121 (multi-task)
    ├── densenet_single.py      # DenseNet121 (single-task)
    ├── ticknet_custom.py       # TickNet-Large (multi-task)
    ├── ticknet_single.py       # TickNet-Large (single-task)
    ├── train.py                # Multi-task training script
    ├── train_single.py         # Single-task training script
    ├── evaluate.py             # Multi-task evaluation script
    ├── evaluate_single.py      # Single-task evaluation script
    ├── pipeline.py             # Inference + Grad-CAM visualization
    └── utils.py                # Seed, device, parameter counting, FLOPs
```

---

## 🍂 Dataset

The dataset contains durian leaf images organized into the following categories:

| Task | Classes |
|------|---------|
| **Disease** | `healthy`, `algal`, `allocaridara_attack`, `blight`, `phomopsis` |
| **Severity** | `healthy` (0), `mild` (1), `moderate` (2), `severe` (3) |

Each split (`train/`, `val/`, `test/`) has a corresponding CSV file (`metadata_*.csv`) with columns:
- `file_name`, `file_path`, `disease_type`, `label_id`, `severity`

> **Note:** The dataset is not included in this repository. Please prepare the data at `data/Durian_Leaf_Diseases/`.

---

## 🧠 Model Architectures

All models are implemented **from scratch** (no pretrained weights) and adapted for multi-task output with two classification heads:

| Model | Type |
|-------|------|
| **MobileNetV2** | Inverted residual blocks |
| **MobileNetV3-Small** | Inverted residual + SE + HSwish |
| **MobileNetV3-Large** | Inverted residual + SE + HSwish |
| **ResNet18** | Residual blocks |
| **ResNet50** | Bottleneck residual blocks |
| **DenseNet121** | Dense connections |
| **TickNet-Large** | FR-PDP blocks + SE |



---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
```
torch
torchvision
pandas
pillow
scikit-learn
matplotlib
tqdm
```

> Recommended: Python 3.10+, CUDA-enabled GPU.

---

## 🚀 Usage

All scripts must be run from the **`src/`** directory (or with the project root in `PYTHONPATH`).

### 1. Multi-task Training

```bash
cd src
python train.py --model mobilenetv2 --epochs 30 --batch_size 16 --lr 1e-4
```

**Available models:** `mobilenetv2`, `mobilenetv3_small`, `mobilenetv3_large`, `resnet18`, `resnet50`, `densenet121`, `ticknet_large`

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Model architecture |
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--image_size` | 224 | Input image resolution |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-4 | Weight decay (AdamW) |
| `--dropout` | 0.2 | Dropout rate |
| `--severity_loss_weight` | 1.0 | Weight multiplier for severity loss |
| `--use_severity_weight` | False | Use class-weighted loss for severity |
| `--save_all` | False | Save checkpoint every epoch |

Checkpoints and training logs are saved to:
```
checkpoints/<model>_img<size>_bs<batch>_lr<lr>_sw<sw>/
```

---

### 2. Single-task Training

```bash
cd src
python train_single.py --model resnet18 --task disease --epochs 30 --batch_size 16
```

**Additional argument:**

| Argument | Options | Description |
|----------|---------|-------------|
| `--task` | `disease` / `severity` | Which task to train |
| `--use_class_weight` | flag | Use class-weighted CrossEntropyLoss |

---

### 3. Evaluation

```bash
cd src
python evaluate.py --model mobilenetv2 --run_dir v2batch16_best
```

Outputs saved to `results/eval_<model>_<run_dir>/`:
- `classification_report.txt`
- `metrics.json`
- `confusion_matrix_disease.png`
- `confusion_matrix_severity.png`

---

### 4. Inference + Grad-CAM

Edit the paths in `pipeline.py` and run:

```bash
cd src   # hoặc chạy từ thư mục gốc
python pipeline.py
```

Or use the function directly:

```python
from src.pipeline import predict_and_gradcam

result = predict_and_gradcam(
    image_path="data/Durian_Leaf_Diseases/test/algal/DLDD_TEST_000006.jpg",
    checkpoint_path="checkpoints/v2batch16_best/best_mobilenetv3_multitask.pth",
    output_dir="Grad-CAM",
    img_size=224,
    task_for_cam="disease"   # or "severity"
)
print(result)
```

**Output example:**
```json
{
  "disease_idx": 1,
  "disease_label": "class_1",
  "disease_confidence": 0.9997,
  "severity_idx": 2,
  "severity_label": "severity_2",
  "severity_confidence": 0.7704,
  "gradcam_task": "disease",
  "gradcam_path": "Grad-CAM/DLDD_TEST_000006_gradcam_disease.jpg"
}
```

Grad-CAM overlays are saved to the `Grad-CAM/` directory.



## 🧪 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Image size | 224 × 224 |
| Epochs | 30 |
| Dropout | 0.2 |
| Loss | CrossEntropyLoss (disease) + CrossEntropyLoss (severity) |
| Normalization | ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| Data augmentation | RandomResizedCrop, HorizontalFlip, Rotation(20°), ColorJitter |
| Seed | 42 |

---

## 📁 Checkpoint Format

Each saved `.pth` checkpoint contains:

```python
{
    "model_name": str,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": int,
    "train_loss": float,
    "train_disease_acc": float,
    "train_severity_acc": float,
    "val_loss": float,
    "val_disease_acc": float,
    "val_severity_acc": float,
    "total_params": int,
    "trainable_params": int,
    "flops": int,
    "flops_readable": str,
    "params_readable": str,
    "image_size": int,
    "batch_size": int,
}
```

---

## 📝 Notes

- The `Unexpected keys` warning about `total_ops` / `total_params` during inference is harmless — these are profiling buffers added by the **THOP** library during FLOPs calculation and are automatically stripped before saving.
- For best inference results, ensure the image path is correct relative to the project root when running `pipeline.py`.
- GPU is used automatically if available (`cuda`), otherwise falls back to `cpu`.


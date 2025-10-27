# 🦴 Marginal Bone Loss Classifier (PyTorch)

Modern, complete training pipeline for **medical image classification** (e.g., dental radiographs for bone loss detection).  
Refactored from notebook to a clean CLI script by **[Martin Badrous](https://github.com/martinbadrous)**.

---

## 🌟 Features
- Folder **or CSV** datasets (`image_path,label`)
- Medical-friendly preprocessing (grayscale, optional histogram equalization)
- Transfer learning: **ResNet50** / **EfficientNet-B0**
- GPU/CPU auto, **AMP** mixed precision
- Class imbalance handling with **class weights**
- Auto **stratified split** (train/val/test) or use pre-split folders
- **Early stopping**, checkpointing (best/last), resume
- Metrics: **Accuracy, Precision, Recall, F1, ROC-AUC**, **Confusion Matrix**
- Exports: `metrics.csv`, `test_metrics.json`, `class_to_idx.json`, `confusion_matrix.png`, `roc.png`, `weights.pth`

---

## 📦 Repository Structure
```bash
Marginal-Bone-Loss-Q/
├── train_boneloss.py           # Main training script (modernized)
├── requirements_boneloss.txt   # Dependencies
├── dataset/                    # Your images (see formats below)
└── runs/
    └── exp1/
        ├── best.pt
        ├── last.pt
        ├── weights.pth
        ├── class_to_idx.json
        ├── metrics.csv
        ├── test_metrics.json
        ├── confusion_matrix.png
        └── roc.png
```

---

## 🧠 Dataset Formats

### A) Single folder (auto-split)
```
dataset/
├── bone_loss/
└── healthy/
```
Script auto-splits using `--val_split` and `--test_split`.

### B) Pre-split folders
```
dataset/
├── train/
├── val/
└── test/
```
Each split contains class subfolders.

### C) CSV file
```
image_path,label
patient001/image.png,bone_loss
patient002/image.png,healthy
```
Use with: `--csv_file ./labels.csv --images_root ./` (if paths are relative).

---

## 🚀 Quick Start (copy/paste)

```bash
# 1) Get the code
git clone https://github.com/martinbadrous/Marginal-Bone-Loss-Q.git
cd Marginal-Bone-Loss-Q

# 2) Setup environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scriptsctivate
pip install -r requirements_boneloss.txt

# 3) Train from folder dataset (auto-split)
python train_boneloss.py   --data_dir ./dataset   --output_dir ./runs/exp1   --epochs 30 --batch_size 32 --img_size 224   --model resnet50 --pretrained --augment --equalize --amp   --val_split 0.15 --test_split 0.10

# (OR) Train from CSV
python train_boneloss.py   --csv_file ./labels.csv --images_root ./images   --output_dir ./runs/exp_csv   --epochs 25 --batch_size 32   --model efficientnet_b0 --pretrained --augment --amp

# 4) Evaluate
# Best model is auto-evaluated on test split; see ./runs/<exp>/test_metrics.json
```

---

## ⚙️ Key Arguments
| Flag | Description |
|------|-------------|
| `--data_dir` | Root with class subfolders OR `train/val/(test)` |
| `--csv_file` / `--images_root` | CSV mode with image paths and labels |
| `--model` | `resnet50` (default) or `efficientnet_b0` |
| `--pretrained` | Use ImageNet weights |
| `--freeze_backbone` | `1` to freeze feature extractor |
| `--augment` | Enable augmentations (crop/flip/rotation) |
| `--equalize` | Histogram equalization (PIL) |
| `--img_size` | Resize/crop size (default 224) |
| `--amp` | Mixed precision training |
| `--patience` | Early stopping patience (default 8) |

---

## 🧪 Outputs & Metrics
- `metrics.csv` — per-epoch train/val loss & accuracy
- `test_metrics.json` — final test accuracy, precision, recall, F1, ROC-AUC
- `confusion_matrix.png` — normalized counts per class
- `roc.png` — ROC curve (binary) or one-vs-rest (multiclass)
- `weights.pth` — weights-only for deployment

---

## 🧭 Roadmap
- [ ] Add `infer.py` for single-image and folder inference
- [ ] Grad-CAM visualizations for explainability
- [ ] DICOM import (via `pydicom`) for radiography workflows
- [ ] ONNX/TorchScript export

---

## 👨‍⚕️ Notes on Medical Use
This code is for **research** and **education**. It is **not** a medical device and should not be used for clinical decisions without appropriate validation and regulatory compliance.

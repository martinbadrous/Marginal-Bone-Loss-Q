#!/usr/bin/env python3
"""
Marginal Bone Loss Classifier — Modern PyTorch Training Script
Author: Martin Badrous (repo modernization)

Supports:
- Dataset as folders (ImageFolder) OR CSV (image_path,label)
- Medical image preprocessing (grayscale, histogram eq optional), data augmentation
- Transfer learning backbones: resnet50, efficientnet_b0
- GPU/CPU auto, AMP mixed precision
- Class weights for imbalance
- Stratified auto split (train/val/test) or use pre-split folders
- Early stopping, checkpointing (best/last), resume
- Metrics: accuracy, precision, recall, f1, ROC-AUC (binary or macro), confusion matrix
- Exports: metrics.csv, test_metrics.json, class_to_idx.json, cm.png, roc.png

Example (auto-split from single folder):
python train_boneloss.py \
  --data_dir ./dataset \
  --output_dir ./runs/exp1 \
  --epochs 30 --batch_size 32 --img_size 224 \
  --model resnet50 --pretrained --augment --amp \
  --val_split 0.15 --test_split 0.10

Example (CSV dataset):
python train_boneloss.py \
  --csv_file ./labels.csv --images_root ./images \
  --output_dir ./runs/exp_csv \
  --epochs 25 --batch_size 32 --model efficientnet_b0 --pretrained --augment

CSV requirements:
- columns: image_path,label
- image_path is relative to --images_root or absolute
- label is class name (string) or index (int)

"""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models, datasets

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt

# -----------------------
# Reproducibility & device
# -----------------------

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Data utilities
# -----------------------

def imagenet_norm():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return mean, std

class CSVDataset(Dataset):
    """
    Reads a CSV with columns: image_path,label
    - image_path can be relative to images_root or absolute
    - label can be class string or integer
    """
    def __init__(self, csv_path: Path, images_root: Optional[Path], transform=None,
                 class_to_idx: Optional[Dict[str, int]] = None):
        self.items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            # basic detection
            cols = {name: i for i, name in enumerate(header)}
            if "image_path" not in cols or "label" not in cols:
                raise ValueError("CSV must have columns: image_path,label")
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 2:
                    continue
                self.items.append((parts[cols["image_path"]], parts[cols["label"]]))
        self.images_root = images_root
        self.transform = transform
        # build class_to_idx if not provided
        if class_to_idx is None:
            classes = [lbl for _, lbl in self.items]
            if all(lbl.isdigit() for lbl in classes):
                numeric_labels = sorted({int(lbl) for lbl in classes})
                self.class_to_idx = {str(num): idx for idx, num in enumerate(numeric_labels)}
            else:
                unique = sorted(set(classes))
                self.class_to_idx = {lbl: i for i, lbl in enumerate(unique)}
        else:
            self.class_to_idx = class_to_idx
        self.label_lookup: Dict[str, int] = {}
        for key, idx in self.class_to_idx.items():
            self.label_lookup[str(key)] = idx
            if isinstance(key, str):
                self.label_lookup[key] = idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path_str, lbl = self.items[idx]
        img_path = Path(path_str)
        if not img_path.is_absolute() and self.images_root is not None:
            img_path = self.images_root / img_path
        img = Image.open(img_path).convert("L")  # medical images often grayscale
        if self.transform:
            img = self.transform(img)
        # convert class label
        key = lbl if lbl in self.label_lookup else str(lbl)
        if key not in self.label_lookup:
            raise KeyError(f"Label '{lbl}' not found in class_to_idx mapping")
        y = self.label_lookup[key]
        return img, y

def build_transforms(img_size: int, augment: bool, equalize: bool, to_rgb: bool):
    mean, std = imagenet_norm()
    # base pipeline for medical images; start with grayscale
    train_tf = []
    if augment:
        train_tf += [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=7),
        ]
    else:
        train_tf += [transforms.Resize((img_size, img_size))]
    # equalize histogram (PIL) optional, applied via Lambda
    if equalize:
        train_tf += [transforms.Lambda(lambda im: ImageOps.equalize(im))]
    # convert to tensor as 1-channel, then optionally expand to 3
    train_tf += [transforms.ToTensor()]
    if to_rgb:
        train_tf += [transforms.Lambda(lambda t: t.expand(3, -1, -1))]
        train_tf += [transforms.Normalize(mean, std)]
    else:
        # normalize grayscale with ImageNet first channel stats as fallback
        train_tf += [transforms.Normalize([0.5], [0.5])]

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda im: ImageOps.equalize(im) if equalize else im),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1)) if to_rgb else transforms.Lambda(lambda t: t),
        transforms.Normalize(mean, std) if to_rgb else transforms.Normalize([0.5], [0.5]),
    ])
    return transforms.Compose(train_tf), eval_tf

def infer_splits(base: Path) -> Optional[Dict[str, Path]]:
    cand = {"train": base / "train", "val": base / "val", "test": base / "test"}
    if all(p.exists() and p.is_dir() for p in cand.values()):
        return cand
    if (base / "train").exists() and (base / "val").exists():
        return {"train": base / "train", "val": base / "val", "test": None}
    return None

def stratified_split_indices(targets: List[int], val_split: float, test_split: float, seed: int = 42):
    assert 0 <= val_split < 1 and 0 <= test_split < 1 and val_split + test_split < 1
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test+n_val])
        train_idx.extend(idxs[n_test+n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

@dataclass
class Datasets:
    train: Dataset
    val: Dataset
    test: Optional[Dataset]
    class_to_idx: Dict[str, int]

def build_datasets_from_folder(data_dir: Path, img_size: int, augment: bool, equalize: bool, seed: int,
                               val_split: float, test_split: float, to_rgb: bool) -> Datasets:
    split = infer_splits(data_dir)
    train_tf, eval_tf = build_transforms(img_size, augment, equalize, to_rgb)

    if split:
        ds_train = datasets.ImageFolder(split["train"], transform=train_tf)
        ds_val = datasets.ImageFolder(split["val"], transform=eval_tf)
        ds_test = datasets.ImageFolder(split["test"], transform=eval_tf) if split.get("test") else None
        # Convert to grayscale first by wrapping transform? ImageFolder opens RGB; patch with additional transform
        # We'll rely on transforms at build_transforms to expand grayscale; for ImageFolder, convert to L in a Lambda
        # So we recompose with an initial grayscale conversion
        def add_gray(tf):
            return transforms.Compose([transforms.Grayscale(num_output_channels=1)] + tf.transforms)
        ds_train.transform = add_gray(ds_train.transform)
        ds_val.transform = add_gray(ds_val.transform)
        if ds_test: ds_test.transform = add_gray(ds_test.transform)
        return Datasets(train=ds_train, val=ds_val, test=ds_test, class_to_idx=ds_train.class_to_idx)

    # Not split: use ImageFolder for index/targets, then Subset
    base_no_tf = datasets.ImageFolder(data_dir)
    targets = [y for _, y in base_no_tf.samples]
    train_idx, val_idx, test_idx = stratified_split_indices(targets, val_split, test_split, seed)
    base_train = datasets.ImageFolder(data_dir, transform=transforms.Compose([transforms.Grayscale(1), train_tf]))
    base_eval  = datasets.ImageFolder(data_dir, transform=transforms.Compose([transforms.Grayscale(1), eval_tf]))
    ds_train = Subset(base_train, train_idx)
    ds_val   = Subset(base_eval,  val_idx)
    ds_test  = Subset(base_eval,  test_idx) if len(test_idx) else None
    return Datasets(train=ds_train, val=ds_val, test=ds_test, class_to_idx=base_no_tf.class_to_idx)

def build_datasets_from_csv(csv_file: Path, images_root: Optional[Path], img_size: int, augment: bool, equalize: bool, seed: int,
                            val_split: float, test_split: float, to_rgb: bool) -> Datasets:
    train_tf, eval_tf = build_transforms(img_size, augment, equalize, to_rgb)
    base = CSVDataset(csv_file, images_root, transform=None, class_to_idx=None)
    # targets from mapping
    lbls = []
    for _, lbl in base.items:
        key = lbl if lbl in base.label_lookup else str(lbl)
        if key not in base.label_lookup:
            raise KeyError(f"Label '{lbl}' not found in CSV class mapping")
        lbls.append(base.label_lookup[key])
    train_idx, val_idx, test_idx = stratified_split_indices(lbls, val_split, test_split, seed)
    ds_train = CSVDataset(csv_file, images_root, transform=train_tf, class_to_idx=base.class_to_idx)
    ds_val   = CSVDataset(csv_file, images_root, transform=eval_tf,  class_to_idx=base.class_to_idx)
    ds_test  = CSVDataset(csv_file, images_root, transform=eval_tf,  class_to_idx=base.class_to_idx) if len(test_idx) else None
    # Wrap via Subset to use our indices
    ds_train = Subset(ds_train, train_idx)
    ds_val   = Subset(ds_val,   val_idx)
    ds_test  = Subset(ds_test,  test_idx) if ds_test else None
    return Datasets(train=ds_train, val=ds_val, test=ds_test, class_to_idx=base.class_to_idx)

# -----------------------
# Models
# -----------------------

def build_model(arch: str, num_classes: int, pretrained: bool, freeze_backbone: bool):
    arch = arch.lower()
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        backbone_prefix = "fc"
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
        backbone_prefix = "classifier"
    else:
        raise ValueError("Supported models: resnet50, efficientnet_b0")
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith(backbone_prefix):
                p.requires_grad = False
    return model

# -----------------------
# Training / Evaluation
# -----------------------

def compute_class_weights(loader: DataLoader, num_classes: int, device):
    # count labels from the loader once
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, targets in loader:
        for t in targets:
            counts[int(t)] += 1
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    return weights.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, running_correct, n = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        # metrics
        preds = logits.argmax(1)
        running_correct += (preds == targets).sum().item()
        running_loss += loss.item() * images.size(0)
        n += images.size(0)
    return running_loss / n, running_correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    running_loss, running_correct, n = 0.0, 0, 0
    all_targets, all_probs = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        running_correct += (preds == targets).sum().item()
        running_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    loss = running_loss / n
    acc = running_correct / n
    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    return loss, acc, y_true, y_prob

def metrics_from_outputs(y_true: np.ndarray, y_prob: np.ndarray, average: str = "macro"):
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    # ROC-AUC (binary or multiclass)
    try:
        if y_prob.shape[1] == 2:
            auc_val = roc_auc_score(y_true, y_prob[:,1])
        else:
            auc_val = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
    except Exception:
        auc_val = float("nan")
    return acc, precision, recall, f1, auc_val, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names: List[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, y_prob, class_names: List[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    if y_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    else:
        # one-vs-rest
        for i, name in enumerate(class_names):
            y_true_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0,1], [0,1], linestyle="--", color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_checkpoint(state: dict, is_best: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / "last.pt")
    if is_best:
        torch.save(state, out_dir / "best.pt")

def csv_logger_init(path: Path):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
    return f, writer

# -----------------------
# Main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Marginal Bone Loss Classifier — PyTorch training pipeline")
    data = p.add_argument_group("Data")
    data.add_argument("--data_dir", type=str, help="Dataset root (folders). If provided, CSV options are ignored.")
    data.add_argument("--csv_file", type=str, help="CSV file with image_path,label")
    data.add_argument("--images_root", type=str, default="", help="Root folder for images when using CSV (optional).")
    data.add_argument("--img_size", type=int, default=224)
    data.add_argument("--augment", action="store_true", help="Enable data augmentation")
    data.add_argument("--equalize", action="store_true", help="Histogram equalization (PIL) before tensor conversion")
    data.add_argument("--val_split", type=float, default=0.15, help="Val fraction if auto-splitting")
    data.add_argument("--test_split", type=float, default=0.10, help="Test fraction if auto-splitting")

    train = p.add_argument_group("Training")
    train.add_argument("--output_dir", type=str, default="./runs/exp1")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--amp", action="store_true", help="Mixed precision training")
    train.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    train.add_argument("--patience", type=int, default=8, help="Early stopping patience")

    model = p.add_argument_group("Model")
    model.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    model.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights")
    model.add_argument("--freeze_backbone", type=int, default=0, help="Freeze backbone (1)")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Transforms and RGB expansion for pretrained models
    to_rgb = True  # expand grayscale to 3-ch for ImageNet backbones
    train_tf, eval_tf = build_transforms(args.img_size, args.augment, args.equalize, to_rgb)

    # Build datasets
    if args.data_dir:
        ds = build_datasets_from_folder(Path(args.data_dir), args.img_size, args.augment, args.equalize,
                                        args.seed, args.val_split, args.test_split, to_rgb)
    elif args.csv_file:
        images_root = Path(args.images_root) if args.images_root else None
        ds = build_datasets_from_csv(Path(args.csv_file), images_root, args.img_size, args.augment, args.equalize,
                                     args.seed, args.val_split, args.test_split, to_rgb)
    else:
        raise ValueError("Provide either --data_dir for folder dataset OR --csv_file (and optional --images_root).")

    num_classes = len(ds.class_to_idx)
    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(ds.class_to_idx, f, indent=2, ensure_ascii=False)

    train_loader = DataLoader(ds.train, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(ds.val,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(ds.test,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if ds.test else None

    # Model, loss, optimizer, scheduler
    model = build_model(args.model, num_classes, pretrained=args.pretrained, freeze_backbone=bool(args.freeze_backbone))
    model.to(device)
    # class weights for imbalance (computed on the fly using a single pass over train_loader)
    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume
    start_epoch, best_val = 0, 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val_acc={best_val:.4f})")

    # CSV logger
    csv_f = open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f); csv_w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # Train
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, y_true, y_prob = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        csv_w.writerow([epoch+1, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr:.6e}"])
        csv_f.flush()

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Early stopping + checkpoint
        improved = val_acc > best_val
        best_val = max(best_val, val_acc)
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val,
            "args": vars(args),
            "class_to_idx": ds.class_to_idx,
        }, is_best=improved, out_dir=out_dir)

        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    csv_f.close()

    # Evaluate on test split if available
    if test_loader is not None:
        # load best checkpoint if exists
        best_ckpt = out_dir / "best.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state["model"])
        test_loss, test_acc, y_true, y_prob = evaluate(model, test_loader, criterion, device, num_classes)
        acc, precision, recall, f1, auc_val, y_pred = metrics_from_outputs(y_true, y_prob, average="macro")
        print(f"\nTest: loss={test_loss:.4f} acc={test_acc:.4f} | macro P={precision:.3f} R={recall:.3f} F1={f1:.3f} AUC={auc_val:.3f}")
        # save metrics
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"loss": float(test_loss), "acc": float(test_acc), "precision": float(precision),
                       "recall": float(recall), "f1": float(f1), "auc": float(auc_val)}, f, indent=2)
        # plots
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")
        plot_roc(y_true, y_prob, class_names, out_dir / "roc.png")

    # export weights-only
    torch.save(model.state_dict(), out_dir / "weights.pth")
    print(f"\nTraining complete. Artifacts saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

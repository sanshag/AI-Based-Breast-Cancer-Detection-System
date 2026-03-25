"""
==========================================================================
 Agent 2 — DEEP LEARNING ENGINEER: Specialist Fine-Tuning Scripts
 Agent 3 — VALIDATION & METRICS: Benchmarking & Grad-CAM
==========================================================================

 Agent 2 Responsibilities:
   - YOLO11 Fine-Tuning via model.train() with Transfer Learning
   - DenseNet201 Specialist Head: Unfreeze last two DenseBlocks and train
   - Save specialist weights: yolo11_cbis_specialist.pt, densenet_cbis_specialist.pt

 Agent 3 Responsibilities:
   - Clinical Benchmark: General vs Specialist weights comparison
   - False Negative Rate (FNR) tracking — zero-miss malignancy goal
   - Grad-CAM explainability to verify the model focuses on mass margins

 Combined into one script for unified workflow.
==========================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

from ultralytics import YOLO

from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

YOLO_SPECIALIST_NAME = "yolo11_cbis_specialist.pt"
DENSENET_SPECIALIST_NAME = "densenet_cbis_specialist.pt"

# Default training hyperparameters
YOLO_EPOCHS = 100
YOLO_IMGSZ = 640
YOLO_BATCH = 16
YOLO_LR = 0.01

DENSENET_EPOCHS = 50
DENSENET_BATCH = 32
DENSENET_LR = 0.0001
DENSENET_INPUT_SIZE = 224


# ══════════════════════════════════════════════════════════════════════
#  AGENT 2: YOLO11 FINE-TUNING (Transfer Learning)
# ══════════════════════════════════════════════════════════════════════

def train_yolo_specialist(
    base_model: str = "yolo11n.pt",
    dataset_yaml: str = "cbis_yolo_dataset/dataset.yaml",
    epochs: int = YOLO_EPOCHS,
    imgsz: int = YOLO_IMGSZ,
    batch: int = YOLO_BATCH,
    lr0: float = YOLO_LR,
    output_dir: str = "training_output",
    project_name: str = "yolo11_cbis_specialist",
) -> str:
    """
    Fine-tune YOLO11 on the CBIS-DDSM dataset using Transfer Learning.

    Starting from the general yolo11n.pt weights, we fine-tune on the
    CBIS-DDSM mammography dataset for 100 epochs, focusing on:
      - box loss (bounding box regression)
      - cls loss (classification: Benign vs Malignant)

    Args:
        base_model:   Path to the pre-trained YOLO11 weights (starting point).
        dataset_yaml: Path to the YOLO dataset.yaml configuration.
        epochs:       Number of training epochs (default: 100).
        imgsz:        Input image size (default: 640).
        batch:        Batch size (default: 16).
        lr0:          Initial learning rate (default: 0.01).
        output_dir:   Base directory for training outputs.
        project_name: Name for the training project/run.

    Returns:
        Path to the best specialist weights file.
    """
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   AGENT 2: YOLO11 SPECIALIST FINE-TUNING               ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Base Model:  {base_model:<40} ║")
    print(f"  ║  Dataset:     {dataset_yaml:<40} ║")
    print(f"  ║  Epochs:      {epochs:<40} ║")
    print(f"  ║  Image Size:  {imgsz:<40} ║")
    print(f"  ║  Batch Size:  {batch:<40} ║")
    print(f"  ║  Learning Rate: {lr0:<38} ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")

    # Load the pre-trained model
    model = YOLO(base_model)

    # Run training with Transfer Learning
    # Ultralytics handles:
    #   - Freezing/unfreezing backbone layers
    #   - Learning rate scheduling (cosine annealing)
    #   - Data augmentation (mosaic, mixup, etc.)
    #   - Best model checkpointing
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=0.01,           # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss focus weights
        box=7.5,            # Box loss gain (emphasize precise localization)
        cls=0.5,            # Classification loss gain
        dfl=1.5,            # Distribution focal loss

        # Augmentation (medical-appropriate)
        hsv_h=0.015,        # Minimal hue shift (mammograms are grayscale-ish)
        hsv_s=0.3,
        hsv_v=0.4,
        degrees=15.0,       # Slight rotation (valid for breast positioning)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,         # Horizontal flip (left/right breast)
        flipud=0.1,         # Slight vertical flip
        mosaic=0.8,
        mixup=0.1,

        # Output
        project=output_dir,
        name=project_name,
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=10,      # Save checkpoint every 10 epochs
        plots=True,          # Generate training plots
        patience=20,         # Early stopping patience
    )

    # Copy best weights to the specialist filename
    best_weights = Path(output_dir) / project_name / "weights" / "best.pt"
    specialist_path = Path(output_dir) / YOLO_SPECIALIST_NAME

    if best_weights.exists():
        import shutil
        shutil.copy2(str(best_weights), str(specialist_path))
        # Also copy to project root for easy access
        root_specialist = Path(YOLO_SPECIALIST_NAME)
        shutil.copy2(str(best_weights), str(root_specialist))

        print(f"\n  ✅ YOLO11 Specialist Training Complete!")
        print(f"     Best weights: {specialist_path}")
        print(f"     Root copy:    {root_specialist}")
        return str(root_specialist)
    else:
        print(f"\n  ⚠ Training completed but best.pt not found at: {best_weights}")
        return str(best_weights)


# ══════════════════════════════════════════════════════════════════════
#  AGENT 2: DenseNet201 SPECIALIST HEAD TRAINING
# ══════════════════════════════════════════════════════════════════════

class DenseNetSpecialistTrainer:
    """
    Fine-tunes DenseNet201 for mammogram ROI classification.

    Strategy:
      1. Load ImageNet-pretrained DenseNet201
      2. Add custom binary classifier head
      3. Freeze all layers except the LAST TWO DenseBlocks + classifier
      4. Train on CBIS-DDSM ROI crops (Benign vs Malignant)
      5. Save as densenet_cbis_specialist.pt

    The last two DenseBlocks (denseblock3, denseblock4) contain the
    highest-level feature representations. By unfreezing only these,
    we adapt the model's fine-grained feature extraction to mammogram
    tissue patterns while preserving lower-level learned features.
    """

    def __init__(
        self,
        data_dir: str = "cbis_densenet_dataset",
        epochs: int = DENSENET_EPOCHS,
        batch_size: int = DENSENET_BATCH,
        learning_rate: float = DENSENET_LR,
        input_size: int = DENSENET_INPUT_SIZE,
        output_dir: str = "training_output",
    ):
        self.data_dir = Path(data_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_acc = 0.0
        self.training_history: List[Dict] = []

    def _build_model(self) -> nn.Module:
        """
        Build DenseNet201 with custom specialist head.
        Freeze everything except last two DenseBlocks + classifier.
        """
        print(f"  🔧 Building DenseNet201 Specialist Model...")

        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

        # ── Step 1: Freeze ALL layers ─────────────────────────────────
        for param in model.parameters():
            param.requires_grad = False

        # ── Step 2: Unfreeze LAST TWO DenseBlocks ─────────────────────
        # DenseNet201 features structure:
        #   features.denseblock1 (6 layers)
        #   features.transition1
        #   features.denseblock2 (12 layers)
        #   features.transition2
        #   features.denseblock3 (48 layers)    ← UNFREEZE
        #   features.transition3
        #   features.denseblock4 (32 layers)    ← UNFREEZE
        #   features.norm5

        unfrozen_blocks = []
        for name, param in model.features.named_parameters():
            if "denseblock3" in name or "denseblock4" in name or "norm5" in name:
                param.requires_grad = True
                if name.split(".")[0] not in unfrozen_blocks:
                    unfrozen_blocks.append(name.split(".")[0])

        # Also unfreeze transition3 (between denseblock3 and denseblock4)
        for name, param in model.features.named_parameters():
            if "transition3" in name:
                param.requires_grad = True

        print(f"  🔓 Unfrozen blocks: {unfrozen_blocks}")

        # ── Step 3: Replace classifier with specialist head ──────────
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # [benign_logit, malignant_logit]
        )

        # Classifier is always trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Count trainable params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  📊 Total params: {total_params:,}")
        print(f"  📊 Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

        return model

    def _get_data_loaders(self) -> Tuple[DataLoader, DataLoader, Dict]:
        """Create training and validation data loaders with medical augmentations."""

        # Training transforms — medical-appropriate augmentations
        train_transform = transforms.Compose([
            transforms.Resize((self.input_size + 32, self.input_size + 32)),
            transforms.RandomCrop((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Validation transforms — no augmentation
        val_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        train_dataset = datasets.ImageFolder(
            str(self.data_dir / "train"), transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            str(self.data_dir / "val"), transform=val_transform
        )

        class_names = train_dataset.classes
        class_counts = {}
        for _, label in train_dataset.samples:
            cls_name = class_names[label]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        print(f"  📊 Training samples: {len(train_dataset)}")
        print(f"  📊 Validation samples: {len(val_dataset)}")
        print(f"  📊 Class distribution: {class_counts}")

        # Weighted sampler for class imbalance
        class_weights = []
        for _, label in train_dataset.samples:
            cls_name = class_names[label]
            class_weights.append(1.0 / class_counts[cls_name])
        sampler = torch.utils.data.WeightedRandomSampler(
            class_weights, len(class_weights)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            sampler=sampler, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )

        stats = {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "classes": class_names,
            "class_distribution": class_counts,
        }

        return train_loader, val_loader, stats

    def train(self) -> str:
        """
        Run the full DenseNet201 specialist training pipeline.

        Returns:
            Path to the saved specialist weights.
        """
        print("\n  ╔══════════════════════════════════════════════════════════╗")
        print("  ║   AGENT 2: DenseNet201 SPECIALIST HEAD TRAINING         ║")
        print("  ╠══════════════════════════════════════════════════════════╣")
        print(f"  ║  Dataset:       {str(self.data_dir):<38} ║")
        print(f"  ║  Epochs:        {self.epochs:<38} ║")
        print(f"  ║  Batch Size:    {self.batch_size:<38} ║")
        print(f"  ║  Learning Rate: {self.learning_rate:<38} ║")
        print(f"  ║  Device:        {str(self.device):<38} ║")
        print("  ║  Strategy: Unfreeze last 2 DenseBlocks + classifier     ║")
        print("  ╚══════════════════════════════════════════════════════════╝\n")

        # Build model
        model = self._build_model()
        model.to(self.device)

        # Data loaders
        train_loader, val_loader, stats = self._get_data_loaders()

        # Optimizer — different LR for backbone vs classifier
        backbone_params = [
            p for n, p in model.features.named_parameters() if p.requires_grad
        ]
        classifier_params = list(model.classifier.parameters())

        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": self.learning_rate * 0.1},  # Lower LR for backbone
            {"params": classifier_params, "lr": self.learning_rate},
        ], weight_decay=1e-4)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-7
        )

        # Loss function with class weights for imbalance
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_model_state = None
        patience = 15
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{self.epochs} [Train]",
                       unit="batch", leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_correct/train_total:.3f}")

            scheduler.step()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # ── Validate ──────────────────────────────────────────────
            val_metrics = self._validate(model, val_loader, criterion)

            # Record history
            epoch_data = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_metrics["val_loss"], 4),
                "val_acc": round(val_metrics["val_acc"], 4),
                "val_fnr": round(val_metrics["false_negative_rate"], 4),
                "val_sensitivity": round(val_metrics["sensitivity"], 4),
                "val_specificity": round(val_metrics["specificity"], 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
            self.training_history.append(epoch_data)

            print(
                f"  Epoch {epoch:>3}/{self.epochs} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"Val: loss={val_metrics['val_loss']:.4f} acc={val_metrics['val_acc']:.3f} | "
                f"FNR={val_metrics['false_negative_rate']:.3f} "
                f"Sens={val_metrics['sensitivity']:.3f}"
            )

            # Save best model
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"  ★ New best model! Val Acc: {self.best_val_acc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch} (patience={patience})")
                break

        # Save specialist weights
        specialist_path = self.output_dir / DENSENET_SPECIALIST_NAME
        root_specialist = Path(DENSENET_SPECIALIST_NAME)

        if best_model_state:
            torch.save(best_model_state, str(specialist_path))
            torch.save(best_model_state, str(root_specialist))
            print(f"\n  ✅ DenseNet201 Specialist Training Complete!")
            print(f"     Best Val Accuracy: {self.best_val_acc:.4f}")
            print(f"     Weights: {specialist_path}")
            print(f"     Root copy: {root_specialist}")
        else:
            torch.save(model.state_dict(), str(specialist_path))
            torch.save(model.state_dict(), str(root_specialist))

        # Save training history
        history_path = self.output_dir / "densenet_training_history.json"
        with open(str(history_path), "w") as f:
            json.dump(self.training_history, f, indent=2)

        return str(root_specialist)

    def _validate(
        self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module
    ) -> Dict:
        """Validate and compute clinical metrics including FNR."""
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        total = len(all_labels)

        val_loss /= total
        val_acc = (all_preds == all_labels).sum() / total

        # Clinical metrics (class 1 = Malignant is the positive class)
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()

        sensitivity = tp / max(tp + fn, 1)       # True Positive Rate
        specificity = tn / max(tn + fp, 1)        # True Negative Rate
        fnr = fn / max(tp + fn, 1)                # False Negative Rate (CRITICAL)
        fpr = fp / max(tn + fp, 1)                # False Positive Rate

        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_negative_rate": fnr,
            "false_positive_rate": fpr,
        }


# ══════════════════════════════════════════════════════════════════════
#  AGENT 3: VALIDATION — GENERAL vs SPECIALIST BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def benchmark_general_vs_specialist(
    val_data_dir: str = "cbis_densenet_dataset/val",
    general_weights: str = "",
    specialist_weights: str = "densenet_cbis_specialist.pt",
    output_dir: str = "training_output",
) -> Dict:
    """
    Compare General (ImageNet) weights vs Specialist (CBIS-DDSM fine-tuned)
    weights on the validation set.

    Tracks:
      - Accuracy, Sensitivity (Recall), Specificity
      - FALSE NEGATIVE RATE (primary clinical metric)
      - Per-class precision and recall

    Returns:
        Comparison results dictionary.
    """
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   AGENT 3: GENERAL vs SPECIALIST BENCHMARK              ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(val_data_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    results = {}

    for model_name, weights_path in [("General (ImageNet)", general_weights),
                                       ("Specialist (CBIS-DDSM)", specialist_weights)]:
        print(f"\n  ── Evaluating: {model_name} ──────────────────────────")

        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"     Loaded weights: {weights_path}")
        else:
            print(f"     Using default ImageNet weights (no specialist fine-tuning)")

        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"  {model_name}", leave=False):
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # P(malignant)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        fnr = fn / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)

        metrics = {
            "accuracy": round(float(accuracy), 4),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "false_negative_rate": round(float(fnr), 4),
            "precision": round(float(precision), 4),
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
        }
        results[model_name] = metrics

        print(f"     Accuracy:    {accuracy:.4f}")
        print(f"     Sensitivity: {sensitivity:.4f}")
        print(f"     Specificity: {specificity:.4f}")
        print(f"     FNR:         {fnr:.4f} {'✅ LOW' if fnr < 0.05 else '⚠ NEEDS IMPROVEMENT'}")
        print(f"     Precision:   {precision:.4f}")
        print(f"     TP:{int(tp)} FN:{int(fn)} FP:{int(fp)} TN:{int(tn)}")

    # Comparison summary
    print("\n  ═══════════════════════════════════════════════════════════")
    print("  BENCHMARK COMPARISON: General vs Specialist")
    print("  ═══════════════════════════════════════════════════════════")
    for metric in ["accuracy", "sensitivity", "specificity", "false_negative_rate", "precision"]:
        gen_val = results.get("General (ImageNet)", {}).get(metric, 0)
        spec_val = results.get("Specialist (CBIS-DDSM)", {}).get(metric, 0)
        diff = spec_val - gen_val
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "─"
        if metric == "false_negative_rate":
            arrow = "↓" if diff < 0 else "↑" if diff > 0 else "─"
            prefix = "✅" if diff < 0 else "⚠"
        else:
            prefix = "✅" if diff > 0 else "⚠"
        print(f"  {metric:<25} General: {gen_val:.4f}  →  Specialist: {spec_val:.4f}  {prefix}{arrow} ({diff:+.4f})")
    print("  ═══════════════════════════════════════════════════════════")

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  📄 Benchmark report: {report_path}")

    return results


# ══════════════════════════════════════════════════════════════════════
#  AGENT 3: GRAD-CAM EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for DenseNet201.

    Verifies that the fine-tuned model focuses on mass margins
    (spiculations, irregular borders) rather than background artifacts.

    Generates heatmap overlays showing which regions of the ROI crop
    most influenced the classification decision.
    """

    def __init__(self, model: nn.Module, target_layer: str = "features.denseblock4"):
        """
        Args:
            model:        The DenseNet201 model.
            target_layer: Name of the layer to compute Grad-CAM for.
                         Default is the last DenseBlock.
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        target = dict(model.named_modules())[target_layer]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W).
            target_class: Class to generate CAM for (None = predicted class).

        Returns:
            Normalized heatmap as numpy array (H, W) in range [0, 1].
        """
        self.model.zero_grad()

        output = self.model(image_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global avg pooling of gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def visualize(
        self,
        image_path: str,
        output_path: str,
        input_size: int = 224,
        target_class: Optional[int] = None,
    ) -> str:
        """
        Generate and save a Grad-CAM visualization overlay.

        Args:
            image_path:   Path to the ROI crop image.
            output_path:  Where to save the visualization.
            input_size:   Model input size.
            target_class: Class to visualize (None = predicted).

        Returns:
            Path to saved visualization.
        """
        from PIL import Image as PILImage

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pil_img = PILImage.open(image_path).convert("RGB")
        tensor = transform(pil_img).unsqueeze(0)

        device = next(self.model.parameters()).device
        tensor = tensor.to(device)

        # Generate heatmap
        cam = self.generate(tensor, target_class)

        # Resize heatmap to original image size
        original = cv2.imread(image_path)
        h, w = original.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Overlay
        overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)

        # Add label
        pred_output = self.model(tensor)
        probs = F.softmax(pred_output, dim=1)[0].detach().cpu().numpy()
        pred_class = "Malignant" if probs[1] > probs[0] else "Benign"
        label = f"Grad-CAM: {pred_class} (B:{probs[0]:.2f} M:{probs[1]:.2f})"

        cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (255, 255, 255), 2, cv2.LINE_AA)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)

        return output_path


def generate_gradcam_report(
    model_weights: str = "densenet_cbis_specialist.pt",
    image_dir: str = "cbis_densenet_dataset/val",
    output_dir: str = "training_output/gradcam",
    max_images: int = 20,
) -> str:
    """
    Generate Grad-CAM visualizations for validation images.

    Verifies the specialist model focuses on mass margins (spiculations)
    and not background noise or artifacts.
    """
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   AGENT 3: GRAD-CAM EXPLAINABILITY ANALYSIS             ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )

    if os.path.exists(model_weights):
        state = torch.load(model_weights, map_location=device)
        model.load_state_dict(state)
        print(f"  Loaded specialist weights: {model_weights}")
    else:
        print(f"  ⚠ Weights not found at {model_weights}, using ImageNet defaults")

    model.to(device)
    model.eval()

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer="features.denseblock4")

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for cls_name in ["Benign", "Malignant"]:
        cls_dir = os.path.join(image_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue

        images = sorted(Path(cls_dir).glob("*.jpg"))[:max_images // 2]

        for img_path in tqdm(images, desc=f"  Grad-CAM [{cls_name}]"):
            out_name = f"gradcam_{cls_name}_{img_path.stem}.jpg"
            out_path = os.path.join(output_dir, out_name)

            try:
                gradcam.visualize(str(img_path), out_path)
                count += 1
            except Exception as e:
                print(f"  ⚠ Grad-CAM error for {img_path.name}: {e}")

    print(f"\n  ✅ Generated {count} Grad-CAM visualizations → {output_dir}")
    return output_dir


# ══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Specialist Training: YOLO11 + DenseNet201 Fine-Tuning & Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training pipeline (YOLO + DenseNet + Benchmark + Grad-CAM)
  python train_specialist.py --all

  # YOLO11 fine-tuning only
  python train_specialist.py --train-yolo --yolo-data cbis_yolo_dataset/dataset.yaml

  # DenseNet201 specialist training only
  python train_specialist.py --train-densenet --densenet-data cbis_densenet_dataset

  # Benchmark general vs specialist
  python train_specialist.py --benchmark

  # Generate Grad-CAM explanations
  python train_specialist.py --gradcam
        """,
    )

    parser.add_argument("--all", action="store_true",
                       help="Run full pipeline: YOLO → DenseNet → Benchmark → Grad-CAM")

    # YOLO training
    parser.add_argument("--train-yolo", action="store_true", help="Train YOLO11 specialist")
    parser.add_argument("--yolo-base", default="yolo11n.pt", help="Base YOLO model for transfer learning")
    parser.add_argument("--yolo-data", default="cbis_yolo_dataset/dataset.yaml", help="YOLO dataset YAML")
    parser.add_argument("--yolo-epochs", type=int, default=YOLO_EPOCHS)
    parser.add_argument("--yolo-batch", type=int, default=YOLO_BATCH)
    parser.add_argument("--yolo-imgsz", type=int, default=YOLO_IMGSZ)

    # DenseNet training
    parser.add_argument("--train-densenet", action="store_true", help="Train DenseNet201 specialist")
    parser.add_argument("--densenet-data", default="cbis_densenet_dataset", help="DenseNet crop dataset")
    parser.add_argument("--densenet-epochs", type=int, default=DENSENET_EPOCHS)
    parser.add_argument("--densenet-batch", type=int, default=DENSENET_BATCH)
    parser.add_argument("--densenet-lr", type=float, default=DENSENET_LR)

    # Benchmark
    parser.add_argument("--benchmark", action="store_true", help="Run general vs specialist benchmark")

    # Grad-CAM
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM explanations")
    parser.add_argument("--gradcam-images", default="cbis_densenet_dataset/val")
    parser.add_argument("--gradcam-max", type=int, default=20)

    # Common
    parser.add_argument("--output-dir", default="training_output", help="Output directory")

    args = parser.parse_args()

    run_all = args.all

    # ── YOLO11 Fine-Tuning ────────────────────────────────────────────
    if run_all or args.train_yolo:
        yolo_specialist = train_yolo_specialist(
            base_model=args.yolo_base,
            dataset_yaml=args.yolo_data,
            epochs=args.yolo_epochs,
            batch=args.yolo_batch,
            imgsz=args.yolo_imgsz,
            output_dir=args.output_dir,
        )

    # ── DenseNet201 Specialist Training ───────────────────────────────
    if run_all or args.train_densenet:
        trainer = DenseNetSpecialistTrainer(
            data_dir=args.densenet_data,
            epochs=args.densenet_epochs,
            batch_size=args.densenet_batch,
            learning_rate=args.densenet_lr,
            output_dir=args.output_dir,
        )
        densenet_specialist = trainer.train()

    # ── Benchmark ─────────────────────────────────────────────────────
    if run_all or args.benchmark:
        benchmark_general_vs_specialist(
            val_data_dir=f"{args.densenet_data}/val",
            specialist_weights=DENSENET_SPECIALIST_NAME,
            output_dir=args.output_dir,
        )

    # ── Grad-CAM ──────────────────────────────────────────────────────
    if run_all or args.gradcam:
        generate_gradcam_report(
            model_weights=DENSENET_SPECIALIST_NAME,
            image_dir=args.gradcam_images,
            output_dir=os.path.join(args.output_dir, "gradcam"),
            max_images=args.gradcam_max,
        )

    if not any([run_all, args.train_yolo, args.train_densenet, args.benchmark, args.gradcam]):
        parser.print_help()


if __name__ == "__main__":
    main()

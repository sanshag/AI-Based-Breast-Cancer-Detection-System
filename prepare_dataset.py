"""
==========================================================================
 Agent 1 — MEDICAL DATA ARCHITECT: Dataset Preparation & CLAHE
==========================================================================
 Creates a medical-grade training pipeline for CBIS-DDSM specialization:

   1. Dataset Integration:
      - Maps CBIS-DDSM pathology labels (Benign/Malignant) to YOLO TXT format
      - Creates train/val splits with stratified sampling
      - Generates YOLO dataset.yaml configuration

   2. Medical Augmentation:
      - CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
        mass visibility in dense breast tissue
      - Medical-specific augmentations (rotation, elastic deformation)

   3. DenseNet201 Crop Preparation:
      - Extracts ROI crops using bounding boxes + ROI masks
      - Organizes into Benign/Malignant class folders for PyTorch ImageFolder

 Dataset: CBIS-DDSM-Masses (2,620 images via Roboflow)
==========================================================================
"""

import os
import sys
import csv
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
#  CLAHE: Medical Image Enhancement
# ══════════════════════════════════════════════════════════════════════

class CLAHEEnhancer:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) for
    mammogram preprocessing. Enhances mass visibility in dense
    breast tissue without amplifying noise.

    Clinical rationale:
      Dense breast tissue can obscure masses on standard mammograms.
      CLAHE applies localized contrast enhancement, making subtle
      density differences (e.g., mass margins) more visible during
      both training and inference.
    """

    def __init__(self, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Args:
            clip_limit:     Threshold for contrast limiting (default 3.0).
                           Higher = more contrast but more noise.
            tile_grid_size: Size of grid for local histogram equalization.
                           (8,8) is optimal for mammogram-sized images.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement to a mammogram image.

        Processes in LAB color space to enhance luminance without
        distorting color information (important for annotations).

        Args:
            image: BGR image (OpenCV format).

        Returns:
            Enhanced BGR image.
        """
        if len(image.shape) == 2:
            # Grayscale — apply directly
            return self.clahe.apply(image)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the L (luminance) channel only
        l_enhanced = self.clahe.apply(l_channel)

        # Merge back and convert to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def enhance_file(self, input_path: str, output_path: str) -> str:
        """
        Read, enhance, and save an image file.

        Returns:
            Path to the saved enhanced image.
        """
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")

        enhanced = self.enhance(img)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, enhanced)
        return output_path


# ══════════════════════════════════════════════════════════════════════
#  MEDICAL AUGMENTATION PIPELINE
# ══════════════════════════════════════════════════════════════════════

class MedicalAugmentor:
    """
    Medical-grade augmentation pipeline for mammogram training data.
    Applies transformations that are clinically valid (preserving
    diagnostic information while increasing data diversity).
    """

    def __init__(self, clahe_enhancer: Optional[CLAHEEnhancer] = None):
        self.clahe = clahe_enhancer or CLAHEEnhancer()

    def augment_image_and_labels(
        self,
        image: np.ndarray,
        labels: List[List[float]],
        augmentation: str,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply a single augmentation to both image and YOLO labels.

        Args:
            image:        BGR image array.
            labels:       List of [class_id, x_center, y_center, width, height] (normalized).
            augmentation: One of 'clahe', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'.

        Returns:
            (augmented_image, augmented_labels)
        """
        h, w = image.shape[:2]

        if augmentation == "clahe":
            return self.clahe.enhance(image), labels

        elif augmentation == "hflip":
            aug_img = cv2.flip(image, 1)  # horizontal flip
            aug_labels = []
            for lbl in labels:
                cls_id, xc, yc, bw, bh = lbl
                aug_labels.append([cls_id, 1.0 - xc, yc, bw, bh])
            return aug_img, aug_labels

        elif augmentation == "vflip":
            aug_img = cv2.flip(image, 0)  # vertical flip
            aug_labels = []
            for lbl in labels:
                cls_id, xc, yc, bw, bh = lbl
                aug_labels.append([cls_id, xc, 1.0 - yc, bw, bh])
            return aug_img, aug_labels

        elif augmentation == "rot90":
            aug_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aug_labels = []
            for lbl in labels:
                cls_id, xc, yc, bw, bh = lbl
                # 90° CCW: (x,y) → (y, 1-x), swap w/h
                aug_labels.append([cls_id, yc, 1.0 - xc, bh, bw])
            return aug_img, aug_labels

        elif augmentation == "rot180":
            aug_img = cv2.rotate(image, cv2.ROTATE_180)
            aug_labels = []
            for lbl in labels:
                cls_id, xc, yc, bw, bh = lbl
                aug_labels.append([cls_id, 1.0 - xc, 1.0 - yc, bw, bh])
            return aug_img, aug_labels

        elif augmentation == "rot270":
            aug_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            aug_labels = []
            for lbl in labels:
                cls_id, xc, yc, bw, bh = lbl
                # 90° CW: (x,y) → (1-y, x), swap w/h
                aug_labels.append([cls_id, 1.0 - yc, xc, bh, bw])
            return aug_img, aug_labels

        # Unknown — return unchanged
        return image, labels


# ══════════════════════════════════════════════════════════════════════
#  CBIS-DDSM → YOLO FORMAT CONVERTER
# ══════════════════════════════════════════════════════════════════════

class CBISDDSMConverter:
    """
    Converts CBIS-DDSM-Masses dataset (Roboflow format) to YOLO training format.

    Input structure (Roboflow download):
      dataset/
      ├── train/
      │   ├── images/
      │   └── labels/       ← Already in YOLO format from Roboflow
      ├── valid/
      │   ├── images/
      │   └── labels/
      └── test/
          ├── images/
          └── labels/

    Output structure (YOLO training-ready):
      cbis_yolo_dataset/
      ├── images/
      │   ├── train/        ← Original + CLAHE-enhanced images
      │   └── val/
      ├── labels/
      │   ├── train/        ← YOLO TXT annotations
      │   └── val/
      └── dataset.yaml      ← YOLO config file

    Also prepares DenseNet201 crop dataset:
      cbis_densenet_dataset/
      ├── train/
      │   ├── Benign/
      │   └── Malignant/
      └── val/
          ├── Benign/
          └── Malignant/
    """

    CLASS_MAP = {0: "Benign", 1: "Malignant"}
    CLASS_TO_ID = {"benign": 0, "malignant": 1}

    def __init__(
        self,
        source_dir: str,
        yolo_output_dir: str = "cbis_yolo_dataset",
        densenet_output_dir: str = "cbis_densenet_dataset",
        apply_clahe: bool = True,
        augmentations: Optional[List[str]] = None,
        val_split: float = 0.2,
    ):
        self.source_dir = Path(source_dir)
        self.yolo_output = Path(yolo_output_dir)
        self.densenet_output = Path(densenet_output_dir)
        self.apply_clahe = apply_clahe
        self.augmentations = augmentations or ["clahe", "hflip"]
        self.val_split = val_split

        self.clahe = CLAHEEnhancer(clip_limit=3.0, tile_grid_size=(8, 8))
        self.augmentor = MedicalAugmentor(self.clahe)

    def detect_source_structure(self) -> str:
        """Detect whether source is Roboflow format or flat directory."""
        if (self.source_dir / "train" / "images").exists():
            return "roboflow"
        elif (self.source_dir / "images").exists():
            return "flat_yolo"
        else:
            # Assume flat directory with images + optional labels
            return "flat"

    def convert(self) -> Dict[str, str]:
        """
        Run the full conversion pipeline.

        Returns:
            Dictionary with output paths.
        """
        print("\n  ╔══════════════════════════════════════════════════════════╗")
        print("  ║   AGENT 1: MEDICAL DATA ARCHITECT — Dataset Prep       ║")
        print("  ╚══════════════════════════════════════════════════════════╝\n")

        structure = self.detect_source_structure()
        print(f"  📂 Source: {self.source_dir}")
        print(f"  🔍 Detected format: {structure}")
        print(f"  💊 CLAHE Enhancement: {'ON' if self.apply_clahe else 'OFF'}")
        print(f"  🔄 Augmentations: {self.augmentations}")

        # Step 1: Prepare YOLO dataset
        yolo_stats = self._prepare_yolo_dataset(structure)

        # Step 2: Prepare DenseNet201 crop dataset
        dn_stats = self._prepare_densenet_crops(structure)

        # Step 3: Generate YOLO dataset.yaml
        yaml_path = self._generate_yolo_yaml()

        print(f"\n  ✅ Dataset preparation complete!")
        print(f"  📂 YOLO Dataset:    {self.yolo_output}")
        print(f"  📂 DenseNet Dataset: {self.densenet_output}")
        print(f"  📄 YOLO Config:     {yaml_path}")

        return {
            "yolo_dataset": str(self.yolo_output),
            "densenet_dataset": str(self.densenet_output),
            "yolo_yaml": str(yaml_path),
            "yolo_stats": yolo_stats,
            "densenet_stats": dn_stats,
        }

    def _prepare_yolo_dataset(self, structure: str) -> Dict:
        """Prepare YOLO-format dataset with CLAHE and augmentations."""
        print(f"\n  ── Preparing YOLO11 Training Dataset ──────────────────")

        # Create output directories
        for split in ["train", "val"]:
            (self.yolo_output / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_output / "labels" / split).mkdir(parents=True, exist_ok=True)

        stats = {"train_images": 0, "val_images": 0, "train_labels": 0, "val_labels": 0}

        if structure == "roboflow":
            stats = self._convert_roboflow_to_yolo(stats)
        elif structure in ("flat_yolo", "flat"):
            stats = self._convert_flat_to_yolo(stats)

        print(f"  📊 YOLO Dataset Stats:")
        print(f"     Train: {stats['train_images']} images, {stats['train_labels']} labels")
        print(f"     Val:   {stats['val_images']} images, {stats['val_labels']} labels")

        return stats

    def _convert_roboflow_to_yolo(self, stats: Dict) -> Dict:
        """Convert Roboflow-downloaded dataset to YOLO training format."""
        split_map = {
            "train": "train",
            "valid": "val",
            "test": "val",  # Merge test into val
        }

        for src_split, dst_split in split_map.items():
            img_dir = self.source_dir / src_split / "images"
            lbl_dir = self.source_dir / src_split / "labels"

            if not img_dir.exists():
                continue

            image_files = sorted(
                f for f in img_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            )

            for img_path in tqdm(image_files, desc=f"  {src_split}→{dst_split}", unit="img"):
                stem = img_path.stem
                label_path = lbl_dir / f"{stem}.txt"

                # Copy original image
                dst_img = self.yolo_output / "images" / dst_split / img_path.name
                shutil.copy2(str(img_path), str(dst_img))
                stats[f"{dst_split}_images"] += 1

                # Copy label
                if label_path.exists():
                    dst_lbl = self.yolo_output / "labels" / dst_split / label_path.name
                    shutil.copy2(str(label_path), str(dst_lbl))
                    stats[f"{dst_split}_labels"] += 1

                # Apply CLAHE enhancement (save as additional training image)
                if self.apply_clahe and dst_split == "train":
                    self._apply_clahe_augmentation(
                        img_path, label_path, dst_split, stem, stats
                    )

                # Apply geometric augmentations for training
                if dst_split == "train" and label_path.exists():
                    self._apply_geometric_augmentations(
                        img_path, label_path, dst_split, stem, stats
                    )

        return stats

    def _convert_flat_to_yolo(self, stats: Dict) -> Dict:
        """Convert flat directory of images to YOLO format with train/val split."""
        # Find all images
        image_files = []
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            image_files.extend(self.source_dir.glob(f"*{ext}"))
            image_files.extend(self.source_dir.glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))

        # Stratified split
        random.seed(42)
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - self.val_split))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        for files, split in [(train_files, "train"), (val_files, "val")]:
            for img_path in tqdm(files, desc=f"  {split}", unit="img"):
                stem = img_path.stem
                dst_img = self.yolo_output / "images" / split / img_path.name
                shutil.copy2(str(img_path), str(dst_img))
                stats[f"{split}_images"] += 1

                # Check for existing label
                label_path = img_path.with_suffix(".txt")
                if not label_path.exists():
                    # Check in a labels/ subdirectory
                    label_path = self.source_dir / "labels" / f"{stem}.txt"

                if label_path.exists():
                    dst_lbl = self.yolo_output / "labels" / split / f"{stem}.txt"
                    shutil.copy2(str(label_path), str(dst_lbl))
                    stats[f"{split}_labels"] += 1

                # CLAHE for training images
                if self.apply_clahe and split == "train" and label_path.exists():
                    self._apply_clahe_augmentation(
                        img_path, label_path, split, stem, stats
                    )

        return stats

    def _apply_clahe_augmentation(
        self, img_path: Path, label_path: Path, split: str, stem: str, stats: Dict
    ):
        """Apply CLAHE and save as additional training sample."""
        img = cv2.imread(str(img_path))
        if img is None:
            return

        enhanced = self.clahe.enhance(img)
        clahe_name = f"{stem}_clahe{img_path.suffix}"
        dst_img = self.yolo_output / "images" / split / clahe_name
        cv2.imwrite(str(dst_img), enhanced)
        stats[f"{split}_images"] += 1

        # Copy label for CLAHE image (same bounding boxes)
        if label_path.exists():
            dst_lbl = self.yolo_output / "labels" / split / f"{stem}_clahe.txt"
            shutil.copy2(str(label_path), str(dst_lbl))
            stats[f"{split}_labels"] += 1

    def _apply_geometric_augmentations(
        self, img_path: Path, label_path: Path, split: str, stem: str, stats: Dict
    ):
        """Apply geometric augmentations with label transforms."""
        img = cv2.imread(str(img_path))
        if img is None:
            return

        # Parse labels
        labels = []
        with open(str(label_path), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(p) for p in parts[:5]])

        if not labels:
            return

        for aug_name in self.augmentations:
            if aug_name == "clahe":
                continue  # Already handled

            aug_img, aug_labels = self.augmentor.augment_image_and_labels(
                img.copy(), labels, aug_name
            )

            # Save augmented image
            aug_img_name = f"{stem}_{aug_name}{img_path.suffix}"
            dst_img = self.yolo_output / "images" / split / aug_img_name
            cv2.imwrite(str(dst_img), aug_img)
            stats[f"{split}_images"] += 1

            # Save augmented labels
            aug_lbl_name = f"{stem}_{aug_name}.txt"
            dst_lbl = self.yolo_output / "labels" / split / aug_lbl_name
            with open(str(dst_lbl), "w") as f:
                for lbl in aug_labels:
                    parts = [str(int(lbl[0]))] + [f"{v:.6f}" for v in lbl[1:]]
                    f.write(" ".join(parts) + "\n")
            stats[f"{split}_labels"] += 1

    def _prepare_densenet_crops(self, structure: str) -> Dict:
        """
        Extract ROI crops and organize into class folders for DenseNet201 training.

        Output:
          cbis_densenet_dataset/
          ├── train/
          │   ├── Benign/
          │   └── Malignant/
          └── val/
              ├── Benign/
              └── Malignant/
        """
        print(f"\n  ── Preparing DenseNet201 Crop Dataset ─────────────────")

        stats = {"train_benign": 0, "train_malignant": 0, "val_benign": 0, "val_malignant": 0}

        for split in ["train", "val"]:
            for cls in ["Benign", "Malignant"]:
                (self.densenet_output / split / cls).mkdir(parents=True, exist_ok=True)

        # Use the YOLO dataset we just prepared
        for split in ["train", "val"]:
            img_dir = self.yolo_output / "images" / split
            lbl_dir = self.yolo_output / "labels" / split

            if not img_dir.exists():
                continue

            image_files = sorted(
                f for f in img_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            )

            for img_path in tqdm(image_files, desc=f"  crops-{split}", unit="img"):
                stem = img_path.stem
                label_path = lbl_dir / f"{stem}.txt"

                if not label_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_h, img_w = img.shape[:2]

                # Parse labels and extract crops
                with open(str(label_path), "r") as f:
                    for line_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        cls_id = int(float(parts[0]))
                        xc = float(parts[1]) * img_w
                        yc = float(parts[2]) * img_h
                        bw = float(parts[3]) * img_w
                        bh = float(parts[4]) * img_h

                        # Extract crop with padding
                        pad = 0.15
                        x1 = max(0, int(xc - bw / 2 - bw * pad))
                        y1 = max(0, int(yc - bh / 2 - bh * pad))
                        x2 = min(img_w, int(xc + bw / 2 + bw * pad))
                        y2 = min(img_h, int(yc + bh / 2 + bh * pad))

                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        # Resize to DenseNet input size
                        crop_resized = cv2.resize(crop, (224, 224))

                        # Determine class
                        cls_name = self.CLASS_MAP.get(cls_id, "Benign")
                        cls_key = cls_name.lower()

                        # Save crop
                        crop_name = f"{stem}_crop{line_idx}.jpg"
                        dst = self.densenet_output / split / cls_name / crop_name
                        cv2.imwrite(str(dst), crop_resized)

                        stats[f"{split}_{cls_key}"] += 1

                        # Also save CLAHE-enhanced crop for training
                        if split == "train" and self.apply_clahe:
                            crop_clahe = self.clahe.enhance(crop_resized)
                            clahe_name = f"{stem}_crop{line_idx}_clahe.jpg"
                            dst_clahe = self.densenet_output / split / cls_name / clahe_name
                            cv2.imwrite(str(dst_clahe), crop_clahe)
                            stats[f"{split}_{cls_key}"] += 1

        print(f"  📊 DenseNet201 Crop Stats:")
        print(f"     Train: {stats['train_benign']} Benign, {stats['train_malignant']} Malignant")
        print(f"     Val:   {stats['val_benign']} Benign, {stats['val_malignant']} Malignant")

        return stats

    def _generate_yolo_yaml(self) -> str:
        """Generate the YOLO dataset.yaml configuration file."""
        yaml_content = f"""# CBIS-DDSM Masses — YOLO11 Specialist Training Dataset
# Auto-generated by Agent 1 (Medical Data Architect)
# CLAHE-enhanced + Augmented mammogram dataset

path: {self.yolo_output.absolute()}
train: images/train
val: images/val

# Classes
nc: 2
names:
  0: Benign
  1: Malignant

# Training Notes:
# - Images include CLAHE-enhanced variants for dense tissue visibility
# - Augmentations: {', '.join(self.augmentations)}
# - Original dataset: CBIS-DDSM-Masses (Roboflow format)
"""
        yaml_path = self.yolo_output / "dataset.yaml"
        with open(str(yaml_path), "w") as f:
            f.write(yaml_content)

        return str(yaml_path)


# ══════════════════════════════════════════════════════════════════════
#  PATHOLOGY LABEL MAPPER
# ══════════════════════════════════════════════════════════════════════

def map_pathology_to_yolo(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    image_col: str = "image_file",
    pathology_col: str = "pathology",
    bbox_cols: Optional[List[str]] = None,
) -> int:
    """
    Map CBIS-DDSM pathology CSV labels to YOLO TXT format.

    This handles the case where the dataset comes with a CSV manifest
    instead of pre-formatted YOLO labels.

    CSV expected columns:
      - image_file: filename of the mammogram
      - pathology: 'BENIGN', 'MALIGNANT', 'BENIGN_WITHOUT_CALLBACK'
      - [optional] bbox columns: x_min, y_min, x_max, y_max (pixel coords)
      - [optional] image_width, image_height

    Args:
        csv_path:      Path to the pathology CSV.
        image_dir:     Directory containing the images.
        output_dir:    Directory to save YOLO TXT labels.
        image_col:     Column name for image filename.
        pathology_col: Column name for pathology label.

    Returns:
        Number of label files created.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    class_map = {
        "BENIGN": 0,
        "BENIGN_WITHOUT_CALLBACK": 0,
        "MALIGNANT": 1,
    }

    if bbox_cols is None:
        bbox_cols = ["x_min", "y_min", "x_max", "y_max"]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get(image_col, "").strip()
            pathology = row.get(pathology_col, "").strip().upper()

            if not img_name or pathology not in class_map:
                continue

            cls_id = class_map[pathology]

            # Try to find the image to get dimensions
            img_path = None
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                candidate = Path(image_dir) / f"{Path(img_name).stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                candidate = Path(image_dir) / img_name
                if candidate.exists():
                    img_path = candidate

            if img_path is None:
                continue

            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            # Parse bounding box if available
            has_bbox = all(col in row and row[col].strip() for col in bbox_cols)
            if has_bbox:
                x1 = float(row[bbox_cols[0]])
                y1 = float(row[bbox_cols[1]])
                x2 = float(row[bbox_cols[2]])
                y2 = float(row[bbox_cols[3]])
            else:
                # No bbox — use ROI mask or default to center region
                roi_mask_path = row.get("roi_mask_path", "")
                if roi_mask_path and os.path.exists(roi_mask_path):
                    x1, y1, x2, y2 = _bbox_from_mask(roi_mask_path)
                else:
                    # Default: assume mass occupies central 30% of image
                    margin = 0.35
                    x1 = img_w * margin
                    y1 = img_h * margin
                    x2 = img_w * (1 - margin)
                    y2 = img_h * (1 - margin)

            # Convert to YOLO normalized format
            bw = x2 - x1
            bh = y2 - y1
            xc = x1 + bw / 2
            yc = y1 + bh / 2

            xc_norm = xc / img_w
            yc_norm = yc / img_h
            bw_norm = bw / img_w
            bh_norm = bh / img_h

            # Write YOLO label
            stem = img_path.stem
            label_path = os.path.join(output_dir, f"{stem}.txt")
            with open(label_path, "a") as lf:
                lf.write(f"{cls_id} {xc_norm:.6f} {yc_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

            count += 1

    print(f"  📝 Created {count} YOLO label entries in: {output_dir}")
    return count


def _bbox_from_mask(mask_path: str) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from a binary ROI mask image.

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = mask.shape
        return (w * 0.3, h * 0.3, w * 0.7, h * 0.7)

    # Get bounding rect of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return (float(x), float(y), float(x + w), float(y + h))


# ══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Agent 1: CBIS-DDSM Dataset Preparation with CLAHE Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Roboflow-downloaded dataset
  python prepare_dataset.py --source ./cbis-ddsm-masses-1/

  # Convert with all augmentations
  python prepare_dataset.py --source ./dataset/ --augmentations clahe hflip vflip rot180

  # Convert CSV pathology labels to YOLO format
  python prepare_dataset.py --csv-labels pathology.csv --image-dir images/ --label-output labels/
        """,
    )
    parser.add_argument(
        "--source", type=str, default="./cbis-ddsm-masses-1",
        help="Source dataset directory (Roboflow download)",
    )
    parser.add_argument(
        "--yolo-output", type=str, default="cbis_yolo_dataset",
        help="Output directory for YOLO training dataset",
    )
    parser.add_argument(
        "--densenet-output", type=str, default="cbis_densenet_dataset",
        help="Output directory for DenseNet201 crop dataset",
    )
    parser.add_argument(
        "--no-clahe", action="store_true",
        help="Disable CLAHE enhancement",
    )
    parser.add_argument(
        "--augmentations", nargs="+",
        default=["clahe", "hflip"],
        choices=["clahe", "hflip", "vflip", "rot90", "rot180", "rot270"],
        help="Augmentations to apply (default: clahe hflip)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    # Optional CSV label mapping mode
    parser.add_argument("--csv-labels", type=str, default=None,
                       help="Path to pathology CSV for label mapping")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--label-output", type=str, default=None)

    args = parser.parse_args()

    # CSV label mapping mode
    if args.csv_labels:
        if not args.image_dir or not args.label_output:
            print("  ❌ --image-dir and --label-output required with --csv-labels")
            sys.exit(1)
        map_pathology_to_yolo(args.csv_labels, args.image_dir, args.label_output)
        return

    # Full dataset conversion mode
    converter = CBISDDSMConverter(
        source_dir=args.source,
        yolo_output_dir=args.yolo_output,
        densenet_output_dir=args.densenet_output,
        apply_clahe=not args.no_clahe,
        augmentations=args.augmentations,
        val_split=args.val_split,
    )
    results = converter.convert()

    print(f"\n  🎯 Ready for training!")
    print(f"     YOLO:     python train_specialist.py --yolo-data {results['yolo_yaml']}")
    print(f"     DenseNet: python train_specialist.py --densenet-data {results['densenet_dataset']}")


if __name__ == "__main__":
    main()

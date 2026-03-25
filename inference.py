"""
==========================================================================
 Agent 2 — AI & INFERENCE ENGINEER: YOLO11 + DenseNet201 Pipeline
==========================================================================
 Two-stage inference workflow:
   Stage 1: YOLO11 whole-mammogram mass detection (ultralytics)
   Stage 2: DenseNet201 high-resolution crop classification (torchvision)

 Additional features:
   - Automated ROI cropping from YOLO bounding boxes
   - Test-Time Augmentation (TTA) for boosted accuracy
   - YOLO TXT label distillation for local model training
==========================================================================
"""

import os
import time
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from copy import deepcopy

import cv2
import numpy as np

# ── Deep Learning Imports ──────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image as PILImage

from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
    DENSENET_INPUT_SIZE, DENSENET_MODEL_PATH,
    YOLO_WEIGHT, DENSENET_WEIGHT,
    ENSEMBLE_AGREEMENT_THRESHOLD, SAFETY_FIRST_BIRADS,
    TTA_ENABLED, TTA_TRANSFORMS,
    SUPPORTED_EXTENSIONS, CROPS_SUBFOLDER, LABELS_SUBFOLDER,
    Classification, BoundingBox, StageOneResult, StageTwoResult,
    EnsembleResult, ImageResult,
    estimate_mass_size_mm, compute_feature_risk_score,
)


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1: YOLO11 Mass Detection
# ══════════════════════════════════════════════════════════════════════

class YOLODetector:
    """
    Wraps the ultralytics YOLO11 model for whole-mammogram mass detection.
    Processes full-resolution mammograms and returns bounding boxes with
    class predictions and confidence scores.
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH,
                 conf_threshold: float = YOLO_CONF_THRESHOLD,
                 iou_threshold: float = YOLO_IOU_THRESHOLD):
        print(f"  🔧 Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  ✅ YOLO11 ready on {self.device}")

    def detect(self, image_path: str) -> List[StageOneResult]:
        """
        Run YOLO11 inference on a mammogram image.

        Returns:
            List of StageOneResult with bounding boxes and class predictions.
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get class name from model
                    class_name = self.model.names.get(class_id, f"class_{class_id}")

                    # Convert to center-format bounding box
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    bbox = BoundingBox(
                        x_center=float(cx),
                        y_center=float(cy),
                        width=float(w),
                        height=float(h),
                    )

                    detections.append(StageOneResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bounding_box=bbox,
                    ))

        return detections


# ══════════════════════════════════════════════════════════════════════
#  AUTOMATED ROI CROPPING
# ══════════════════════════════════════════════════════════════════════

def extract_roi_crops(
    image_path: str,
    detections: List[StageOneResult],
    output_dir: str,
    padding_ratio: float = 0.15,
) -> List[StageOneResult]:
    """
    Extract high-resolution ROI crops from each YOLO detection bounding box.

    Adds contextual padding around each crop for better classification.
    Saves crops to disk and updates each detection's crop_path.

    Args:
        image_path:    Path to the full mammogram image.
        detections:    List of YOLO StageOneResults with bounding boxes.
        output_dir:    Directory to save crop images.
        padding_ratio: Extra context padding as fraction of box size.

    Returns:
        Updated detections with crop_path set.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    for idx, det in enumerate(detections):
        bbox = det.bounding_box
        x_min, y_min, x_max, y_max = bbox.as_ints()

        # Add contextual padding
        pad_x = int(bbox.width * padding_ratio)
        pad_y = int(bbox.height * padding_ratio)
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(img_w, x_max + pad_x)
        y_max = min(img_h, y_max + pad_y)

        # Extract crop
        crop = img[y_min:y_max, x_min:x_max]

        if crop.size == 0:
            continue

        # Save crop
        crop_filename = f"{stem}_crop_{idx + 1}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        det.crop_path = os.path.abspath(crop_path)

    return detections


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2: DenseNet201 Crop Classification
# ══════════════════════════════════════════════════════════════════════

class DenseNetClassifier:
    """
    Uses a pre-trained DenseNet201 (torchvision) fine-tuneable for
    binary classification: Benign vs. Malignant.

    Features:
      - Pre-trained ImageNet weights as feature extractor
      - Modified final classifier for binary output
      - Test-Time Augmentation support
    """

    def __init__(self, model_path: str = DENSENET_MODEL_PATH,
                 input_size: int = DENSENET_INPUT_SIZE):
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  🔧 Loading DenseNet201 classifier...")

        # Load DenseNet201 with pre-trained weights
        self.model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

        # Replace final classifier for binary (Benign/Malignant)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),  # [benign_logit, malignant_logit]
        )

        # Load custom weights if provided
        if model_path and os.path.exists(model_path):
            print(f"  📦 Loading custom weights: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"  ℹ️  Using ImageNet pre-trained DenseNet201 (no custom weights)")

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.base_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"  ✅ DenseNet201 ready on {self.device}")

    def _apply_tta_transform(self, image: PILImage.Image, transform_name: str) -> PILImage.Image:
        """Apply a single TTA transform to a PIL image."""
        if transform_name == "original":
            return image
        elif transform_name == "hflip":
            return image.transpose(PILImage.FLIP_LEFT_RIGHT)
        elif transform_name == "vflip":
            return image.transpose(PILImage.FLIP_TOP_BOTTOM)
        elif transform_name == "rot90":
            return image.rotate(90, expand=True)
        elif transform_name == "rot180":
            return image.rotate(180)
        elif transform_name == "rot270":
            return image.rotate(270, expand=True)
        return image

    def classify(self, crop_path: str, use_tta: bool = True) -> StageTwoResult:
        """
        Classify a single ROI crop as Benign or Malignant.

        Args:
            crop_path: Path to the cropped mass image.
            use_tta:   Whether to apply Test-Time Augmentation.

        Returns:
            StageTwoResult with classification, probabilities, and TTA scores.
        """
        pil_image = PILImage.open(crop_path).convert("RGB")

        if use_tta:
            return self._classify_with_tta(pil_image)
        else:
            return self._classify_single(pil_image)

    def _classify_single(self, pil_image: PILImage.Image) -> StageTwoResult:
        """Classify a single image without TTA."""
        tensor = self.base_transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        benign_prob = float(probs[0])
        malignant_prob = float(probs[1])

        classification = (
            Classification.MALIGNANT if malignant_prob > benign_prob
            else Classification.BENIGN
        )
        confidence = max(benign_prob, malignant_prob)

        return StageTwoResult(
            classification=classification,
            confidence=round(confidence, 4),
            benign_prob=round(benign_prob, 4),
            malignant_prob=round(malignant_prob, 4),
        )

    def _classify_with_tta(self, pil_image: PILImage.Image) -> StageTwoResult:
        """
        Classify with Test-Time Augmentation.
        Runs the image through multiple transforms and averages predictions
        for a 2-3% accuracy boost.
        """
        all_probs = []
        tta_scores: Dict[str, float] = {}

        for transform_name in TTA_TRANSFORMS:
            augmented = self._apply_tta_transform(pil_image, transform_name)
            tensor = self.base_transform(augmented).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            all_probs.append(probs)
            tta_scores[transform_name] = float(probs[1])  # malignant probability

        # Average across all TTA augmentations
        avg_probs = np.mean(all_probs, axis=0)
        benign_prob = float(avg_probs[0])
        malignant_prob = float(avg_probs[1])

        classification = (
            Classification.MALIGNANT if malignant_prob > benign_prob
            else Classification.BENIGN
        )
        confidence = max(benign_prob, malignant_prob)

        return StageTwoResult(
            classification=classification,
            confidence=round(confidence, 4),
            benign_prob=round(benign_prob, 4),
            malignant_prob=round(malignant_prob, 4),
            tta_scores=tta_scores,
        )


# ══════════════════════════════════════════════════════════════════════
#  ENSEMBLE: Weighted Voting Combiner
# ══════════════════════════════════════════════════════════════════════

def compute_ensemble_decision(
    yolo_result: StageOneResult,
    densenet_result: Optional[StageTwoResult],
    detection_id: int,
    img_width: int,
) -> EnsembleResult:
    """
    Combine YOLO11 and DenseNet201 predictions using weighted voting.

    Weights: YOLO = 0.40, DenseNet = 0.60 (DenseNet sees high-res crops)

    Rules:
      - Both Malignant >70% conf → BI-RADS 5, Priority 10
      - Disagreement → Safety First BI-RADS 3
    """
    yolo_is_malignant = "malignant" in yolo_result.class_name.lower()
    yolo_mal_score = yolo_result.confidence if yolo_is_malignant else (1 - yolo_result.confidence)

    if densenet_result is not None:
        dn_mal_score = densenet_result.malignant_prob

        # Weighted combination
        weighted_yolo = yolo_mal_score * YOLO_WEIGHT
        weighted_dn = dn_mal_score * DENSENET_WEIGHT
        ensemble_mal_score = weighted_yolo + weighted_dn
        ensemble_ben_score = 1.0 - ensemble_mal_score

        if ensemble_mal_score > 0.5:
            ensemble_classification = Classification.MALIGNANT
            ensemble_confidence = ensemble_mal_score
        else:
            ensemble_classification = Classification.BENIGN
            ensemble_confidence = ensemble_ben_score
    else:
        # Single model fallback
        weighted_yolo = yolo_mal_score * 1.0
        weighted_dn = 0.0
        if yolo_is_malignant:
            ensemble_classification = Classification.MALIGNANT
            ensemble_confidence = yolo_result.confidence
        else:
            ensemble_classification = Classification.BENIGN
            ensemble_confidence = yolo_result.confidence

    # Estimate mass size
    mass_size_mm = estimate_mass_size_mm(yolo_result.bounding_box, img_width)

    # Infer morphological features from detection context
    detected_features = _infer_features(yolo_result, mass_size_mm)
    feature_risk = compute_feature_risk_score(detected_features)

    return EnsembleResult(
        detection_id=detection_id,
        yolo_result=yolo_result,
        densenet_result=densenet_result,
        ensemble_classification=ensemble_classification,
        ensemble_confidence=round(ensemble_confidence, 4),
        yolo_weighted_score=round(weighted_yolo, 4),
        densenet_weighted_score=round(weighted_dn, 4),
        mass_size_mm=mass_size_mm,
        detected_features=detected_features,
        feature_risk_score=feature_risk,
    )


def _infer_features(yolo_result: StageOneResult, mass_size_mm: float) -> Dict[str, str]:
    """
    Infer morphological features from detection attributes.
    Maps YOLO class names and box geometry to clinical descriptors.
    """
    features = {}

    # Size category
    if mass_size_mm < 10:
        features["Size_mm"] = "< 10"
    elif mass_size_mm < 20:
        features["Size_mm"] = "10-20"
    elif mass_size_mm < 30:
        features["Size_mm"] = "20-30"
    else:
        features["Size_mm"] = "> 30"

    # Shape estimation from aspect ratio
    bbox = yolo_result.bounding_box
    aspect_ratio = bbox.width / max(bbox.height, 1)
    if 0.85 <= aspect_ratio <= 1.15:
        features["Shape"] = "Round"
    elif 0.6 <= aspect_ratio <= 1.4:
        features["Shape"] = "Oval"
    else:
        features["Shape"] = "Irregular"

    # Margin estimation from class name keywords
    class_lower = yolo_result.class_name.lower()
    if "spiculated" in class_lower or "spic" in class_lower:
        features["Margin"] = "Spiculated"
    elif "indistinct" in class_lower:
        features["Margin"] = "Indistinct"
    elif "malignant" in class_lower:
        features["Margin"] = "Microlobulated"
    else:
        features["Margin"] = "Circumscribed"

    # Density estimation (confidence proxy)
    if yolo_result.confidence > 0.8:
        features["Density"] = "High"
    elif yolo_result.confidence > 0.5:
        features["Density"] = "Equal"
    else:
        features["Density"] = "Low"

    return features


# ══════════════════════════════════════════════════════════════════════
#  LABEL DISTILLATION: Save Detections as YOLO TXT
# ══════════════════════════════════════════════════════════════════════

def distill_yolo_labels(
    detections: List[StageOneResult],
    image_path: str,
    img_width: int,
    img_height: int,
    output_dir: str,
    class_map: Optional[Dict[str, int]] = None,
) -> str:
    """
    Save YOLO detections as YOLO-format TXT labels for local model training.

    Format per line: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].

    Args:
        detections:  Detection results.
        image_path:  Original image path (for naming).
        img_width:   Image width in pixels.
        img_height:  Image height in pixels.
        output_dir:  Directory to save label files.
        class_map:   Optional mapping of class names → class IDs.

    Returns:
        Path to the saved label file.
    """
    if class_map is None:
        class_map = {"benign": 0, "malignant": 1}

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    label_path = os.path.join(output_dir, f"{stem}.txt")

    lines = []
    for det in detections:
        # Determine class ID
        cls_name = det.class_name.lower()
        cls_id = class_map.get(cls_name, det.class_id)

        # Normalize coordinates
        nx, ny, nw, nh = det.bounding_box.to_yolo_normalized(img_width, img_height)
        lines.append(f"{cls_id} {nx} {ny} {nw} {nh}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    return os.path.abspath(label_path)


# ══════════════════════════════════════════════════════════════════════
#  MASTER INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════

class EnsembleInferenceEngine:
    """
    Master engine combining YOLO11 detection + DenseNet201 classification
    into a unified two-stage ensemble pipeline.
    """

    def __init__(self,
                 yolo_path: str = YOLO_MODEL_PATH,
                 densenet_path: str = DENSENET_MODEL_PATH,
                 use_tta: bool = TTA_ENABLED,
                 distill_labels: bool = False):
        self.use_tta = use_tta
        self.distill_labels = distill_labels

        # Initialize Stage 1: YOLO11
        self.yolo = YOLODetector(model_path=yolo_path)

        # Initialize Stage 2: DenseNet201
        self.densenet = DenseNetClassifier(model_path=densenet_path)

        print(f"  🔗 Ensemble Pipeline Ready (TTA: {'ON' if use_tta else 'OFF'})")

    def run_inference(self, image_path: str, output_root: str) -> ImageResult:
        """
        Run the full two-stage inference pipeline on a single mammogram.

        Pipeline:
          1. YOLO11 detects masses in the whole mammogram
          2. Crops are extracted from each detection
          3. DenseNet201 classifies each crop
          4. Weighted voting produces ensemble results
          5. (Optional) Labels are distilled for training

        Returns:
            ImageResult with all ensemble detections and clinical scores.
        """
        start_time = time.time()

        # Read image dimensions
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_h, img_w = img.shape[:2]
        filename = Path(image_path).name

        # ── Stage 1: YOLO11 Detection ─────────────────────────────────
        try:
            stage1_results = self.yolo.detect(image_path)
        except Exception as e:
            print(f"  ⚠ YOLO11 error for {filename}: {e}")
            traceback.print_exc()
            return ImageResult(
                image_filename=filename,
                image_path=image_path,
                image_width=img_w,
                image_height=img_h,
                processing_time_sec=round(time.time() - start_time, 2),
                tta_enabled=self.use_tta,
            )

        if not stage1_results:
            return ImageResult(
                image_filename=filename,
                image_path=image_path,
                image_width=img_w,
                image_height=img_h,
                processing_time_sec=round(time.time() - start_time, 2),
                tta_enabled=self.use_tta,
            )

        # ── Automated ROI Cropping ────────────────────────────────────
        crops_dir = os.path.join(output_root, CROPS_SUBFOLDER)
        stage1_results = extract_roi_crops(
            image_path, stage1_results, crops_dir
        )

        # ── Stage 2: DenseNet201 Classification ───────────────────────
        ensemble_results = []
        for idx, yolo_det in enumerate(stage1_results, start=1):
            densenet_result = None
            if yolo_det.crop_path and os.path.exists(yolo_det.crop_path):
                try:
                    densenet_result = self.densenet.classify(
                        yolo_det.crop_path, use_tta=self.use_tta
                    )
                except Exception as e:
                    print(f"  ⚠ DenseNet201 error for crop {idx}: {e}")

            # ── Ensemble Voting ───────────────────────────────────────
            ensemble = compute_ensemble_decision(
                yolo_result=yolo_det,
                densenet_result=densenet_result,
                detection_id=idx,
                img_width=img_w,
            )
            ensemble_results.append(ensemble)

        # ── Label Distillation (optional) ─────────────────────────────
        if self.distill_labels:
            labels_dir = os.path.join(output_root, LABELS_SUBFOLDER)
            distill_yolo_labels(
                stage1_results, image_path, img_w, img_h, labels_dir
            )

        elapsed = round(time.time() - start_time, 2)

        return ImageResult(
            image_filename=filename,
            image_path=image_path,
            image_width=img_w,
            image_height=img_h,
            stage1_detections=stage1_results,
            ensemble_results=ensemble_results,
            processing_time_sec=elapsed,
            tta_enabled=self.use_tta,
        )


# ══════════════════════════════════════════════════════════════════════
#  UTILITY: Image Collection
# ══════════════════════════════════════════════════════════════════════

def collect_images(folder_path: str) -> List[str]:
    """
    Recursively collect all supported image files from a folder.

    Returns:
        Sorted list of absolute paths to image files.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))
        images.extend(folder.rglob(f"*{ext.upper()}"))

    # Deduplicate
    unique = sorted(set(str(p.resolve()) for p in images))
    return unique

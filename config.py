"""
==========================================================================
 Agent 1 — CLINICAL ARCHITECT: System Configuration & Ensemble Schema
==========================================================================
 Defines the complete data schema for a two-stage inference pipeline:
   Stage 1 → YOLO11 whole-mammogram mass detection
   Stage 2 → DenseNet201 high-resolution crop classification

 Clinical logic:
   - Weighted Voting System (ensemble agreement / disagreement)
   - Feature-to-risk-score mapping (Shape, Margin → clinical scores)
   - BI-RADS assignment & Triage Priority (1-10 numeric scale)

 Data source: CBIS-DDSM-Masses dataset
==========================================================================
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# ── Specialist Weight Filenames ─────────────────────────────────────────
YOLO_SPECIALIST_FILENAME: str = "yolo11_cbis_specialist.pt"
DENSENET_SPECIALIST_FILENAME: str = "densenet_cbis_specialist.pt"
YOLO_GENERAL_FILENAME: str = "yolo11n.pt"


def _resolve_model_path(env_key: str, specialist_name: str, fallback: str) -> str:
    """
    Auto-detect the best available model weights.
    Priority: ENV override > Specialist weights > General weights > fallback
    """
    env_val = os.getenv(env_key, "")
    if env_val and os.path.exists(env_val):
        return env_val
    if os.path.exists(specialist_name):
        return specialist_name
    return fallback


# ── Model Paths & Constants ────────────────────────────────────────────
YOLO_MODEL_PATH: str = _resolve_model_path(
    "YOLO_MODEL_PATH", YOLO_SPECIALIST_FILENAME, YOLO_GENERAL_FILENAME
)
DENSENET_MODEL_PATH: str = _resolve_model_path(
    "DENSENET_MODEL_PATH", DENSENET_SPECIALIST_FILENAME, ""
)
YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
DENSENET_INPUT_SIZE: int = int(os.getenv("DENSENET_INPUT_SIZE", "224"))

# ── Ensemble Voting Weights ───────────────────────────────────────────
YOLO_WEIGHT: float = float(os.getenv("YOLO_WEIGHT", "0.40"))
DENSENET_WEIGHT: float = float(os.getenv("DENSENET_WEIGHT", "0.60"))
ENSEMBLE_AGREEMENT_THRESHOLD: float = 0.70  # Both models >70% → strong signal
SAFETY_FIRST_BIRADS: int = 3  # Disagreement default

# ── Test-Time Augmentation ─────────────────────────────────────────────
TTA_ENABLED: bool = os.getenv("TTA_ENABLED", "true").lower() == "true"
TTA_TRANSFORMS: List[str] = ["original", "hflip", "vflip", "rot90", "rot180", "rot270"]

# ── File & Directory Constants ──────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")
INPUT_FOLDER: str = "input_mammograms"
OUTPUT_FOLDER: str = "output_results"
ANNOTATED_SUBFOLDER: str = "annotated_images"
REPORTS_SUBFOLDER: str = "pdf_reports"
CROPS_SUBFOLDER: str = "roi_crops"
LABELS_SUBFOLDER: str = "yolo_labels"
AUDIT_SUBFOLDER: str = "audit_logs"
SUMMARY_CSV_NAME: str = "summary.csv"

# ── Visual Annotation Config ──────────────────────────────────────────
YOLO_BENIGN_COLOR: tuple = (0, 200, 0)        # Green (BGR) — YOLO detection benign
YOLO_MALIGNANT_COLOR: tuple = (0, 0, 255)     # Red (BGR) — YOLO detection malignant
ENSEMBLE_BENIGN_COLOR: tuple = (200, 255, 200) # Light green — ensemble benign
ENSEMBLE_MALIGNANT_COLOR: tuple = (0, 0, 180)  # Dark red — ensemble malignant
ENSEMBLE_UNCERTAIN_COLOR: tuple = (0, 165, 255) # Orange — ensemble uncertain
YOLO_BOX_THICKNESS: int = 2
ENSEMBLE_BOX_THICKNESS: int = 5
FONT_SCALE: float = 0.65
LABEL_BG_ALPHA: float = 0.65


# ── Clinical Enumerations ──────────────────────────────────────────────
class Classification(Enum):
    """Mass classification."""
    BENIGN = "Benign"
    MALIGNANT = "Malignant"
    UNKNOWN = "Unknown"

    @classmethod
    def from_label(cls, label: str) -> "Classification":
        """Robustly map any label string to a Classification."""
        normalized = label.strip().lower()
        if "malignant" in normalized or "malig" in normalized:
            return cls.MALIGNANT
        elif "benign" in normalized:
            return cls.BENIGN
        return cls.UNKNOWN


class TriagePriority(Enum):
    """Clinical triage urgency level."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    ROUTINE = "ROUTINE"


class BiRads(Enum):
    """ACR BI-RADS assessment categories."""
    BIRADS_0 = (0, "Incomplete — Need Additional Imaging")
    BIRADS_1 = (1, "Negative")
    BIRADS_2 = (2, "Benign Finding")
    BIRADS_3 = (3, "Probably Benign — Short-interval Follow-up")
    BIRADS_4 = (4, "Suspicious Abnormality — Biopsy Should Be Considered")
    BIRADS_5 = (5, "Highly Suggestive of Malignancy — Appropriate Action Required")

    def __init__(self, score: int, description: str):
        self.score = score
        self.description = description

    def __str__(self):
        return f"BI-RADS {self.score}: {self.description}"


# ── Clinical Feature-to-Risk Score Mapping ─────────────────────────────
# Maps morphological detection attributes to clinical risk scores (0–10).
# Higher scores = higher malignancy suspicion.

FEATURE_RISK_MAP: Dict[str, Dict[str, float]] = {
    "Shape": {
        "Round": 1.0,
        "Oval": 1.5,
        "Lobular": 4.0,
        "Irregular": 8.0,
        "Distorted": 9.0,
    },
    "Margin": {
        "Circumscribed": 1.0,
        "Obscured": 3.0,
        "Microlobulated": 5.5,
        "Indistinct": 7.0,
        "Spiculated": 9.5,
    },
    "Density": {
        "Fat-containing": 0.5,
        "Low": 1.5,
        "Equal": 3.0,
        "High": 6.5,
    },
    "Size_mm": {
        "< 10": 1.0,
        "10-20": 3.0,
        "20-30": 5.0,
        "> 30": 7.5,
    },
}


def compute_feature_risk_score(features: Dict[str, str]) -> float:
    """
    Compute a composite clinical risk score (0–10) from detected features.

    Args:
        features: e.g. {"Shape": "Irregular", "Margin": "Spiculated"}

    Returns:
        Weighted average risk score.
    """
    scores = []
    weights = {"Shape": 1.5, "Margin": 2.0, "Density": 1.0, "Size_mm": 1.0}

    for category, value in features.items():
        if category in FEATURE_RISK_MAP and value in FEATURE_RISK_MAP[category]:
            weight = weights.get(category, 1.0)
            scores.append((FEATURE_RISK_MAP[category][value], weight))

    if not scores:
        return 5.0  # default moderate risk

    total = sum(s * w for s, w in scores)
    total_weight = sum(w for _, w in scores)
    return round(total / total_weight, 2)


# ── Data Schema ─────────────────────────────────────────────────────────
@dataclass
class BoundingBox:
    """Pixel-coordinate bounding box for a detected mass."""
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def x_min(self) -> float:
        return self.x_center - self.width / 2

    @property
    def y_min(self) -> float:
        return self.y_center - self.height / 2

    @property
    def x_max(self) -> float:
        return self.x_center + self.width / 2

    @property
    def y_max(self) -> float:
        return self.y_center + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_ints(self) -> tuple:
        """Return (x_min, y_min, x_max, y_max) as integers for drawing."""
        return (int(self.x_min), int(self.y_min),
                int(self.x_max), int(self.y_max))

    def to_yolo_normalized(self, img_w: int, img_h: int) -> tuple:
        """Convert to YOLO format: (x_center, y_center, width, height) normalized."""
        return (
            round(self.x_center / img_w, 6),
            round(self.y_center / img_h, 6),
            round(self.width / img_w, 6),
            round(self.height / img_h, 6),
        )


@dataclass
class StageOneResult:
    """YOLO11 detection result for a single mass region."""
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    crop_path: Optional[str] = None


@dataclass
class StageTwoResult:
    """DenseNet201 classification result for a cropped ROI."""
    classification: Classification
    confidence: float
    benign_prob: float
    malignant_prob: float
    tta_scores: Optional[Dict[str, float]] = None  # per-augmentation scores


@dataclass
class EnsembleResult:
    """Combined result from both YOLO11 and DenseNet201."""
    detection_id: int
    yolo_result: StageOneResult
    densenet_result: Optional[StageTwoResult]
    ensemble_classification: Classification
    ensemble_confidence: float
    yolo_weighted_score: float
    densenet_weighted_score: float
    birads: Optional[BiRads] = None
    triage: Optional[TriagePriority] = None
    triage_priority_score: int = 1  # 1-10 numeric scale
    mass_size_mm: Optional[float] = None
    feature_risk_score: float = 5.0
    detected_features: Optional[Dict[str, str]] = None
    explainability_text: str = ""

    def __post_init__(self):
        """Auto-compute clinical scores after creation."""
        if self.birads is None:
            self.birads = compute_ensemble_birads(
                self.yolo_result, self.densenet_result,
                self.ensemble_classification, self.ensemble_confidence
            )
        if self.triage is None:
            self.triage = compute_triage(self.birads, self.ensemble_confidence)
            self.triage_priority_score = compute_triage_priority_score(
                self.birads, self.ensemble_confidence
            )
        if not self.explainability_text:
            self.explainability_text = generate_explainability(self)


@dataclass
class ImageResult:
    """Complete analysis result for a single mammogram image."""
    image_filename: str
    image_path: str
    image_width: int
    image_height: int
    stage1_detections: List[StageOneResult] = field(default_factory=list)
    ensemble_results: List[EnsembleResult] = field(default_factory=list)
    annotated_path: Optional[str] = None
    report_path: Optional[str] = None
    processing_time_sec: float = 0.0
    tta_enabled: bool = False

    @property
    def total_detections(self) -> int:
        return len(self.ensemble_results)

    @property
    def has_malignant(self) -> bool:
        return any(
            r.ensemble_classification == Classification.MALIGNANT
            for r in self.ensemble_results
        )

    @property
    def highest_triage(self) -> Optional[TriagePriority]:
        priority_order = [
            TriagePriority.CRITICAL, TriagePriority.HIGH,
            TriagePriority.MODERATE, TriagePriority.LOW,
            TriagePriority.ROUTINE,
        ]
        for p in priority_order:
            if any(r.triage == p for r in self.ensemble_results):
                return p
        return None

    @property
    def highest_birads(self) -> Optional[BiRads]:
        if not self.ensemble_results:
            return None
        return max(self.ensemble_results, key=lambda r: r.birads.score).birads

    @property
    def max_triage_score(self) -> int:
        if not self.ensemble_results:
            return 1
        return max(r.triage_priority_score for r in self.ensemble_results)


# ── Weighted Voting / Ensemble BI-RADS ──────────────────────────────────

def compute_ensemble_birads(
    yolo_result: StageOneResult,
    densenet_result: Optional[StageTwoResult],
    ensemble_class: Classification,
    ensemble_conf: float,
) -> BiRads:
    """
    Assign BI-RADS using the Weighted Voting System.

    AGREEMENT RULE:
      Both models flag 'Malignant' with >70% confidence → BI-RADS 5, Priority 10
    DISAGREEMENT RULE:
      Models disagree → Safety First → BI-RADS 3 for radiologist review
    SINGLE-MODEL FALLBACK:
      If DenseNet unavailable, use YOLO-only scoring
    """
    yolo_is_malignant = "malignant" in yolo_result.class_name.lower()
    yolo_conf = yolo_result.confidence

    if densenet_result is not None:
        dn_is_malignant = densenet_result.classification == Classification.MALIGNANT
        dn_conf = densenet_result.confidence

        # AGREEMENT: Both say Malignant with high confidence
        if (yolo_is_malignant and dn_is_malignant
                and yolo_conf > ENSEMBLE_AGREEMENT_THRESHOLD
                and dn_conf > ENSEMBLE_AGREEMENT_THRESHOLD):
            return BiRads.BIRADS_5

        # AGREEMENT: Both say Benign with high confidence
        if (not yolo_is_malignant and not dn_is_malignant
                and yolo_conf > ENSEMBLE_AGREEMENT_THRESHOLD
                and dn_conf > ENSEMBLE_AGREEMENT_THRESHOLD):
            return BiRads.BIRADS_2

        # DISAGREEMENT: Safety First
        if yolo_is_malignant != dn_is_malignant:
            return BiRads.BIRADS_3

        # Agreement with moderate confidence
        if ensemble_class == Classification.MALIGNANT:
            if ensemble_conf >= 0.75:
                return BiRads.BIRADS_5
            elif ensemble_conf >= 0.50:
                return BiRads.BIRADS_4
            else:
                return BiRads.BIRADS_3
        elif ensemble_class == Classification.BENIGN:
            if ensemble_conf >= 0.80:
                return BiRads.BIRADS_2
            elif ensemble_conf >= 0.50:
                return BiRads.BIRADS_3
            else:
                return BiRads.BIRADS_0
    else:
        # Single-model fallback (YOLO only)
        if yolo_is_malignant:
            if yolo_conf >= 0.75:
                return BiRads.BIRADS_4  # Lower than ensemble agreement
            elif yolo_conf >= 0.50:
                return BiRads.BIRADS_3
            else:
                return BiRads.BIRADS_0
        else:
            if yolo_conf >= 0.80:
                return BiRads.BIRADS_2
            elif yolo_conf >= 0.50:
                return BiRads.BIRADS_3
            else:
                return BiRads.BIRADS_0

    return BiRads.BIRADS_0


def compute_triage(birads: BiRads, confidence: float) -> TriagePriority:
    """
    Assign triage priority from BI-RADS category and confidence.
    """
    score = birads.score

    if score == 5:
        return TriagePriority.CRITICAL
    elif score == 4:
        return TriagePriority.HIGH if confidence >= 0.70 else TriagePriority.MODERATE
    elif score == 3:
        return TriagePriority.MODERATE
    elif score == 2:
        return TriagePriority.LOW
    else:
        return TriagePriority.ROUTINE


def compute_triage_priority_score(birads: BiRads, confidence: float) -> int:
    """
    Compute a numeric 1–10 triage priority score for the visual urgency map.

    10 = Most urgent (confirmed malignant, both models agree)
     1 = Routine / benign
    """
    score = birads.score

    if score == 5:
        return 10 if confidence >= 0.85 else 9
    elif score == 4:
        if confidence >= 0.80:
            return 8
        elif confidence >= 0.60:
            return 7
        else:
            return 6
    elif score == 3:
        return 5 if confidence >= 0.60 else 4
    elif score == 2:
        return 2
    elif score == 1:
        return 1
    else:
        return 3  # BI-RADS 0 = incomplete


def estimate_mass_size_mm(bbox: BoundingBox, image_width: int,
                          sensor_width_mm: float = 240.0) -> float:
    """Estimate physical mass size in mm using pixel-to-mm ratio."""
    pixel_per_mm = image_width / sensor_width_mm
    avg_pixel_size = (bbox.width + bbox.height) / 2
    return round(avg_pixel_size / pixel_per_mm, 1)


# ── Explainability Generator ───────────────────────────────────────────

def generate_explainability(result: "EnsembleResult") -> str:
    """
    Generate a human-readable text explanation of why the AI flagged this area.
    """
    parts = []

    # Classification summary
    cls_name = result.ensemble_classification.value
    conf = result.ensemble_confidence
    parts.append(
        f"The ensemble classified this region as '{cls_name}' "
        f"with {conf:.0%} combined confidence."
    )

    # Model agreement
    yolo_cls = result.yolo_result.class_name
    yolo_conf = result.yolo_result.confidence
    parts.append(
        f"YOLO11 detected '{yolo_cls}' (conf: {yolo_conf:.0%})."
    )

    if result.densenet_result:
        dn = result.densenet_result
        parts.append(
            f"DenseNet201 classified as '{dn.classification.value}' "
            f"(Benign: {dn.benign_prob:.0%}, Malignant: {dn.malignant_prob:.0%})."
        )

        # Agreement check
        yolo_mal = "malignant" in yolo_cls.lower()
        dn_mal = dn.classification == Classification.MALIGNANT
        if yolo_mal == dn_mal:
            parts.append("Both models AGREE on the classification.")
        else:
            parts.append(
                "Models DISAGREE — defaulting to Safety-First protocol "
                "(BI-RADS 3 for radiologist review)."
            )

    # Feature risks
    if result.detected_features:
        feat_parts = []
        for cat, val in result.detected_features.items():
            risk = FEATURE_RISK_MAP.get(cat, {}).get(val, None)
            if risk is not None:
                feat_parts.append(f"{cat}: {val} (risk: {risk:.1f}/10)")
        if feat_parts:
            parts.append("Detected features: " + ", ".join(feat_parts) + ".")

    # Size info
    if result.mass_size_mm:
        parts.append(f"Estimated mass size: {result.mass_size_mm} mm.")

    # BI-RADS / Triage
    parts.append(f"Assessment: {result.birads}.")
    parts.append(
        f"Triage Priority: {result.triage.value} "
        f"(Score: {result.triage_priority_score}/10)."
    )

    return " ".join(parts)

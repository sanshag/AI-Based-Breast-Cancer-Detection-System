# Multi-Stage Ensemble Breast Cancer Diagnostic System

> **YOLO11 + DenseNet201 Two-Stage Ensemble Pipeline for 2D Mammography AI**  
> Transforms raw mammograms into comparison-box annotated images + ensemble clinical PDF reports with BI-RADS scoring and triage priority mapping.

---

## Architecture Overview

```
input_mammograms/            ← Drop your mammogram images here
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │            STAGE 1: YOLO11 — Whole-Mammogram Detection          │
  │  • Ultralytics YOLO11 scans full image                          │
  │  • Outputs bounding boxes + class (Benign/Malignant) + conf     │
  └──────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │            AUTOMATED ROI CROPPING                                │
  │  • Extract high-resolution sub-images from each detection       │
  │  • Add contextual padding (15%) around each mass                │
  └──────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │            STAGE 2: DenseNet201 — Crop Classification           │
  │  • Pre-trained torchvision DenseNet201                          │
  │  • Binary classification: Benign vs Malignant                   │
  │  • Test-Time Augmentation (6 transforms) → +2-3% accuracy      │
  └──────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │            WEIGHTED VOTING ENSEMBLE                              │
  │  • YOLO weight: 0.40  |  DenseNet weight: 0.60                  │
  │  • Both Malignant >70% → BI-RADS 5, Priority 10                │
  │  • Disagreement → Safety-First → BI-RADS 3 for human review    │
  └──────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
        output_results/
        ├── annotated_images/     ← Comparison boxes (YOLO + Ensemble)
        ├── roi_crops/            ← Extracted mass sub-images
        ├── pdf_reports/          ← Ensemble clinical PDF reports
        ├── yolo_labels/          ← Distilled YOLO TXT labels
        ├── audit_logs/           ← Per-image model score logs + JSON
        └── summary.csv           ← Complete findings spreadsheet
```

## Module Structure

| Module | Agent | Role |
|--------|-------|------|
| `config.py` | Agent 1 (Clinical Architect) | Ensemble schema, weighted voting, feature-to-risk mapping, BI-RADS |
| `inference.py` | Agent 2 (AI & Inference) | YOLO11 detection, ROI cropping, DenseNet201 + TTA, label distillation |
| `reporting.py` | Agent 3 (Visualization) | Comparison-box annotation, ensemble PDF reports, triage priority map |
| `main.py` | Agent 4 (Integration) | Pipeline orchestration, summary CSV, audit trail, CLI |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Mammogram Images

Place your mammogram images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`) into the `input_mammograms/` folder.

### 3. Run Ensemble Analysis

```bash
python main.py
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Input folder with mammogram images | `input_mammograms/` |
| `-o`, `--output` | Output folder for results | `output_results/` |
| `--yolo-model` | Path to YOLO11 model weights | `yolo11n.pt` |
| `--densenet-model` | Path to custom DenseNet201 weights | ImageNet pretrained |
| `--no-tta` | Disable Test-Time Augmentation | TTA ON |
| `--distill-labels` | Save YOLO-format TXT labels for training | OFF |
| `--no-pdf` | Skip PDF report generation | OFF |
| `--no-annotate` | Skip image annotation | OFF |

### Examples

```bash
# Custom YOLO model with label distillation
python main.py --yolo-model best.pt --distill-labels

# Disable TTA for faster processing
python main.py --no-tta

# Custom folders
python main.py -i scans/ -o results/
```

## Output Structure

```
output_results/
├── annotated_images/
│   └── image1_annotated.jpg      # Comparison boxes (YOLO inner + Ensemble outer)
├── roi_crops/
│   └── image1_crop_1.jpg         # High-res cropped mass regions
├── pdf_reports/
│   └── image1_report.pdf         # Full ensemble clinical report
├── yolo_labels/                  # (if --distill-labels)
│   └── image1.txt                # YOLO-format annotations
├── audit_logs/
│   ├── audit_20260323_191500.log # Detailed per-image logs
│   └── audit_20260323_191500.json# Machine-readable audit data
└── summary.csv                   # All findings with dual-model scores
```

## Ensemble Logic

### Weighted Voting System

| Scenario | YOLO11 | DenseNet201 | Result |
|----------|--------|-------------|--------|
| Both Malignant >70% | Malignant (high) | Malignant (high) | **BI-RADS 5, Priority 10** |
| Both Benign >70% | Benign (high) | Benign (high) | **BI-RADS 2, Priority 2** |
| Disagreement | Malignant | Benign | **BI-RADS 3 (Safety First)** |
| Disagreement | Benign | Malignant | **BI-RADS 3 (Safety First)** |

### Clinical Feature Risk Mapping

| Category | Feature | Risk Score (0-10) |
|----------|---------|-------------------|
| Shape | Round | 1.0 |
| Shape | Irregular | 8.0 |
| Margin | Circumscribed | 1.0 |
| Margin | Spiculated | 9.5 |
| Density | Low | 1.5 |
| Density | High | 6.5 |

### Triage Priority Scale (1-10)

| Score | Level | BI-RADS | Action |
|-------|-------|---------|--------|
| 10 | 🔴 CRITICAL | 5 (>85%) | Immediate biopsy |
| 8-9 | 🟠 HIGH | 4-5 | Urgent workup |
| 5-7 | 🟡 MODERATE | 3-4 | Follow-up |
| 2-4 | 🟢 LOW | 2-3 | Routine |
| 1 | 🔵 ROUTINE | 0-1 | Standard screening |

## Test-Time Augmentation (TTA)

TTA boosts accuracy by 2-3% by analyzing each crop at multiple orientations:
- Original
- Horizontal flip
- Vertical flip
- 90° rotation
- 180° rotation
- 270° rotation

All predictions are averaged for the final classification.

## PDF Report Contents

Each ensemble report includes:
- **Exam Information** — models used, weights, TTA status, processing time
- **Ensemble Clinical Summary** — combined confidence, highest BI-RADS, triage
- **Triage Priority Map** — visual 1-10 scale with color gradient
- **Comparison-Box Annotated Mammogram** — YOLO inner + Ensemble outer boxes
- **Detailed Ensemble Findings Table** — per-detection YOLO + DenseNet scores
- **AI Explainability** — text explaining why each area was flagged
- **Clinical Recommendations** — BI-RADS-based action items
- **Disclaimer** — AI/research-use notice

## Requirements

- Python 3.9+
- CUDA GPU recommended (works on CPU but slower)
- No API keys required — all inference is local

## Environment Variables (Optional)

Create a `.env` file to customize:
```env
YOLO_MODEL_PATH=yolo11n.pt
YOLO_CONF_THRESHOLD=0.25
YOLO_IOU_THRESHOLD=0.45
DENSENET_INPUT_SIZE=224
YOLO_WEIGHT=0.40
DENSENET_WEIGHT=0.60
TTA_ENABLED=true
```

## Disclaimer

This system is for **research and educational purposes only**. It is NOT a medical device and should NOT be used for clinical diagnosis. All findings must be reviewed by a qualified radiologist.

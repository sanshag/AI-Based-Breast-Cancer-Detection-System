"""
==========================================================================
 Agent 4 — PIPELINE INTEGRATION & ACCURACY LEAD
==========================================================================
 Combines all agents into a single main.py that:
   1. Processes a folder of mammogram images end-to-end
   2. Runs YOLO11 → crop → DenseNet201 → Ensemble Voting
   3. Generates comparison-box annotated images
   4. Creates clinical PDF reports with ensemble confidence
   5. Exports summary.csv with all findings
   6. Maintains an audit trail (per-image YOLO + DenseNet scores)
   7. Supports Test-Time Augmentation toggle
   8. Supports YOLO label distillation mode
==========================================================================
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from tqdm import tqdm

from config import (
    INPUT_FOLDER, OUTPUT_FOLDER,
    ANNOTATED_SUBFOLDER, REPORTS_SUBFOLDER, AUDIT_SUBFOLDER,
    SUMMARY_CSV_NAME,
    ImageResult, Classification,
    YOLO_MODEL_PATH, DENSENET_MODEL_PATH,
    TTA_ENABLED, YOLO_WEIGHT, DENSENET_WEIGHT,
    YOLO_SPECIALIST_FILENAME, DENSENET_SPECIALIST_FILENAME,
    YOLO_GENERAL_FILENAME,
)
from inference import EnsembleInferenceEngine, collect_images
from reporting import annotate_image, generate_report


# ── Audit Logger Setup ─────────────────────────────────────────────────

def setup_audit_logger(output_root: str) -> logging.Logger:
    """
    Create a dedicated audit logger that writes per-image model scores
    to a timestamped log file for clinical auditing.
    """
    audit_dir = os.path.join(output_root, AUDIT_SUBFOLDER)
    os.makedirs(audit_dir, exist_ok=True)

    log_file = os.path.join(
        audit_dir,
        f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("ensemble_audit")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-runs
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Also log to console at INFO level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def log_audit_entry(
    logger: logging.Logger,
    result: ImageResult,
):
    """
    Log individual scores from both YOLO11 and DenseNet201 for every image.
    This creates the audit trail required for clinical auditing.
    """
    logger.info(f"{'='*70}")
    logger.info(f"IMAGE: {result.image_filename}")
    logger.info(f"  Dimensions: {result.image_width}x{result.image_height}")
    logger.info(f"  Processing Time: {result.processing_time_sec:.2f}s")
    logger.info(f"  TTA Enabled: {result.tta_enabled}")
    logger.info(f"  Stage 1 (YOLO11) Detections: {len(result.stage1_detections)}")

    for i, yolo_det in enumerate(result.stage1_detections, 1):
        logger.debug(
            f"    YOLO #{i}: class='{yolo_det.class_name}' "
            f"conf={yolo_det.confidence:.4f} "
            f"bbox=({yolo_det.bounding_box.x_center:.1f}, "
            f"{yolo_det.bounding_box.y_center:.1f}, "
            f"{yolo_det.bounding_box.width:.1f}, "
            f"{yolo_det.bounding_box.height:.1f}) "
            f"crop={yolo_det.crop_path or 'N/A'}"
        )

    logger.info(f"  Ensemble Results: {len(result.ensemble_results)}")

    for ens in result.ensemble_results:
        logger.info(
            f"    ENS #{ens.detection_id}: "
            f"ensemble={ens.ensemble_classification.value} "
            f"conf={ens.ensemble_confidence:.4f} "
            f"BI-RADS={ens.birads.score} "
            f"priority={ens.triage_priority_score}/10 "
            f"triage={ens.triage.value}"
        )
        logger.debug(
            f"      YOLO weighted: {ens.yolo_weighted_score:.4f} | "
            f"DenseNet weighted: {ens.densenet_weighted_score:.4f}"
        )
        if ens.densenet_result:
            dn = ens.densenet_result
            logger.debug(
                f"      DenseNet201: class={dn.classification.value} "
                f"conf={dn.confidence:.4f} "
                f"P(benign)={dn.benign_prob:.4f} "
                f"P(malignant)={dn.malignant_prob:.4f}"
            )
            if dn.tta_scores:
                logger.debug(f"      TTA Scores: {json.dumps(dn.tta_scores, indent=8)}")
        if ens.detected_features:
            logger.debug(f"      Features: {json.dumps(ens.detected_features)}")
        logger.debug(f"      Risk Score: {ens.feature_risk_score:.2f}/10")
        logger.debug(f"      Explainability: {ens.explainability_text[:200]}...")


# ── Directory Setup ────────────────────────────────────────────────────

def setup_directories(output_root: str) -> dict:
    """Create the output directory structure."""
    dirs = {
        "root": output_root,
        "annotated": os.path.join(output_root, ANNOTATED_SUBFOLDER),
        "reports": os.path.join(output_root, REPORTS_SUBFOLDER),
        "audit": os.path.join(output_root, AUDIT_SUBFOLDER),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ── Summary CSV Export ─────────────────────────────────────────────────

def export_summary_csv(results: List[ImageResult], output_path: str) -> str:
    """
    Export a comprehensive summary CSV with ensemble results.

    Columns include dual-model scores for full transparency.
    """
    fieldnames = [
        "image_file", "detection_id",
        "ensemble_classification", "ensemble_confidence",
        "yolo_class", "yolo_confidence", "yolo_weighted_score",
        "densenet_class", "densenet_confidence", "densenet_weighted_score",
        "birads_score", "birads_description",
        "triage_priority", "triage_priority_score",
        "feature_risk_score", "est_size_mm",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "tta_enabled", "processing_time_sec",
        "annotated_image", "pdf_report",
    ]

    rows = []
    for result in results:
        if result.ensemble_results:
            for ens in result.ensemble_results:
                dn_cls = (ens.densenet_result.classification.value
                          if ens.densenet_result else "N/A")
                dn_conf = (f"{ens.densenet_result.confidence:.4f}"
                           if ens.densenet_result else "")

                rows.append({
                    "image_file": result.image_filename,
                    "detection_id": ens.detection_id,
                    "ensemble_classification": ens.ensemble_classification.value,
                    "ensemble_confidence": f"{ens.ensemble_confidence:.4f}",
                    "yolo_class": ens.yolo_result.class_name,
                    "yolo_confidence": f"{ens.yolo_result.confidence:.4f}",
                    "yolo_weighted_score": f"{ens.yolo_weighted_score:.4f}",
                    "densenet_class": dn_cls,
                    "densenet_confidence": dn_conf,
                    "densenet_weighted_score": f"{ens.densenet_weighted_score:.4f}",
                    "birads_score": ens.birads.score,
                    "birads_description": ens.birads.description,
                    "triage_priority": ens.triage.value,
                    "triage_priority_score": ens.triage_priority_score,
                    "feature_risk_score": ens.feature_risk_score,
                    "est_size_mm": ens.mass_size_mm or "",
                    "bbox_x": f"{ens.yolo_result.bounding_box.x_center:.1f}",
                    "bbox_y": f"{ens.yolo_result.bounding_box.y_center:.1f}",
                    "bbox_w": f"{ens.yolo_result.bounding_box.width:.1f}",
                    "bbox_h": f"{ens.yolo_result.bounding_box.height:.1f}",
                    "tta_enabled": result.tta_enabled,
                    "processing_time_sec": result.processing_time_sec,
                    "annotated_image": result.annotated_path or "",
                    "pdf_report": result.report_path or "",
                })
        else:
            rows.append({
                "image_file": result.image_filename,
                "detection_id": 0,
                "ensemble_classification": "No Detection",
                "ensemble_confidence": "",
                "yolo_class": "", "yolo_confidence": "",
                "yolo_weighted_score": "",
                "densenet_class": "", "densenet_confidence": "",
                "densenet_weighted_score": "",
                "birads_score": 1,
                "birads_description": "Negative",
                "triage_priority": "ROUTINE",
                "triage_priority_score": 1,
                "feature_risk_score": "",
                "est_size_mm": "",
                "bbox_x": "", "bbox_y": "", "bbox_w": "", "bbox_h": "",
                "tta_enabled": result.tta_enabled,
                "processing_time_sec": result.processing_time_sec,
                "annotated_image": "",
                "pdf_report": result.report_path or "",
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(output_path)


# ── Audit JSON Export ──────────────────────────────────────────────────

def export_audit_json(results: List[ImageResult], output_path: str) -> str:
    """
    Export a detailed JSON audit file with all model scores for every image.
    """
    audit_data = []

    for result in results:
        image_entry = {
            "image_filename": result.image_filename,
            "image_dimensions": f"{result.image_width}x{result.image_height}",
            "processing_time_sec": result.processing_time_sec,
            "tta_enabled": result.tta_enabled,
            "total_yolo_detections": len(result.stage1_detections),
            "total_ensemble_results": len(result.ensemble_results),
            "yolo_detections": [],
            "ensemble_results": [],
        }

        for yolo_det in result.stage1_detections:
            image_entry["yolo_detections"].append({
                "class_id": yolo_det.class_id,
                "class_name": yolo_det.class_name,
                "confidence": yolo_det.confidence,
                "bbox": {
                    "x_center": yolo_det.bounding_box.x_center,
                    "y_center": yolo_det.bounding_box.y_center,
                    "width": yolo_det.bounding_box.width,
                    "height": yolo_det.bounding_box.height,
                },
                "crop_path": yolo_det.crop_path,
            })

        for ens in result.ensemble_results:
            ens_entry = {
                "detection_id": ens.detection_id,
                "ensemble_classification": ens.ensemble_classification.value,
                "ensemble_confidence": ens.ensemble_confidence,
                "yolo_weighted_score": ens.yolo_weighted_score,
                "densenet_weighted_score": ens.densenet_weighted_score,
                "birads_score": ens.birads.score,
                "birads_description": ens.birads.description,
                "triage_priority": ens.triage.value,
                "triage_priority_score": ens.triage_priority_score,
                "feature_risk_score": ens.feature_risk_score,
                "detected_features": ens.detected_features,
                "mass_size_mm": ens.mass_size_mm,
                "explainability": ens.explainability_text,
            }

            if ens.densenet_result:
                dn = ens.densenet_result
                ens_entry["densenet_detail"] = {
                    "classification": dn.classification.value,
                    "confidence": dn.confidence,
                    "benign_prob": dn.benign_prob,
                    "malignant_prob": dn.malignant_prob,
                    "tta_scores": dn.tta_scores,
                }

            image_entry["ensemble_results"].append(ens_entry)

        audit_data.append(image_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, default=str)

    return os.path.abspath(output_path)


# ── Banner & Summary ───────────────────────────────────────────────────

def _detect_specialist_status() -> dict:
    """Detect which model weights are available and being used."""
    yolo_specialist = os.path.exists(YOLO_SPECIALIST_FILENAME)
    dn_specialist = os.path.exists(DENSENET_SPECIALIST_FILENAME)
    yolo_using = YOLO_MODEL_PATH
    dn_using = DENSENET_MODEL_PATH or "ImageNet pretrained"

    return {
        "yolo_specialist_available": yolo_specialist,
        "densenet_specialist_available": dn_specialist,
        "yolo_active": yolo_using,
        "densenet_active": dn_using,
        "is_specialist_mode": yolo_specialist or dn_specialist,
    }


def print_banner():
    """Print a stylized startup banner with specialist detection."""
    status = _detect_specialist_status()

    yolo_tag = "SPECIALIST ✦" if status["yolo_specialist_available"] else "GENERAL"
    dn_tag = "SPECIALIST ✦" if status["densenet_specialist_available"] else "GENERAL"
    mode_str = "SPECIALIST MODE" if status["is_specialist_mode"] else "GENERAL MODE"

    banner = rf"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   MULTI-STAGE ENSEMBLE BREAST CANCER DIAGNOSTIC SYSTEM  v2.1   ║
    ║   YOLO11 Detection + DenseNet201 Classification                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Agent 1: Clinical Architect — Ensemble Schema & Weighted Vote ║
    ║  Agent 2: AI Engineer — YOLO11 + DenseNet201 + TTA Pipeline    ║
    ║  Agent 3: Reporting — Comparison Boxes + Clinical PDF          ║
    ║  Agent 4: Integration — Audit Trail & Accuracy Optimization    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Ensemble: YOLO (w={YOLO_WEIGHT}) + DenseNet (w={DENSENET_WEIGHT})                   ║
    ║  Safety-First: Disagreement → BI-RADS 3 for human review       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  🔬 {mode_str:<54} ║
    ║     YOLO11:     [{yolo_tag:<14}] {status['yolo_active']:<30} ║
    ║     DenseNet201: [{dn_tag:<14}] {status['densenet_active']:<30} ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_result_summary(results: List[ImageResult]):
    """Print a formatted summary of all processing results."""
    total_images = len(results)
    total_detections = sum(r.total_detections for r in results)
    total_malignant = sum(
        sum(1 for e in r.ensemble_results
            if e.ensemble_classification == Classification.MALIGNANT)
        for r in results
    )
    total_benign = sum(
        sum(1 for e in r.ensemble_results
            if e.ensemble_classification == Classification.BENIGN)
        for r in results
    )
    total_time = sum(r.processing_time_sec for r in results)

    critical_count = sum(
        1 for r in results
        if r.highest_triage and r.highest_triage.value == "CRITICAL"
    )
    high_count = sum(
        1 for r in results
        if r.highest_triage and r.highest_triage.value == "HIGH"
    )

    print("\n" + "=" * 70)
    print("  ENSEMBLE PROCESSING COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  📁 Images Processed:     {total_images}")
    print(f"  🔍 Total Detections:     {total_detections}")
    print(f"     ├─ Malignant:         {total_malignant}")
    print(f"     └─ Benign:            {total_benign}")
    print(f"  🧠 Pipeline:             YOLO11 → DenseNet201 → Ensemble")
    print(f"  ⏱  Total Time:           {total_time:.1f}s")
    print(f"     └─ Avg per image:     {total_time / max(total_images, 1):.1f}s")
    print(f"  🚨 Critical Priority:    {critical_count}")
    print(f"  ⚠  High Priority:        {high_count}")
    print("=" * 70)

    if results:
        print(f"\n  {'Image':<30} {'Det':>4} {'Mal':>4} {'BI-RADS':>8} {'Priority':>10} {'Triage':>10} {'Time':>6}")
        print("  " + "─" * 76)
        for r in results:
            n_mal = sum(1 for e in r.ensemble_results
                       if e.ensemble_classification == Classification.MALIGNANT)
            birads = f"  {r.highest_birads.score}" if r.highest_birads else "  —"
            triage = r.highest_triage.value if r.highest_triage else "—"
            priority = f"{r.max_triage_score}/10"
            name = (r.image_filename[:28] + ".."
                    if len(r.image_filename) > 30
                    else r.image_filename)
            print(
                f"  {name:<30} {r.total_detections:>4} {n_mal:>4} "
                f"{birads:>8} {priority:>10} {triage:>10} {r.processing_time_sec:>5.1f}s"
            )
    print()


# ── Main Entry Point ───────────────────────────────────────────────────

def main():
    """Main entry point — orchestrates the full ensemble detection pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-Stage Ensemble Breast Cancer Diagnostic System — "
            "YOLO11 + DenseNet201 batch mammogram analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Default settings with TTA
  python main.py -i scans/ -o results/        # Custom folders
  python main.py --no-tta                     # Disable Test-Time Augmentation
  python main.py --distill-labels             # Save YOLO TXT labels
  python main.py --yolo-model best.pt         # Custom YOLO weights
        """,
    )
    parser.add_argument(
        "-i", "--input", default=INPUT_FOLDER,
        help=f"Input folder with mammogram images (default: {INPUT_FOLDER})",
    )
    parser.add_argument(
        "-o", "--output", default=OUTPUT_FOLDER,
        help=f"Output folder for results (default: {OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--yolo-model", default=YOLO_MODEL_PATH,
        help=f"Path to YOLO11 model weights (default: {YOLO_MODEL_PATH})",
    )
    parser.add_argument(
        "--densenet-model", default=DENSENET_MODEL_PATH,
        help="Path to custom DenseNet201 weights (default: ImageNet pretrained)",
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable Test-Time Augmentation",
    )
    parser.add_argument(
        "--distill-labels", action="store_true",
        help="Save YOLO-format TXT labels for model training",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF report generation",
    )
    parser.add_argument(
        "--no-annotate", action="store_true",
        help="Skip image annotation",
    )
    args = parser.parse_args()

    print_banner()

    # ── Step 1: Validate Input ─────────────────────────────────────────
    input_folder = os.path.abspath(args.input)
    if not os.path.isdir(input_folder):
        print(f"  ❌ Input folder not found: {input_folder}")
        print(f"     Create it and add mammogram images, then re-run.")
        os.makedirs(input_folder, exist_ok=True)
        print(f"     📂 Created empty folder: {input_folder}")
        sys.exit(1)

    image_paths = collect_images(input_folder)
    if not image_paths:
        print(f"  ❌ No supported images found in: {input_folder}")
        print(f"     Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF")
        sys.exit(1)

    print(f"  📂 Input folder:   {input_folder}")
    print(f"  🖼  Images found:   {len(image_paths)}")

    # ── Step 2: Setup Output & Audit ───────────────────────────────────
    output_root = os.path.abspath(args.output)
    dirs = setup_directories(output_root)
    print(f"  📂 Output folder:  {output_root}")

    audit_logger = setup_audit_logger(output_root)
    audit_logger.info("=" * 70)
    audit_logger.info("MULTI-STAGE ENSEMBLE SESSION STARTED")
    audit_logger.info(f"  Input: {input_folder}")
    audit_logger.info(f"  Output: {output_root}")
    audit_logger.info(f"  Images: {len(image_paths)}")
    audit_logger.info(f"  YOLO Model: {args.yolo_model}")
    audit_logger.info(f"  DenseNet Model: {args.densenet_model or 'ImageNet pretrained'}")
    audit_logger.info(f"  TTA: {'OFF' if args.no_tta else 'ON'}")
    audit_logger.info(f"  Distill Labels: {args.distill_labels}")

    # Log specialist detection
    specialist_status = _detect_specialist_status()
    if specialist_status["is_specialist_mode"]:
        audit_logger.info("  ✦ SPECIALIST MODE ACTIVE")
        audit_logger.info(f"    YOLO specialist:    {'LOADED' if specialist_status['yolo_specialist_available'] else 'NOT FOUND'}")
        audit_logger.info(f"    DenseNet specialist: {'LOADED' if specialist_status['densenet_specialist_available'] else 'NOT FOUND'}")
    else:
        audit_logger.info("  ⓘ GENERAL MODE (no specialist weights found)")
        audit_logger.info(f"    Run 'python train_specialist.py --all' to create specialist weights")
    audit_logger.info("=" * 70)

    # ── Step 3: Initialize Ensemble Engine ─────────────────────────────
    use_tta = not args.no_tta and TTA_ENABLED
    try:
        engine = EnsembleInferenceEngine(
            yolo_path=args.yolo_model,
            densenet_path=args.densenet_model,
            use_tta=use_tta,
            distill_labels=args.distill_labels,
        )
    except Exception as e:
        print(f"\n  ❌ Engine Initialization Error:\n     {e}")
        audit_logger.error(f"Engine initialization failed: {e}")
        sys.exit(1)

    # ── Step 4: Process Each Image ─────────────────────────────────────
    print(f"\n  🚀 Starting ensemble analysis (TTA: {'ON' if use_tta else 'OFF'})...\n")
    all_results: List[ImageResult] = []
    start_total = time.time()

    for img_path in tqdm(
        image_paths, desc="  Processing", unit="image",
        bar_format="  {l_bar}{bar:30}{r_bar}"
    ):
        filename = Path(img_path).stem
        tqdm.write(f"  ── Analyzing: {Path(img_path).name}")

        # 4a. Run two-stage ensemble inference
        result = engine.run_inference(img_path, output_root)

        # 4b. Annotate image with comparison boxes
        if not args.no_annotate and result.ensemble_results:
            ann_path = os.path.join(
                dirs["annotated"],
                f"{filename}_annotated.jpg"
            )
            try:
                result.annotated_path = annotate_image(
                    img_path, result.ensemble_results, ann_path
                )
                tqdm.write(f"     ✅ Annotated → {Path(ann_path).name}")
            except Exception as e:
                tqdm.write(f"     ⚠ Annotation failed: {e}")

        # 4c. Generate clinical PDF report
        if not args.no_pdf:
            pdf_path = os.path.join(
                dirs["reports"],
                f"{filename}_report.pdf"
            )
            try:
                result.report_path = generate_report(
                    result, pdf_path,
                    annotated_image_path=result.annotated_path,
                )
                tqdm.write(f"     ✅ Report   → {Path(pdf_path).name}")
            except Exception as e:
                tqdm.write(f"     ⚠ Report failed: {e}")

        # 4d. Log audit entry
        log_audit_entry(audit_logger, result)

        all_results.append(result)

    # ── Step 5: Export Summary CSV ─────────────────────────────────────
    csv_path = os.path.join(output_root, SUMMARY_CSV_NAME)
    export_summary_csv(all_results, csv_path)
    print(f"\n  📊 Summary CSV  → {csv_path}")

    # ── Step 6: Export Audit JSON ──────────────────────────────────────
    audit_json_path = os.path.join(
        dirs["audit"],
        f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    export_audit_json(all_results, audit_json_path)
    print(f"  📋 Audit JSON   → {audit_json_path}")

    # ── Step 7: Print Summary ──────────────────────────────────────────
    total_elapsed = round(time.time() - start_total, 1)
    print_result_summary(all_results)

    print(f"  📂 All outputs saved to: {output_root}")
    print(f"     ├─ {ANNOTATED_SUBFOLDER}/      (comparison-box annotated images)")
    print(f"     ├─ {REPORTS_SUBFOLDER}/          (ensemble clinical PDF reports)")
    print(f"     ├─ {AUDIT_SUBFOLDER}/          (audit logs & JSON)")
    print(f"     └─ {SUMMARY_CSV_NAME}          (complete findings spreadsheet)")
    print(f"\n  ✅ Ensemble pipeline complete in {total_elapsed}s.\n")

    audit_logger.info(f"SESSION COMPLETE: {len(all_results)} images in {total_elapsed}s")


if __name__ == "__main__":
    main()

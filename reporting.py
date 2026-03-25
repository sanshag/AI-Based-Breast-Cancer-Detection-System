"""
==========================================================================
 Agent 3 — VISUALIZATION & RADIOLOGIST REPORTING
==========================================================================
 Two responsibilities:
   1. Advanced Annotation with 'Comparison Boxes':
      - YOLO detection in Green/Red (thin border)
      - Ensemble final decision as a thick highlighted border
   2. Professional Clinical PDF Reports with ReportLab:
      - Ensemble Confidence Score (combined probability)
      - Triage Priority Map (visual 1–10 scale)
      - Explainability text
      - Dual-model audit scores
==========================================================================
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF

from config import (
    Classification, TriagePriority, BiRads,
    EnsembleResult, ImageResult, StageOneResult,
    YOLO_BENIGN_COLOR, YOLO_MALIGNANT_COLOR,
    ENSEMBLE_BENIGN_COLOR, ENSEMBLE_MALIGNANT_COLOR, ENSEMBLE_UNCERTAIN_COLOR,
    YOLO_BOX_THICKNESS, ENSEMBLE_BOX_THICKNESS,
    FONT_SCALE, LABEL_BG_ALPHA,
    YOLO_WEIGHT, DENSENET_WEIGHT,
)


# ════════════════════════════════════════════════════════════════════════
#  PART 1: Advanced Comparison-Box Annotation with OpenCV
# ════════════════════════════════════════════════════════════════════════

def annotate_image(
    image_path: str,
    ensemble_results: List[EnsembleResult],
    output_path: str,
) -> str:
    """
    Draw 'Comparison Boxes' on a mammogram:
      - Inner box (thin): YOLO11 detection (Green = Benign, Red = Malignant)
      - Outer box (thick): Ensemble final decision (highlighted border)

    Also adds:
      - Detailed labels with dual confidence scores
      - Corner markers for premium look
      - Status bar with summary statistics
      - Model legend

    Returns:
        Absolute path of the saved annotated image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    overlay = img.copy()
    img_h, img_w = img.shape[:2]

    for ens in ensemble_results:
        yolo = ens.yolo_result
        bbox = yolo.bounding_box
        x_min, y_min, x_max, y_max = bbox.as_ints()

        # ── YOLO Detection Box (inner, thin) ──────────────────────────
        yolo_is_malignant = "malignant" in yolo.class_name.lower()
        yolo_color = YOLO_MALIGNANT_COLOR if yolo_is_malignant else YOLO_BENIGN_COLOR

        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max),
                       yolo_color, YOLO_BOX_THICKNESS)

        # ── Ensemble Decision Box (outer, thick highlight) ────────────
        pad = ENSEMBLE_BOX_THICKNESS + 3
        ex_min = max(0, x_min - pad)
        ey_min = max(0, y_min - pad)
        ex_max = min(img_w, x_max + pad)
        ey_max = min(img_h, y_max + pad)

        if ens.ensemble_classification == Classification.MALIGNANT:
            ens_color = ENSEMBLE_MALIGNANT_COLOR
        elif ens.ensemble_classification == Classification.BENIGN:
            ens_color = ENSEMBLE_BENIGN_COLOR
        else:
            ens_color = ENSEMBLE_UNCERTAIN_COLOR

        cv2.rectangle(overlay, (ex_min, ey_min), (ex_max, ey_max),
                       ens_color, ENSEMBLE_BOX_THICKNESS)

        # ── Multi-Line Label ──────────────────────────────────────────
        font = cv2.FONT_HERSHEY_SIMPLEX
        small_font_scale = FONT_SCALE * 0.85

        # Line 1: Ensemble decision
        line1 = (
            f"ENSEMBLE: {ens.ensemble_classification.value} "
            f"({ens.ensemble_confidence:.0%})"
        )
        # Line 2: YOLO score
        line2 = f"YOLO: {yolo.class_name} ({yolo.confidence:.0%})"
        # Line 3: DenseNet score (if available)
        if ens.densenet_result:
            dn = ens.densenet_result
            line3 = (
                f"DenseNet: {dn.classification.value} "
                f"({dn.confidence:.0%})"
            )
        else:
            line3 = "DenseNet: N/A"
        # Line 4: BI-RADS & Priority
        line4 = f"BI-RADS {ens.birads.score} | Priority {ens.triage_priority_score}/10"

        lines = [line1, line2, line3, line4]
        line_heights = []
        max_width = 0
        for line in lines:
            (tw, th), baseline = cv2.getTextSize(line, font, small_font_scale, 1)
            line_heights.append(th + baseline + 4)
            max_width = max(max_width, tw)

        total_label_h = sum(line_heights) + 10
        label_y_start = max(ey_min - total_label_h - 5, 5)

        # Label background
        cv2.rectangle(
            overlay,
            (ex_min, label_y_start),
            (ex_min + max_width + 14, label_y_start + total_label_h),
            (30, 30, 30), -1
        )

        # Draw each line
        y_cursor = label_y_start + line_heights[0]
        for i, line in enumerate(lines):
            # Color first line by classification
            if i == 0:
                text_color = (
                    (100, 100, 255) if ens.ensemble_classification == Classification.MALIGNANT
                    else (100, 255, 100)
                )
            elif i == 1:
                text_color = (
                    (80, 80, 255) if yolo_is_malignant
                    else (80, 200, 80)
                )
            elif i == 2:
                if ens.densenet_result:
                    text_color = (
                        (80, 80, 255) if ens.densenet_result.classification == Classification.MALIGNANT
                        else (80, 200, 80)
                    )
                else:
                    text_color = (160, 160, 160)
            else:
                text_color = (220, 220, 220)

            cv2.putText(overlay, line, (ex_min + 7, y_cursor),
                         font, small_font_scale, text_color, 1, cv2.LINE_AA)
            if i < len(lines) - 1:
                y_cursor += line_heights[i + 1]

        # ── Corner Markers ────────────────────────────────────────────
        corner_len = min(25, int(bbox.width * 0.12), int(bbox.height * 0.12))
        corner_thick = ENSEMBLE_BOX_THICKNESS + 1

        corners = [
            ((ex_min, ey_min), (ex_min + corner_len, ey_min), (ex_min, ey_min + corner_len)),
            ((ex_max, ey_min), (ex_max - corner_len, ey_min), (ex_max, ey_min + corner_len)),
            ((ex_min, ey_max), (ex_min + corner_len, ey_max), (ex_min, ey_max - corner_len)),
            ((ex_max, ey_max), (ex_max - corner_len, ey_max), (ex_max, ey_max - corner_len)),
        ]
        for origin, h_end, v_end in corners:
            cv2.line(overlay, origin, h_end, ens_color, corner_thick)
            cv2.line(overlay, origin, v_end, ens_color, corner_thick)

    # ── Blend Overlay ─────────────────────────────────────────────────
    result = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    # ── Status Bar ────────────────────────────────────────────────────
    bar_h = 55
    h, w = result.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 35)

    n_total = len(ensemble_results)
    n_mal = sum(1 for e in ensemble_results
                if e.ensemble_classification == Classification.MALIGNANT)
    n_ben = sum(1 for e in ensemble_results
                if e.ensemble_classification == Classification.BENIGN)
    max_priority = max((e.triage_priority_score for e in ensemble_results), default=0)

    status_l1 = f"ENSEMBLE RESULTS: {n_total} detections | Malignant: {n_mal} | Benign: {n_ben} | Max Priority: {max_priority}/10"
    status_l2 = f"Pipeline: YOLO11 (w={YOLO_WEIGHT}) + DenseNet201 (w={DENSENET_WEIGHT}) | Comparison Boxes Active"

    cv2.putText(bar, status_l1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(bar, status_l2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                 0.45, (150, 150, 160), 1, cv2.LINE_AA)

    # Legend colors
    cv2.rectangle(bar, (w - 340, 8), (w - 320, 22), YOLO_BENIGN_COLOR, -1)
    cv2.putText(bar, "YOLO Benign", (w - 315, 20), cv2.FONT_HERSHEY_SIMPLEX,
                 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.rectangle(bar, (w - 340, 28), (w - 320, 42), YOLO_MALIGNANT_COLOR, -1)
    cv2.putText(bar, "YOLO Malign", (w - 315, 40), cv2.FONT_HERSHEY_SIMPLEX,
                 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.rectangle(bar, (w - 180, 8), (w - 160, 22), ENSEMBLE_BENIGN_COLOR, -1)
    cv2.putText(bar, "Ens. Benign", (w - 155, 20), cv2.FONT_HERSHEY_SIMPLEX,
                 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.rectangle(bar, (w - 180, 28), (w - 160, 42), ENSEMBLE_MALIGNANT_COLOR, -1)
    cv2.putText(bar, "Ens. Malign", (w - 155, 40), cv2.FONT_HERSHEY_SIMPLEX,
                 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    result = np.vstack([result, bar])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    return os.path.abspath(output_path)


# ════════════════════════════════════════════════════════════════════════
#  PART 2: Professional Clinical PDF Report with ReportLab
# ════════════════════════════════════════════════════════════════════════

def _triage_color(triage: TriagePriority) -> colors.Color:
    """Map triage priority to ReportLab color."""
    return {
        TriagePriority.CRITICAL: colors.Color(0.85, 0.1, 0.1),
        TriagePriority.HIGH: colors.Color(0.9, 0.4, 0.0),
        TriagePriority.MODERATE: colors.Color(0.9, 0.7, 0.0),
        TriagePriority.LOW: colors.Color(0.2, 0.6, 0.2),
        TriagePriority.ROUTINE: colors.Color(0.3, 0.5, 0.7),
    }.get(triage, colors.grey)


def _birads_color(birads: BiRads) -> colors.Color:
    """Map BI-RADS score to color."""
    mapping = {
        0: colors.Color(0.5, 0.5, 0.5),
        1: colors.Color(0.2, 0.6, 0.2),
        2: colors.Color(0.2, 0.6, 0.2),
        3: colors.Color(0.9, 0.7, 0.0),
        4: colors.Color(0.9, 0.4, 0.0),
        5: colors.Color(0.85, 0.1, 0.1),
    }
    return mapping.get(birads.score, colors.grey)


def _create_triage_priority_map(priority_score: int) -> Drawing:
    """
    Create a visual Triage Priority Map (1–10 scale) as a ReportLab Drawing.
    Shows a horizontal bar with colored segments and a marker.
    """
    d = Drawing(450, 50)

    # Background bar segments (1–10)
    segment_w = 40
    segment_h = 22
    y_bar = 15

    gradient_colors = [
        colors.Color(0.2, 0.6, 0.8),   # 1 - Blue (routine)
        colors.Color(0.2, 0.7, 0.5),   # 2 - Teal
        colors.Color(0.3, 0.8, 0.3),   # 3 - Green
        colors.Color(0.5, 0.8, 0.2),   # 4 - Yellow-green
        colors.Color(0.7, 0.8, 0.1),   # 5 - Yellow
        colors.Color(0.9, 0.7, 0.0),   # 6 - Orange-yellow
        colors.Color(0.95, 0.5, 0.0),  # 7 - Orange
        colors.Color(0.95, 0.3, 0.0),  # 8 - Dark orange
        colors.Color(0.9, 0.15, 0.1),  # 9 - Red
        colors.Color(0.75, 0.0, 0.0),  # 10 - Dark red (critical)
    ]

    for i in range(10):
        x = 5 + i * segment_w
        rect = Rect(x, y_bar, segment_w - 2, segment_h,
                     fillColor=gradient_colors[i],
                     strokeColor=colors.Color(0.3, 0.3, 0.3),
                     strokeWidth=0.5)
        d.add(rect)

        # Number label
        num = String(x + segment_w / 2 - 3, y_bar + 6,
                      str(i + 1),
                      fontName="Helvetica",
                      fontSize=8,
                      fillColor=colors.white if i >= 5 else colors.Color(0.2, 0.2, 0.2))
        d.add(num)

    # Marker for current priority
    marker_x = 5 + (priority_score - 1) * segment_w + segment_w / 2
    marker = String(marker_x - 3, y_bar + segment_h + 5,
                     "▼",
                     fontName="Helvetica-Bold",
                     fontSize=14,
                     fillColor=colors.Color(0.1, 0.1, 0.1))
    d.add(marker)

    # Scale labels
    low_label = String(5, 3, "LOW RISK",
                        fontName="Helvetica", fontSize=7,
                        fillColor=colors.Color(0.4, 0.4, 0.4))
    d.add(low_label)
    high_label = String(350, 3, "CRITICAL RISK",
                         fontName="Helvetica", fontSize=7,
                         fillColor=colors.Color(0.4, 0.4, 0.4))
    d.add(high_label)

    return d


def generate_report(
    result: ImageResult,
    output_path: str,
    annotated_image_path: str = None,
) -> str:
    """
    Generate a professional clinical PDF report for one mammogram.

    The report includes:
      - Header with system branding
      - Exam metadata
      - ENSEMBLE Summary Dashboard:
          * Combined confidence score
          * Triage Priority Map (1–10 visual)
          * BI-RADS assessment
      - Annotated image (comparison boxes)
      - Detailed dual-model findings table
      - Explainability section
      - TTA audit info
      - Disclaimer
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── Custom Styles ──────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontSize=18, textColor=colors.Color(0.12, 0.16, 0.30),
        spaceAfter=2 * mm, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontSize=9.5, textColor=colors.Color(0.4, 0.4, 0.5),
        alignment=TA_CENTER, spaceAfter=5 * mm,
    )
    section_style = ParagraphStyle(
        "SectionHeader", parent=styles["Heading2"],
        fontSize=12.5, textColor=colors.Color(0.12, 0.16, 0.30),
        spaceBefore=7 * mm, spaceAfter=3 * mm,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "BodyText", parent=styles["Normal"],
        fontSize=9.5, leading=13,
        textColor=colors.Color(0.2, 0.2, 0.2),
    )
    small_style = ParagraphStyle(
        "SmallText", parent=styles["Normal"],
        fontSize=8, textColor=colors.Color(0.5, 0.5, 0.5),
        alignment=TA_CENTER,
    )
    explain_style = ParagraphStyle(
        "Explain", parent=styles["Normal"],
        fontSize=9, leading=12.5,
        textColor=colors.Color(0.25, 0.25, 0.30),
        leftIndent=8, rightIndent=8,
        borderColor=colors.Color(0.85, 0.85, 0.90),
        borderWidth=0.5, borderPadding=6,
        backColor=colors.Color(0.96, 0.96, 0.98),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=7.5, textColor=colors.Color(0.55, 0.55, 0.55),
        leading=10, spaceBefore=6 * mm,
        borderColor=colors.Color(0.8, 0.8, 0.8),
        borderWidth=0.5, borderPadding=6,
    )

    now = datetime.now()

    # ── Header ─────────────────────────────────────────────────────────
    elements.append(Paragraph(
        "MULTI-STAGE ENSEMBLE MAMMOGRAPHY REPORT", title_style
    ))
    elements.append(Paragraph(
        "YOLO11 Detection + DenseNet201 Classification  •  "
        "Weighted Voting Ensemble  •  "
        f"Generated: {now.strftime('%B %d, %Y at %H:%M')}",
        subtitle_style
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=colors.Color(0.12, 0.16, 0.30),
        spaceAfter=4 * mm,
    ))

    # ── Exam Information ───────────────────────────────────────────────
    elements.append(Paragraph("EXAM INFORMATION", section_style))

    tta_status = "Enabled (6 augmentations)" if result.tta_enabled else "Disabled"
    exam_data = [
        ["Image File:", result.image_filename],
        ["Dimensions:", f"{result.image_width} × {result.image_height} px"],
        ["Analysis Date:", now.strftime("%Y-%m-%d %H:%M:%S")],
        ["Processing Time:", f"{result.processing_time_sec:.2f} seconds"],
        ["Detection Model:", "YOLO11 (ultralytics)"],
        ["Classification Model:", "DenseNet201 (torchvision)"],
        ["Ensemble Weights:", f"YOLO={YOLO_WEIGHT} / DenseNet={DENSENET_WEIGHT}"],
        ["Test-Time Augmentation:", tta_status],
        ["Report ID:", f"ENS-{now.strftime('%Y%m%d%H%M%S')}-{abs(hash(result.image_filename)) % 10000:04d}"],
    ]

    exam_table = Table(exam_data, colWidths=[48 * mm, 120 * mm])
    exam_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.Color(0.3, 0.3, 0.4)),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.Color(0.15, 0.15, 0.15)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(exam_table)

    # ── Ensemble Summary Dashboard ─────────────────────────────────────
    elements.append(Paragraph("ENSEMBLE CLINICAL SUMMARY", section_style))

    highest_birads = result.highest_birads
    highest_triage = result.highest_triage
    max_priority_score = result.max_triage_score

    birads_text = str(highest_birads) if highest_birads else "No findings"
    triage_text = highest_triage.value if highest_triage else "N/A"
    triage_clr = _triage_color(highest_triage) if highest_triage else colors.grey
    birads_clr = _birads_color(highest_birads) if highest_birads else colors.grey

    # Compute overall ensemble confidence
    if result.ensemble_results:
        avg_conf = sum(e.ensemble_confidence for e in result.ensemble_results) / len(result.ensemble_results)
    else:
        avg_conf = 0.0

    summary_data = [
        ["Detections", "Ensemble Confidence", "Highest BI-RADS", "Triage Priority"],
        [
            str(result.total_detections),
            f"{avg_conf:.1%}",
            f"BI-RADS {highest_birads.score}" if highest_birads else "—",
            triage_text,
        ],
    ]

    summary_table = Table(summary_data, colWidths=[40 * mm, 45 * mm, 40 * mm, 40 * mm])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.12, 0.16, 0.30)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 14),
        ("TEXTCOLOR", (0, 1), (0, 1), colors.Color(0.12, 0.16, 0.30)),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.Color(0.12, 0.16, 0.30)),
        ("TEXTCOLOR", (2, 1), (2, 1), birads_clr),
        ("TEXTCOLOR", (3, 1), (3, 1), triage_clr),
        ("TOPPADDING", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.85)),
    ]))
    elements.append(summary_table)

    # BI-RADS description
    if highest_birads:
        elements.append(Spacer(1, 3 * mm))
        elements.append(Paragraph(
            f"<b>Assessment:</b> {highest_birads.description}", body_style
        ))

    # ── Triage Priority Map (Visual 1–10 Scale) ───────────────────────
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph("<b>Triage Priority Map:</b>", body_style))
    elements.append(Spacer(1, 2 * mm))

    priority_drawing = _create_triage_priority_map(max_priority_score)
    elements.append(priority_drawing)

    # ── Annotated Image ────────────────────────────────────────────────
    if annotated_image_path and os.path.exists(annotated_image_path):
        elements.append(Paragraph("ANNOTATED MAMMOGRAM (COMPARISON BOXES)", section_style))
        try:
            img_w = 155 * mm
            elements.append(RLImage(annotated_image_path, width=img_w,
                                     kind="proportional"))
            elements.append(Spacer(1, 2 * mm))
            elements.append(Paragraph(
                "Inner Box = YOLO11 Detection (Green/Red)  |  "
                "Outer Box = Ensemble Final Decision (Thick Border)",
                small_style
            ))
        except Exception:
            elements.append(Paragraph(
                "<i>Annotated image could not be embedded.</i>", body_style
            ))

    # ── Detailed Ensemble Findings Table ───────────────────────────────
    if result.ensemble_results:
        elements.append(Paragraph("DETAILED ENSEMBLE FINDINGS", section_style))

        header = [
            "#", "Ensemble\nClass", "Ensemble\nConf",
            "YOLO\nClass", "YOLO\nConf",
            "DenseNet\nClass", "DenseNet\nConf",
            "BI-RADS", "Priority\nScore",
        ]

        rows = [header]
        for ens in result.ensemble_results:
            dn_cls = ens.densenet_result.classification.value if ens.densenet_result else "N/A"
            dn_conf = f"{ens.densenet_result.confidence:.0%}" if ens.densenet_result else "—"

            rows.append([
                str(ens.detection_id),
                ens.ensemble_classification.value,
                f"{ens.ensemble_confidence:.0%}",
                ens.yolo_result.class_name,
                f"{ens.yolo_result.confidence:.0%}",
                dn_cls,
                dn_conf,
                str(ens.birads.score),
                f"{ens.triage_priority_score}/10",
            ])

        findings_table = Table(rows, colWidths=[
            8 * mm, 22 * mm, 18 * mm, 20 * mm, 16 * mm, 20 * mm, 18 * mm, 16 * mm, 18 * mm
        ])

        table_cmds = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.12, 0.16, 0.30)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.8, 0.8, 0.85)),
        ]

        for i, ens in enumerate(result.ensemble_results, start=1):
            if ens.ensemble_classification == Classification.MALIGNANT:
                table_cmds.append(
                    ("BACKGROUND", (0, i), (-1, i), colors.Color(1, 0.92, 0.92))
                )
                table_cmds.append(
                    ("TEXTCOLOR", (1, i), (1, i), colors.Color(0.75, 0.1, 0.1))
                )
            elif ens.ensemble_classification == Classification.BENIGN:
                table_cmds.append(
                    ("BACKGROUND", (0, i), (-1, i), colors.Color(0.92, 1, 0.92))
                )

        findings_table.setStyle(TableStyle(table_cmds))
        elements.append(findings_table)

    else:
        elements.append(Paragraph("FINDINGS", section_style))
        elements.append(Paragraph(
            "No masses detected in this mammogram by the YOLO11 detector. "
            "Classified as <b>BI-RADS 1 — Negative</b>. Routine screening recommended.",
            body_style
        ))

    # ── Explainability Section ─────────────────────────────────────────
    if result.ensemble_results:
        elements.append(Paragraph("AI EXPLAINABILITY", section_style))
        elements.append(Paragraph(
            "The following text explains why the AI ensemble flagged each region:",
            body_style
        ))
        elements.append(Spacer(1, 2 * mm))

        for ens in result.ensemble_results:
            elements.append(Paragraph(
                f"<b>Detection #{ens.detection_id}:</b> {ens.explainability_text}",
                explain_style
            ))
            elements.append(Spacer(1, 2 * mm))

    # ── Clinical Recommendations ───────────────────────────────────────
    elements.append(Paragraph("CLINICAL RECOMMENDATIONS", section_style))

    if highest_birads and highest_birads.score >= 5:
        rec_text = (
            "The ensemble has <b>high confidence in malignancy</b> with both YOLO11 and "
            "DenseNet201 in agreement. Immediate biopsy or diagnostic workup is "
            "<b>strongly recommended</b>. Refer to breast surgery or interventional radiology."
        )
    elif highest_birads and highest_birads.score == 4:
        rec_text = (
            "Findings are <b>suspicious</b>. Biopsy should be considered. "
            "The ensemble detected possible malignancy but models may have partial disagreement. "
            "Further diagnostic imaging and tissue sampling recommended."
        )
    elif highest_birads and highest_birads.score == 3:
        rec_text = (
            "Findings are <b>probably benign</b>, but the models showed some disagreement. "
            "Safety-First protocol applied. Short-interval follow-up (6 months) with "
            "diagnostic mammography or ultrasound is recommended."
        )
    elif highest_birads and highest_birads.score == 2:
        rec_text = (
            "Findings are <b>definitively benign</b> — both models agree with high confidence. "
            "Continue routine screening at standard intervals."
        )
    elif highest_birads and highest_birads.score == 0:
        rec_text = (
            "Assessment is <b>incomplete</b>. Additional imaging evaluation needed. "
            "Consider spot compression, magnification, or ultrasonography."
        )
    else:
        rec_text = "No significant findings. Continue routine screening."

    elements.append(Paragraph(rec_text, body_style))

    # ── Disclaimer ─────────────────────────────────────────────────────
    elements.append(Spacer(1, 4 * mm))
    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.Color(0.8, 0.8, 0.8),
        spaceAfter=3 * mm,
    ))
    elements.append(Paragraph(
        "<b>IMPORTANT DISCLAIMER:</b> This report was generated by a Multi-Stage "
        "Ensemble AI system (YOLO11 + DenseNet201) and is intended for research and "
        "educational purposes only. It does NOT constitute a medical diagnosis. "
        "All findings must be reviewed and confirmed by a qualified radiologist. "
        "Clinical decisions should never be based solely on AI-generated results. "
        "The system uses pre-trained models and may produce false positives or negatives. "
        "Test-Time Augmentation was "
        f"{'enabled' if result.tta_enabled else 'disabled'} for this analysis.",
        disclaimer_style
    ))

    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(
        f"Report generated on {now.strftime('%Y-%m-%d %H:%M:%S')} • "
        f"Multi-Stage Ensemble Mammography System v2.0",
        small_style
    ))

    # Build PDF
    doc.build(elements)
    return os.path.abspath(output_path)

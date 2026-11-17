"""
PDF Report Generator for AstraVisionX.

Produces a professional multi-page PDF report from analysis results using
ReportLab.  Each report includes the original image, Grad-CAM heatmap,
annotated detection image, data tables, NL summary, and a medical disclaimer
footer on every page.

Why ReportLab over WeasyPrint?  ReportLab is pure Python — no system
dependencies (cairo, pango) which are painful to install on Windows.
"""

import io
import base64
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)
from reportlab.lib.utils import ImageReader

from astravisionx.config import settings
from astravisionx.core.session import Session


# ---------------------------------------------------------------------------
# Colour palette (matches the web UI theme)
# ---------------------------------------------------------------------------
PURPLE = colors.HexColor("#8b5cf6")
CYAN = colors.HexColor("#06b6d4")
DARK_BG = colors.HexColor("#111118")
LIGHT_TEXT = colors.HexColor("#f0f0f5")
MUTED_TEXT = colors.HexColor("#9ca3af")
ACCENT_ROSE = colors.HexColor("#f43f5e")


def _b64_to_image_reader(b64_str: str) -> ImageReader:
    """Convert a base64-encoded JPEG string into a ReportLab ImageReader."""
    img_bytes = base64.b64decode(b64_str)
    return ImageReader(io.BytesIO(img_bytes))


def _build_styles():
    """Create the custom paragraph styles used throughout the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=26,
        textColor=PURPLE,
        spaceAfter=6,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=MUTED_TEXT,
        alignment=TA_CENTER,
        spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=15,
        textColor=PURPLE,
        spaceBefore=16,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "ReportBodyText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.black,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=ACCENT_ROSE,
        alignment=TA_CENTER,
        spaceBefore=8,
    ))
    return styles


def generate_pdf(session: Session) -> bytes:
    """
    Generate a full analysis PDF report and return the raw PDF bytes.

    The report contains:
      - Title page with branding and disclaimer
      - Image gallery (original + Grad-CAM + annotated)
      - YOLO detections table
      - ViT classifications table
      - NL summary
      - Disclaimer footer on every page
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=25 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    styles = _build_styles()
    story = []

    # --- Title ---
    story.append(Spacer(1, 30))
    story.append(Paragraph("AstraVisionX", styles["ReportTitle"]))
    story.append(Paragraph("AI-Powered Image Analysis Report", styles["ReportSubtitle"]))
    story.append(HRFlowable(width="60%", thickness=2, color=PURPLE, spaceAfter=12))

    # Meta info
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_text = (
        f"<b>File:</b> {session.filename or 'N/A'} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Session:</b> {session.session_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Generated:</b> {timestamp}"
    )
    story.append(Paragraph(meta_text, styles["ReportBodyText"]))
    story.append(Spacer(1, 8))

    # Disclaimer banner
    story.append(Paragraph(
        f"⚠ {settings.medical_disclaimer}",
        styles["Disclaimer"],
    ))
    story.append(Spacer(1, 16))

    # --- Original Image ---
    if session.original_image_b64:
        story.append(Paragraph("Original Image", styles["SectionHeading"]))
        try:
            img = RLImage(
                _b64_to_image_reader(session.original_image_b64),
                width=5.5 * inch,
                height=3.8 * inch,
                kind="proportional",
            )
            story.append(img)
        except Exception:
            story.append(Paragraph("<i>Image could not be rendered.</i>", styles["ReportBodyText"]))
        story.append(Spacer(1, 12))

    # --- Grad-CAM Heatmap ---
    if session.gradcam_image_b64:
        story.append(Paragraph("Grad-CAM Heatmap Overlay", styles["SectionHeading"]))
        try:
            img = RLImage(
                _b64_to_image_reader(session.gradcam_image_b64),
                width=5.5 * inch,
                height=3.8 * inch,
                kind="proportional",
            )
            story.append(img)
        except Exception:
            story.append(Paragraph("<i>Heatmap could not be rendered.</i>", styles["ReportBodyText"]))
        story.append(Spacer(1, 12))

    # --- Annotated Image ---
    if session.annotated_image_b64:
        story.append(PageBreak())
        story.append(Paragraph("Annotated Image (YOLO Detections)", styles["SectionHeading"]))
        try:
            img = RLImage(
                _b64_to_image_reader(session.annotated_image_b64),
                width=5.5 * inch,
                height=3.8 * inch,
                kind="proportional",
            )
            story.append(img)
        except Exception:
            story.append(Paragraph("<i>Image could not be rendered.</i>", styles["ReportBodyText"]))
        story.append(Spacer(1, 12))

    # --- YOLO Detections Table ---
    story.append(Paragraph("Object Detections (YOLO)", styles["SectionHeading"]))
    if session.detections:
        table_data = [["#", "Class", "Confidence", "Bounding Box"]]
        for i, d in enumerate(session.detections, 1):
            table_data.append([
                str(i),
                d.get("class_name", "unknown"),
                f"{d.get('confidence', 0):.1%}",
                str([round(v) for v in d.get("bbox", [])]),
            ])
        t = Table(table_data, colWidths=[30, 120, 80, 200])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.96, 0.96, 0.98)]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No objects detected.", styles["ReportBodyText"]))
    story.append(Spacer(1, 16))

    # --- ViT Classifications Table ---
    story.append(Paragraph("Scene Classification (ViT)", styles["SectionHeading"]))
    if session.classifications:
        table_data = [["Rank", "Class", "Confidence"]]
        for i, c in enumerate(session.classifications, 1):
            label = c.get("class_name", f"Class {c.get('class_id', '?')}")
            table_data.append([
                str(i),
                label,
                f"{c.get('confidence', 0):.1%}",
            ])
        t = Table(table_data, colWidths=[40, 250, 80])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CYAN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.96, 0.96, 0.98)]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    story.append(Spacer(1, 16))

    # --- Grad-CAM Info ---
    if session.gradcam_info and not session.gradcam_info.get("error"):
        story.append(Paragraph("Grad-CAM Explainability", styles["SectionHeading"]))
        target = session.gradcam_info.get("target_class_name", "N/A")
        conf = session.gradcam_info.get("confidence", 0)
        story.append(Paragraph(
            f"<b>Target class:</b> {target} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Confidence:</b> {conf:.4f}",
            styles["ReportBodyText"],
        ))
        story.append(Spacer(1, 12))

    # --- NL Summary ---
    if session.nl_summary:
        story.append(Paragraph("Analysis Summary", styles["SectionHeading"]))
        story.append(Paragraph(session.nl_summary, styles["ReportBodyText"]))
        story.append(Spacer(1, 16))

    # --- Footer disclaimer on every page ---
    def _add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(ACCENT_ROSE)
        canvas.drawCentredString(
            A4[0] / 2,
            12 * mm,
            f"⚠ {settings.medical_disclaimer}",
        )
        canvas.setFillColor(MUTED_TEXT)
        canvas.drawCentredString(
            A4[0] / 2,
            8 * mm,
            f"AstraVisionX Report — Session {session.session_id} — Page {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
    return buffer.getvalue()

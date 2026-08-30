"""Reusable PDF exports for predictions, evaluation and surface analysis.

The exporter only serialises results that already exist in the UI. It never
runs inference, evaluates Test data, changes source images or overwrites files.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import io
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportImage,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

from .ui_core import check_destination


PAGE_WIDTH, PAGE_HEIGHT = A4
AVAILABLE_WIDTH = PAGE_WIDTH - 30 * mm
MAX_IMAGE_HEIGHT = PAGE_HEIGHT - 65 * mm

PREDICTION_PDF_FIELDS = [
    ("source", "Image"),
    ("method", "Method"),
    ("predicted_stage", "Ripeness"),
    ("objects_detected", "Fruits"),
    ("detected_fruits", "Types"),
    ("score_unripe", "Unripe"),
    ("score_ripe", "Ripe"),
    ("score_overripe", "Overripe"),
    ("processing_ms", "Processing"),
]

METRICS_PDF_FIELDS = [
    ("method", "Method"),
    ("scope", "Scope"),
    ("n_images", "Images"),
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro F1"),
]

SURFACE_PDF_FIELDS = [
    ("source", "Image"),
    ("segmentation_method", "Segmentation"),
    ("blemish_fraction", "Blemish"),
    ("quality_grade", "Quality grade"),
    ("objects_detected", "Objects"),
    ("equivalent_diameter_px", "Diameter"),
]


def _text(value) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _format_value(key: str, value, labels: dict[str, str]) -> str:
    if key in {"method", "segmentation_method"}:
        return labels.get(str(value), _text(value))

    if value is None or value == "":
        return "-"

    if key in {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "blemish_fraction",
        "foreground_fraction",
        "score_unripe",
        "score_ripe",
        "score_overripe",
    } and isinstance(value, (int, float)):
        return f"{float(value):.2%}"

    if key.endswith("_ms") and isinstance(value, (int, float)):
        return f"{float(value):.1f} ms"

    if key == "equivalent_diameter_px" and isinstance(value, (int, float)):
        return f"{float(value):.1f} px"

    if key == "source":
        return Path(str(value)).name

    return _text(value)


def _column_widths(fields: list[tuple[str, str]]) -> list[float]:
    weights = []
    for key, _ in fields:
        if key == "source":
            weights.append(2.3)
        elif key in {"method", "segmentation_method"}:
            weights.append(1.8)
        else:
            weights.append(1.0)
    unit = AVAILABLE_WIDTH / sum(weights)
    return [unit * weight for weight in weights]


def _image_flowables(image: Image.Image, buffers: list[io.BytesIO], styles):
    rgb = image.convert("RGB")
    scale = min(1.0, AVAILABLE_WIDTH / rgb.width)
    segment_height_px = max(1, int(MAX_IMAGE_HEIGHT / scale))
    segments = [
        rgb.crop((0, top, rgb.width, min(top + segment_height_px, rgb.height)))
        for top in range(0, rgb.height, segment_height_px)
    ]

    flowables = []
    for index, segment in enumerate(segments, 1):
        buffer = io.BytesIO()
        segment.save(buffer, format="PNG")
        buffer.seek(0)
        buffers.append(buffer)

        width = segment.width * scale
        height = segment.height * scale
        label = "Visual evidence"
        if len(segments) > 1:
            label += f" - part {index} of {len(segments)}"
        flowables.extend(
            [
                Paragraph(label, styles["SectionHeading"]),
                Spacer(1, 2 * mm),
                ReportImage(buffer, width=width, height=height),
            ]
        )
        if index != len(segments):
            flowables.append(PageBreak())
    return flowables


def _page(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd8d0"))
    canvas.line(15 * mm, 14 * mm, PAGE_WIDTH - 15 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52645a"))
    canvas.drawString(15 * mm, 9 * mm, "Fruit Ripeness - generated report")
    canvas.drawRightString(
        PAGE_WIDTH - 15 * mm,
        9 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def export_report_pdf(
    path: Path,
    *,
    title: str,
    rows: list[dict],
    fields: list[tuple[str, str]],
    protected: list[Path],
    image: Image.Image | None = None,
    notes: list[str] | tuple[str, ...] = (),
    labels: dict[str, str] | None = None,
) -> None:
    """Create a new, non-overwriting A4 PDF from existing UI results."""

    if not rows:
        raise ValueError("No report rows are available to export")
    if not fields:
        raise ValueError("At least one report field is required")

    target = check_destination(path, protected)
    labels = labels or {}
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#214737"),
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.HexColor("#214737"),
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeaderCell",
            parent=styles["Cell"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
            alignment=TA_CENTER,
        )
    )

    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M:%S UTC")
    story = [
        Paragraph(escape(title), styles["ReportTitle"]),
        Paragraph(f"Generated: {escape(generated)}", styles["BodyText"]),
    ]
    for note in notes:
        story.extend([Spacer(1, 1.5 * mm), Paragraph(escape(note), styles["BodyText"])])

    buffers: list[io.BytesIO] = []
    if image is not None:
        story.extend([Spacer(1, 3 * mm), *_image_flowables(image, buffers, styles)])

    story.extend([Spacer(1, 4 * mm), Paragraph("Recorded results", styles["SectionHeading"])])
    table_data = [
        [Paragraph(escape(heading), styles["HeaderCell"]) for _, heading in fields]
    ]
    for row in rows:
        table_data.append(
            [
                Paragraph(
                    escape(_format_value(key, row.get(key), labels)),
                    styles["Cell"],
                )
                for key, _ in fields
            ]
        )

    table = LongTable(
        table_data,
        colWidths=_column_widths(fields),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#214737")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aebdb4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf3ef")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)

    payload = io.BytesIO()
    document = SimpleDocTemplate(
        payload,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=20 * mm,
        title=title,
        author="Fruit Ripeness Comparison Studio",
    )
    document.build(story, onFirstPage=_page, onLaterPages=_page)

    with target.open("xb") as stream:
        stream.write(payload.getvalue())

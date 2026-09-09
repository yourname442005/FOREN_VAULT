import json
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "evidence" / "reports"


def _safe(value, default="Not available"):
    if value is None or value == "":
        return default
    return str(value)


def _format_bytes(value):
    if value is None:
        return "Not available"

    value = float(value)

    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} TB"


def _percentage(value):
    if value is None:
        return "Not available"

    return f"{float(value) * 100:.1f}%"


def _create_styles():
    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=10 * mm,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            spaceBefore=7 * mm,
            spaceAfter=4 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=10,
        ),
    }


def _table(data, widths=None, header=True):
    table = Table(
        data,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )

    style_commands = [
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey,
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP",
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold",
        ),
        (
            "FONTNAME",
            (0, 1),
            (-1, -1),
            "Helvetica",
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            7.5,
        ),
        (
            "LEFTPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "RIGHTPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            4,
        ),
        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            4,
        ),
    ]

    if header:
        style_commands.append(
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey,
            )
        )

    table.setStyle(TableStyle(style_commands))

    return table


def _add_key_value_section(story, styles, title, values):
    story.append(Paragraph(title, styles["heading"]))

    data = [["Field", "Value"]]

    for key, value in values:
        data.append(
            [
                Paragraph(_safe(key), styles["small"]),
                Paragraph(_safe(value), styles["small"]),
            ]
        )

    story.append(
        _table(
            data,
            widths=[55 * mm, 125 * mm],
        )
    )


def _add_chain_of_custody(story, styles, history):
    story.append(
        Paragraph(
            "CHAIN OF CUSTODY",
            styles["heading"],
        )
    )

    if not history:
        story.append(
            Paragraph(
                "No chain-of-custody events recorded.",
                styles["body"],
            )
        )
        return

    data = [
        [
            "Timestamp",
            "Action",
            "Actor",
            "Description",
        ]
    ]

    for event in history:
        data.append(
            [
                Paragraph(
                    _safe(event.get("timestamp")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(event.get("action")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(event.get("actor")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(event.get("description")),
                    styles["small"],
                ),
            ]
        )

    story.append(
        _table(
            data,
            widths=[
                38 * mm,
                38 * mm,
                28 * mm,
                76 * mm,
            ],
        )
    )


def _add_cameras(story, styles, cameras):
    story.append(
        Paragraph(
            "CAMERAS",
            styles["heading"],
        )
    )

    if not cameras:
        story.append(
            Paragraph(
                "No cameras identified.",
                styles["body"],
            )
        )
        return

    data = [
        [
            "Camera ID",
            "Name",
            "Source",
        ]
    ]

    for camera in cameras:
        data.append(
            [
                Paragraph(
                    _safe(camera.get("camera_id")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(camera.get("name")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(camera.get("source")),
                    styles["small"],
                ),
            ]
        )

    story.append(
        _table(
            data,
            widths=[
                35 * mm,
                65 * mm,
                80 * mm,
            ],
        )
    )


def _add_timeline(story, styles, timeline):
    story.append(
        Paragraph(
            "RECORDING TIMELINE",
            styles["heading"],
        )
    )

    if not timeline:
        story.append(
            Paragraph(
                "No recording timeline entries available.",
                styles["body"],
            )
        )
        return

    data = [
        [
            "Camera",
            "Filename",
            "Start",
            "End",
            "Timezone",
            "Status",
            "Format",
            "Deleted",
        ]
    ]

    for item in timeline:
        tz_info = ""
        status_info = ""

        start_ts = item.get("start_timestamp")
        if start_ts:
            tz_info = _safe(start_ts.get("timezone")) or "Unknown"
            status_info = _safe(start_ts.get("normalization_status")) or ""

        data.append(
            [
                Paragraph(
                    _safe(item.get("camera_id")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("filename")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("start_time")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("end_time")),
                    styles["small"],
                ),
                Paragraph(
                    tz_info,
                    styles["small"],
                ),
                Paragraph(
                    status_info,
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("format")),
                    styles["small"],
                ),
                Paragraph(
                    str(bool(item.get("deleted"))),
                    styles["small"],
                ),
            ]
        )

    story.append(
        _table(
            data,
            widths=[
                18 * mm,
                35 * mm,
                28 * mm,
                28 * mm,
                22 * mm,
                18 * mm,
                15 * mm,
                12 * mm,
            ],
        )
    )


def _add_correlations(story, styles, correlations):
    story.append(
        Paragraph(
            "CROSS-CAMERA CORRELATION",
            styles["heading"],
        )
    )

    if not correlations:
        story.append(
            Paragraph(
                "No overlapping cross-camera recording intervals identified.",
                styles["body"],
            )
        )
        return

    data = [
        [
            "Camera A",
            "Camera B",
            "Recording A",
            "Recording B",
            "Overlap Start",
            "Overlap End",
            "TZ Comparable",
        ]
    ]

    for item in correlations:
        tz_comparable = item.get("timezone_comparable")
        tz_comparable_str = (
            "Yes" if tz_comparable else "No"
        ) if tz_comparable is not None else "Unknown"

        data.append(
            [
                Paragraph(
                    _safe(item.get("camera_a")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("camera_b")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("recording_a")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("recording_b")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("overlap_start")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(item.get("overlap_end")),
                    styles["small"],
                ),
                Paragraph(
                    tz_comparable_str,
                    styles["small"],
                ),
            ]
        )

    story.append(
        _table(
            data,
            widths=[
                18 * mm,
                18 * mm,
                30 * mm,
                30 * mm,
                28 * mm,
                28 * mm,
                20 * mm,
            ],
        )
    )


def _add_recovery(story, styles, recovery_results):
    story.append(
        Paragraph(
            "DELETED FILE RECOVERY",
            styles["heading"],
        )
    )

    if not recovery_results:
        story.append(
            Paragraph(
                "No deleted-file recovery results were produced.",
                styles["body"],
            )
        )
        return

    data = [
        [
            "Filename",
            "Method",
            "Status",
            "Size",
            "Boundary",
            "SHA-256",
        ]
    ]

    for result in recovery_results:
        recovery_status = result.get(
            "recovery_status",
            "UNKNOWN",
        )

        boundary = result.get("boundary") or {}
        boundary_confidence = boundary.get(
            "confidence", "N/A"
        )
        boundary_method = boundary.get("method")
        boundary_label = boundary_confidence

        if boundary_method:
            boundary_label = (
                f"{boundary_confidence} "
                f"({boundary_method})"
            )

        data.append(
            [
                Paragraph(
                    _safe(result.get("filename")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(result.get("method")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(recovery_status),
                    styles["small"],
                ),
                Paragraph(
                    _format_bytes(result.get("size")),
                    styles["small"],
                ),
                Paragraph(
                    _safe(boundary_label),
                    styles["small"],
                ),
                Paragraph(
                    _safe(result.get("sha256")),
                    styles["small"],
                ),
            ]
        )

    story.append(
        _table(
            data,
            widths=[
                35 * mm,
                28 * mm,
                28 * mm,
                20 * mm,
                30 * mm,
                49 * mm,
            ],
        )
    )


def generate_forensic_report(
    evidence_data: dict,
    output_directory: Path | None = None,
) -> dict:
    evidence_id = _safe(
        evidence_data.get("evidence_id"),
        "UNKNOWN",
    )

    if output_directory is None:
        output_directory = REPORTS_DIR

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        output_directory
        / f"{evidence_id}_forensic_report.pdf"
    )

    json_path = (
        output_directory
        / f"{evidence_id}_forensic_report.json"
    )

    report_generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    report_data = {
        "report_type": "FORENVAULT Forensic Evidence Analysis Report",
        "report_version": "1.0",
        "generated_at": report_generated_at,
        "evidence": evidence_data,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report_data,
            file,
            indent=2,
            default=str,
        )

    styles = _create_styles()

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="FORENVAULT Forensic Evidence Analysis Report",
        author="FORENVAULT",
    )

    story = []

    story.append(
        Paragraph(
            "FORENVAULT",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            "FORENSIC EVIDENCE ANALYSIS REPORT",
            styles["subtitle"],
        )
    )

    _add_key_value_section(
        story,
        styles,
        "EVIDENCE INFORMATION",
        [
            (
                "Evidence ID",
                evidence_data.get("evidence_id"),
            ),
            (
                "Original Filename",
                evidence_data.get("filename"),
            ),
            (
                "Evidence Type",
                evidence_data.get("file_type"),
            ),
            (
                "Classification",
                evidence_data.get("classification"),
            ),
            (
                "File Size",
                _format_bytes(
                    evidence_data.get("size")
                ),
            ),
            (
                "SHA-256",
                evidence_data.get("sha256"),
            ),
            (
                "Status",
                evidence_data.get("status"),
            ),
        ],
    )

    integrity = evidence_data.get(
        "integrity_validation"
    ) or {}

    _add_key_value_section(
        story,
        styles,
        "INTEGRITY VALIDATION",
        [
            (
                "Algorithm",
                integrity.get("algorithm"),
            ),
            (
                "Validation Status",
                integrity.get("status"),
            ),
            (
                "Valid",
                integrity.get("valid"),
            ),
            (
                "Expected SHA-256",
                integrity.get("expected_sha256"),
            ),
            (
                "Current SHA-256",
                integrity.get("current_sha256"),
            ),
        ],
    )

    vendor_analysis = (
        evidence_data.get("vendor_analysis")
        or {}
    )

    _add_key_value_section(
        story,
        styles,
        "VENDOR IDENTIFICATION",
        [
            (
                "Vendor",
                evidence_data.get("vendor")
                or vendor_analysis.get("vendor"),
            ),
            (
                "Confidence",
                _percentage(
                    evidence_data.get(
                        "vendor_confidence"
                    )
                ),
            ),
            (
                "Detection Method",
                evidence_data.get(
                    "detection_method"
                ),
            ),
        ],
    )

    parser_result = (
        evidence_data.get("parser_result")
        or {}
    )

    dvr_evidence = (
        parser_result.get("dvr_evidence")
        or {}
    )

    _add_key_value_section(
        story,
        styles,
        "DVR INFORMATION",
        [
            (
                "Parser",
                parser_result.get("parser"),
            ),
            (
                "Vendor",
                dvr_evidence.get("vendor"),
            ),
            (
                "Model",
                dvr_evidence.get("model"),
            ),
            (
                "Firmware",
                dvr_evidence.get("firmware"),
            ),
        ],
    )

    capabilities = parser_result.get("capabilities") or {}
    supported_caps = capabilities.get("supported") or []
    unsupported_caps = capabilities.get("unsupported") or []

    if supported_caps or unsupported_caps:
        cap_lines = []
        if supported_caps:
            cap_lines.append(
                ("Supported", ", ".join(supported_caps))
            )
        if unsupported_caps:
            cap_lines.append(
                ("Not Supported", ", ".join(unsupported_caps))
            )
        _add_key_value_section(
            story,
            styles,
            "PARSER CAPABILITIES",
            cap_lines,
        )

    _add_cameras(
        story,
        styles,
        dvr_evidence.get("cameras") or [],
    )

    _add_timeline(
        story,
        styles,
        evidence_data.get("timeline") or [],
    )

    _add_correlations(
        story,
        styles,
        evidence_data.get(
            "cross_camera_correlations"
        )
        or [],
    )

    _add_recovery(
        story,
        styles,
        evidence_data.get(
            "recovery_results"
        )
        or [],
    )

    recovery_summary = (
        evidence_data.get("recovery_summary")
        or {}
    )

    _add_key_value_section(
        story,
        styles,
        "RECOVERY SUMMARY",
        [
            (
                "Recovery Attempted",
                recovery_summary.get(
                    "attempted"
                ),
            ),
            (
                "Recovered Files",
                recovery_summary.get(
                    "recovered"
                ),
            ),
            (
                "Failed Attempts",
                recovery_summary.get(
                    "failed"
                ),
            ),
        ],
    )

    _add_chain_of_custody(
        story,
        styles,
        evidence_data.get(
            "chain_of_custody"
        )
        or [],
    )

    analysis_pipeline = (
        evidence_data.get("analysis_pipeline")
        or {}
    )

    stages = analysis_pipeline.get("stages") or {}

    if stages:
        story.append(
            Paragraph(
                "ANALYSIS PIPELINE STATUS",
                styles["heading"],
            )
        )

        data = [
            [
                "Stage",
                "Status",
                "Details",
            ]
        ]

        for stage_name, stage_info in stages.items():
            status = stage_info.get("status", "UNKNOWN")
            error = stage_info.get("error")
            details = stage_info.get("details", {})

            detail_parts = []
            if error:
                detail_parts.append(f"Error: {error}")
            if details:
                for key, value in details.items():
                    detail_parts.append(f"{key}: {value}")

            detail_str = "; ".join(detail_parts) if detail_parts else ""

            display_name = stage_name.replace("_", " ").title()

            data.append(
                [
                    Paragraph(
                        _safe(display_name),
                        styles["small"],
                    ),
                    Paragraph(
                        _safe(status),
                        styles["small"],
                    ),
                    Paragraph(
                        _safe(detail_str),
                        styles["small"],
                    ),
                ]
            )

        story.append(
            _table(
                data,
                widths=[55 * mm, 35 * mm, 90 * mm],
            )
        )

    analysis_errors = (
        evidence_data.get("analysis_errors") or []
    )

    if analysis_errors:
        story.append(
            Paragraph(
                "WARNINGS AND ERRORS",
                styles["heading"],
            )
        )

        data = [
            [
                "Stage",
                "Error",
            ]
        ]

        for error_entry in analysis_errors:
            data.append(
                [
                    Paragraph(
                        _safe(
                            error_entry.get("stage", "unknown")
                        ),
                        styles["small"],
                    ),
                    Paragraph(
                        _safe(
                            error_entry.get("error", "unknown")
                        ),
                        styles["small"],
                    ),
                ]
            )

        story.append(
            _table(
                data,
                widths=[55 * mm, 125 * mm],
            )
        )

    limitations = []

    if not dvr_evidence.get("model"):
        limitations.append(
            "Device model could not be determined from available metadata."
        )

    if not dvr_evidence.get("cameras"):
        limitations.append(
            "No camera configuration was identified in the evidence."
        )

    if not evidence_data.get("timeline"):
        limitations.append(
            "No recording timeline could be constructed."
        )

    if recovery_summary.get("duplicates_suppressed", 0) > 0:
        limitations.append(
            f"{recovery_summary['duplicates_suppressed']} duplicate recovery "
            f"candidate(s) were suppressed."
        )

    if analysis_errors:
        limitations.append(
            f"{len(analysis_errors)} analysis error(s) occurred during processing."
        )

    limitations.append(
        "No proprietary filesystem decoding was performed. "
        "Generic forensic analysis was used."
    )

    limitations.append(
        "Deleted file recovery boundaries are estimated and "
        "may not represent exact original file boundaries."
    )

    story.append(
        Paragraph(
            "LIMITATIONS AND DISCLAIMERS",
            styles["heading"],
        )
    )

    for limitation in limitations:
        story.append(
            Paragraph(
                f"- {_safe(limitation)}",
                styles["body"],
            )
        )

    story.append(
        Paragraph(
            "REPORT GENERATION",
            styles["heading"],
        )
    )

    story.append(
        Paragraph(
            (
                f"Report generated by FORENVAULT at "
                f"{report_generated_at}."
            ),
            styles["body"],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                "This report contains the structured forensic "
                "analysis results available to FORENVAULT at "
                "the time of generation."
            ),
            styles["small"],
        )
    )

    document.build(story)

    return {
        "status": "GENERATED",
        "evidence_id": evidence_id,
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "generated_at": report_generated_at,
    }
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VendorDetectionResult:
    vendor: str
    confidence: float
    detection_method: str
    evidence: list[str] = field(default_factory=list)


VENDOR_MARKERS = {
    "hikvision": [
        "hikvision",
        "hik",
        "ivms",
        "isapi",
        "ds-",
        "hikvision.com",
    ],
    "dahua": [
        "dahua",
        "dhi-",
        "dh-",
        "smartpss",
        "dmss",
        "dahuasecurity",
    ],
    "cp_plus": [
        "cp plus",
        "cpplus",
        "cp-plus",
        "cpplusworld",
    ],
    "uniview": [
        "uniview",
        "unv",
        "nvr301",
        "nvr302",
        "ezstation",
    ],
    "honeywell": [
        "honeywell",
        "honeywell security",
        "hrn",
        "hnb",
    ],
    "tp_link": [
        "tp-link",
        "tplink",
        "vigi",
        "vigi security",
    ],
    "godrej": [
        "godrej",
        "godrej security",
        "securicam",
    ],
    "matrix": [
        "matrix",
        "matrix comsec",
        "visionpro",
        "sarv",
    ],
}


VENDOR_DISPLAY_NAMES = {
    "hikvision": "Hikvision",
    "dahua": "Dahua",
    "cp_plus": "CP Plus",
    "uniview": "Uniview",
    "honeywell": "Honeywell",
    "tp_link": "TP-Link",
    "godrej": "Godrej",
    "matrix": "Matrix",
}


TEXT_EXTENSIONS = {
    ".conf",
    ".cfg",
    ".ini",
    ".json",
    ".xml",
    ".txt",
    ".log",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def extract_file_content(
    image_path: Path,
    filesystem_code: str,
    offset: int,
    inode: int,
) -> str | None:
    """
    Extract a file directly from the forensic image using Sleuth Kit icat.
    """

    command = [
        "icat",
        "-f",
        filesystem_code,
        "-o",
        str(offset),
        str(image_path),
        str(inode),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
        )

        return result.stdout.decode(
            "utf-8",
            errors="ignore",
        )

    except subprocess.CalledProcessError:
        return None


def score_vendor(
    vendor: str,
    sources: list[tuple[str, str]],
) -> tuple[float, list[str]]:
    markers = VENDOR_MARKERS[vendor]

    score = 0.0
    evidence = []
    matched_markers = set()

    for source_name, source_text in sources:
        text = normalize(source_text)

        for marker in markers:
            marker_normalized = normalize(marker)

            if marker_normalized not in text:
                continue

            if marker_normalized in matched_markers:
                continue

            matched_markers.add(marker_normalized)

            if source_name == "configuration":
                points = 0.50
            elif source_name == "metadata":
                points = 0.45
            elif source_name == "log":
                points = 0.25
            elif source_name == "filename":
                points = 0.10
            elif source_name == "directory":
                points = 0.10
            else:
                points = 0.05

            score += points

            evidence.append(
                f"{source_name}: detected marker '{marker}'"
            )

    return min(score, 1.0), evidence


def detect_vendor(
    file_path: Path,
    file_type: str,
    filesystem_analysis: dict | None = None,
) -> VendorDetectionResult:

    if not filesystem_analysis:
        return VendorDetectionResult(
            vendor="unknown",
            confidence=0.0,
            detection_method="insufficient_evidence",
            evidence=[
                "Filesystem analysis is required for vendor detection."
            ],
        )

    files = filesystem_analysis.get("files", [])

    filesystem = filesystem_analysis.get("filesystem", {})
    filesystem_code = filesystem.get("code")
    partition = filesystem_analysis.get("partition", {})
    offset = partition.get("start_sector")

    if not filesystem_code or offset is None:
        return VendorDetectionResult(
            vendor="unknown",
            confidence=0.0,
            detection_method="missing_filesystem_context",
            evidence=[
                "Filesystem type or partition offset is unavailable."
            ],
        )

    sources: list[tuple[str, str]] = []

    for entry in files:
        name = entry.get("name", "")
        entry_type = entry.get("type")
        suffix = Path(name).suffix.lower()

        if not name:
            continue

        if entry_type == "directory":
            sources.append(("directory", name))
            continue

        sources.append(("filename", name))

        if suffix not in TEXT_EXTENSIONS:
            continue

        content = extract_file_content(
            file_path,
            filesystem_code,
            offset,
            entry["inode"],
        )

        if not content:
            continue

        if suffix == ".log":
            source_type = "log"
        elif suffix in {".json", ".xml"}:
            source_type = "metadata"
        else:
            source_type = "configuration"

        sources.append((source_type, content))

    results = []

    for vendor in VENDOR_MARKERS:
        score, evidence = score_vendor(
            vendor,
            sources,
        )

        results.append(
            {
                "vendor": vendor,
                "score": score,
                "evidence": evidence,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    winner = results[0]

    if winner["score"] == 0:
        return VendorDetectionResult(
            vendor="unknown",
            confidence=0.0,
            detection_method="no_vendor_markers",
            evidence=[
                "No known vendor markers were detected."
            ],
        )

    if winner["score"] < 0.40:
        return VendorDetectionResult(
            vendor="unknown",
            confidence=round(winner["score"], 2),
            detection_method="weak_vendor_signal",
            evidence=winner["evidence"],
        )

    if winner["score"] >= 0.70:
        method = "strong_content_marker_match"
    else:
        method = "content_marker_match"

    return VendorDetectionResult(
        vendor=VENDOR_DISPLAY_NAMES[winner["vendor"]],
        confidence=round(winner["score"], 2),
        detection_method=method,
        evidence=winner["evidence"],
    )

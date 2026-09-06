from pathlib import Path


MEDIA_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "audio/mpeg",
    "audio/wav",
    "image/jpeg",
    "image/png",
}


FORENSIC_IMAGE_TYPES = {
    "forensic-image/e01",
    "forensic-image/raw",
}


FORENSIC_IMAGE_EXTENSIONS = {
    ".e01",
    ".ex01",
    ".raw",
    ".dd",
    ".img",
    ".dmg",
    ".001",
}


def classify_evidence(file_path: Path, file_type: str) -> str:
    if file_type in FORENSIC_IMAGE_TYPES:
        return "FORENSIC_IMAGE"

    if file_path.suffix.lower() in FORENSIC_IMAGE_EXTENSIONS:
        return "FORENSIC_IMAGE"

    if file_type in MEDIA_TYPES:
        return "MEDIA"

    return "UNKNOWN"
from pathlib import Path


def identify_file_type(file_path: Path) -> str:
    with file_path.open("rb") as file:
        header = file.read(32)

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "video/mp4"

    if header.startswith(b"EVF"):
        return "forensic-image/e01"

    return "application/octet-stream"
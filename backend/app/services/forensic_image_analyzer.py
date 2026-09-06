import subprocess
from pathlib import Path


EWF_EXTENSIONS = {
    ".e01",
    ".ex01",
}


RAW_IMAGE_EXTENSIONS = {
    ".dmg",
    ".raw",
    ".dd",
    ".img",
    ".001",
}


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )


def analyze_e01(file_path: Path) -> dict:
    result = _run_command(
        [
            "ewfinfo",
            str(file_path),
        ]
    )

    return {
        "tool": "ewfinfo",
        "format": "E01/EWF",
        "raw_output": result.stdout,
    }


def analyze_raw_image(file_path: Path) -> dict:
    return {
        "tool": "direct_image",
        "format": "RAW/DMG",
        "path": str(file_path),
        "size": file_path.stat().st_size,
    }


def analyze_forensic_image(file_path: Path) -> dict:
    suffix = file_path.suffix.lower()

    if suffix in EWF_EXTENSIONS:
        return analyze_e01(file_path)

    if suffix in RAW_IMAGE_EXTENSIONS:
        return analyze_raw_image(file_path)

    raise ValueError(
        f"Unsupported forensic image format: "
        f"{file_path.suffix or 'unknown'}"
    )
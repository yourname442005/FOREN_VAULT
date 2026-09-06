import subprocess
import tempfile
from pathlib import Path


EWF_EXTENSIONS = {
    ".e01",
    ".ex01",
}


DIRECT_IMAGE_EXTENSIONS = {
    ".dmg",
    ".raw",
    ".dd",
    ".img",
    ".001",
}


def export_e01_to_raw(
    e01_path: Path,
) -> tuple[Path, tempfile.TemporaryDirectory]:
    temp_directory = tempfile.TemporaryDirectory(
        prefix="forenvault_"
    )

    output_base = Path(temp_directory.name) / "evidence"

    command = [
        "ewfexport",
        "-t",
        str(output_base),
        "-u",
        str(e01_path),
    ]

    result = subprocess.run(
        command,
        input="\n",
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        temp_directory.cleanup()

        raise RuntimeError(
            "ewfexport failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    raw_candidates = [
        Path(f"{output_base}.raw"),
        Path(f"{output_base}.001"),
        Path(f"{output_base}.000"),
        output_base,
    ]

    raw_path = next(
        (
            candidate
            for candidate in raw_candidates
            if candidate.exists()
        ),
        None,
    )

    if raw_path is None:
        temp_directory.cleanup()

        raise FileNotFoundError(
            "ewfexport completed but RAW output was not found.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return raw_path, temp_directory


def export_forensic_image_to_raw(
    image_path: Path,
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    suffix = image_path.suffix.lower()

    if suffix in EWF_EXTENSIONS:
        return export_e01_to_raw(image_path)

    if suffix in DIRECT_IMAGE_EXTENSIONS:
        return image_path, None

    raise ValueError(
        f"Unsupported forensic image format: "
        f"{image_path.suffix or 'unknown'}"
    )
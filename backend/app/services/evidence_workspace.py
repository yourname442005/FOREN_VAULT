import shutil
from pathlib import Path

from app.services.evidence_service import calculate_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INCOMING_DIR = PROJECT_ROOT / "evidence" / "incoming"
WORKING_DIR = PROJECT_ROOT / "evidence" / "working"


def create_working_copy(
    evidence_id: str,
    stored_filename: str,
    original_sha256: str,
) -> Path:
    source = INCOMING_DIR / stored_filename
    destination = WORKING_DIR / evidence_id / stored_filename

    if not source.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source, destination)

    working_sha256 = calculate_sha256(destination)

    if working_sha256 != original_sha256:
        destination.unlink(missing_ok=True)

        raise ValueError(
            "Working copy SHA-256 does not match original evidence."
        )

    return destination
from pathlib import Path

from app.services.evidence_service import calculate_sha256


def validate_evidence_integrity(
    file_path: Path,
    expected_sha256: str,
) -> dict:
    current_sha256 = calculate_sha256(file_path)

    is_valid = current_sha256.lower() == expected_sha256.lower()

    return {
        "valid": is_valid,
        "status": "VALID" if is_valid else "TAMPERED",
        "expected_sha256": expected_sha256,
        "current_sha256": current_sha256,
        "algorithm": "SHA-256",
    }
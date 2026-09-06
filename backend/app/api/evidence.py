import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile

from app.core.database import get_connection
from app.parsers.vendor_registry import create_default_registry
from app.services.chain_of_custody import (
    get_custody_history,
    record_custody_event,
)
from app.services.deleted_file_recovery import (
    carve_h264_candidates,
    recover_deleted_file,
)
from app.services.dvr_structure_detector import detect_dvr_structure
from app.services.event_correlator import correlate_recordings
from app.services.evidence_classifier import classify_evidence
from app.services.evidence_service import (
    EVIDENCE_DIR,
    calculate_sha256,
)
from app.services.evidence_validator import validate_evidence_integrity
from app.services.evidence_workspace import create_working_copy
from app.services.file_identifier import identify_file_type
from app.services.filesystem_analyzer import analyze_filesystem
from app.services.forensic_image_analyzer import analyze_forensic_image
from app.services.forensic_image_exporter import export_forensic_image_to_raw
from app.services.forensic_report_generator import (
    generate_forensic_report,
)
from app.services.media_analyzer import analyze_media
from app.services.recording_timeline import build_timeline
from app.services.vendor_detector import detect_vendor


router = APIRouter(prefix="/api/evidence", tags=["Evidence"])


@router.post("/upload")
async def upload_evidence(file: UploadFile = File(...)):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    evidence_id = f"EV-{uuid4().hex[:8].upper()}"
    original_name = Path(file.filename or "unknown").name
    destination = EVIDENCE_DIR / f"{evidence_id}_{original_name}"

    with destination.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    sha256 = calculate_sha256(destination)
    file_type = identify_file_type(destination)
    classification = classify_evidence(destination, file_type)
    vendor_result = detect_vendor(destination, file_type)

    record_custody_event(
        evidence_id=evidence_id,
        action="EVIDENCE_IMPORTED",
        description="Evidence file imported into FORENVAULT.",
        details={
            "filename": original_name,
            "file_size": destination.stat().st_size,
            "sha256": sha256,
            "file_type": file_type,
        },
    )

    working_copy = create_working_copy(
        evidence_id=evidence_id,
        stored_filename=destination.name,
        original_sha256=sha256,
    )

    record_custody_event(
        evidence_id=evidence_id,
        action="WORKING_COPY_CREATED",
        description="Forensic working copy created from original evidence.",
        details={
            "working_copy": str(working_copy),
            "original_sha256": sha256,
        },
    )

    integrity_validation = validate_evidence_integrity(
        destination,
        sha256,
    )

    record_custody_event(
        evidence_id=evidence_id,
        action="INTEGRITY_VERIFIED",
        description="Original evidence integrity verified using SHA-256.",
        details=integrity_validation,
    )

    media_metadata = None
    forensic_image_metadata = None
    filesystem_metadata = None
    dvr_structure = None
    vendor_analysis = None
    recovery_results = []
    carving_failures = []
    parser_result = None
    timeline = []
    cross_camera_correlations = []
    analysis_errors = []

    if classification == "MEDIA":
        media_metadata = analyze_media(working_copy)

    elif classification == "FORENSIC_IMAGE":
        record_custody_event(
            evidence_id=evidence_id,
            action="FORENSIC_ANALYSIS_STARTED",
            description="Forensic image analysis started.",
            details={
                "file_type": file_type,
                "filename": original_name,
            },
        )

        forensic_image_metadata = analyze_forensic_image(
            working_copy
        )

        raw_image, temp_directory = export_forensic_image_to_raw(
            working_copy
        )

        try:
            # -------------------------------------------------
            # Filesystem analysis
            # -------------------------------------------------

            try:
                filesystem_metadata = analyze_filesystem(raw_image)
            except (
                subprocess.CalledProcessError,
                ValueError,
                OSError,
            ) as error:
                analysis_errors.append({
                    "stage": "filesystem_analysis",
                    "error": str(error),
                })

            # -------------------------------------------------
            # DVR structure detection
            # -------------------------------------------------

            if filesystem_metadata is not None:
                try:
                    dvr_structure = detect_dvr_structure(
                        filesystem_metadata["files"]
                    )
                except Exception as error:
                    analysis_errors.append({
                        "stage": "dvr_structure_detection",
                        "error": str(error),
                    })

            if filesystem_metadata is not None:
                record_custody_event(
                    evidence_id=evidence_id,
                    action="FILESYSTEM_ANALYZED",
                    description="Filesystem analysis completed.",
                    details={
                        "filesystem": filesystem_metadata["filesystem"],
                        "partition": filesystem_metadata["partition"],
                        "file_count": len(filesystem_metadata["files"]),
                    }
                    if dvr_structure is not None
                    else {
                        "filesystem": filesystem_metadata.get("filesystem"),
                        "partition": filesystem_metadata.get("partition"),
                    },
                )

            # -------------------------------------------------
            # Recovery
            # -------------------------------------------------

            if filesystem_metadata is not None:

                filesystem_code = filesystem_metadata["filesystem"]["code"]
                filesystem_offset = filesystem_metadata["partition"]["start_sector"]

                recovery_directory = (
                    EVIDENCE_DIR.parent / "recovered" / evidence_id
                )

                # ------------------------------------------
                # Phase 1: Filesystem-level deleted recovery
                # ------------------------------------------

                inode_recovered_offsets = set()

                for entry in filesystem_metadata["files"]:
                    if not entry.get("deleted"):
                        continue

                    if entry.get("type") != "file":
                        continue

                    inode = entry.get("inode")

                    if inode is None:
                        continue

                    filename = Path(
                        entry.get("name", f"inode_{inode}")
                    ).name

                    output_path = (
                        recovery_directory / f"{inode}_{filename}"
                    )

                    try:
                        recovery_result = recover_deleted_file(
                            image_path=raw_image,
                            filesystem_code=filesystem_code,
                            offset=filesystem_offset,
                            inode=inode,
                            output_path=output_path,
                            original_filename=filename,
                        )

                        recovery_result["filename"] = filename
                        recovery_results.append(recovery_result)

                        record_custody_event(
                            evidence_id=evidence_id,
                            action="DELETED_FILE_RECOVERY",
                            description=(
                                "Deleted file recovery attempt completed."
                            ),
                            details={
                                "inode": recovery_result.get("inode"),
                                "filename": filename,
                                "recovered": recovery_result.get("recovered"),
                                "recovery_status": recovery_result.get(
                                    "recovery_status"
                                ),
                                "sha256": recovery_result.get("sha256"),
                                "method": recovery_result.get("method"),
                                "size": recovery_result.get("size"),
                            },
                        )

                    except subprocess.CalledProcessError as error:
                        stderr = (
                            error.stderr.decode(
                                "utf-8",
                                errors="ignore",
                            )
                            if error.stderr
                            else "icat failed"
                        )

                        recovery_result = {
                            "method": "inode_recovery",
                            "inode": inode,
                            "filename": filename,
                            "output_path": str(output_path),
                            "size": 0,
                            "sha256": None,
                            "recovered": False,
                            "recovery_status": "FAILED",
                            "error": stderr,
                            "validation": None,
                        }

                        recovery_results.append(recovery_result)

                        record_custody_event(
                            evidence_id=evidence_id,
                            action="DELETED_FILE_RECOVERY",
                            description=(
                                "Deleted file recovery attempt failed."
                            ),
                            details={
                                "inode": inode,
                                "filename": filename,
                                "recovered": False,
                                "recovery_status": "FAILED",
                                "error": stderr,
                            },
                        )

                # ------------------------------------------
                # Phase 2: Raw H.264 carving
                # ------------------------------------------

                carving_attempted = False

                try:
                    carving_attempted = True

                    carving_output = carve_h264_candidates(
                        image_path=raw_image,
                        output_directory=recovery_directory / "carved",
                    )

                    carving_results = carving_output["results"]
                    carving_failures = carving_output["failures"]

                    for failure in carving_failures:
                        record_custody_event(
                            evidence_id=evidence_id,
                            action="RAW_H264_CARVING",
                            description=(
                                "Raw H.264 carving candidate rejected."
                            ),
                            details={
                                "image_offset": failure.get("image_offset"),
                                "reason": failure.get("reason"),
                                "detail": failure.get("detail"),
                                "source": "raw_h264_carving",
                            },
                        )

                    for carving_result in carving_results:
                        carving_result["filename"] = Path(
                            carving_result["output_path"]
                        ).name

                        carving_result["source"] = "raw_h264_carving"
                        carving_result["recovery_status"] = (
                            "VALIDATED_CANDIDATE"
                        )

                        is_duplicate = False

                        for existing in recovery_results:
                            if not existing.get("recovered"):
                                continue

                            existing_sha = existing.get("sha256")
                            carving_sha = carving_result.get("sha256")

                            if (
                                existing_sha
                                and carving_sha
                                and existing_sha == carving_sha
                            ):
                                is_duplicate = True
                                break

                            existing_path = existing.get("output_path")
                            if existing_path and Path(existing_path).exists():
                                existing_offset = None
                                carving_offset = carving_result.get(
                                    "image_offset"
                                )

                                if (
                                    carving_offset is not None
                                    and existing.get("method") == "inode_recovery"
                                ):
                                    carving_size = carving_result.get("size", 0)
                                    existing_size = existing.get("size", 0)

                                    if (
                                        carving_size > 0
                                        and existing_size > 0
                                        and carving_size == existing_size
                                    ):
                                        is_duplicate = True
                                        break

                        if is_duplicate:
                            carving_result["recovery_status"] = (
                                "DUPLICATE_OF_EXISTING"
                            )
                            carving_result["duplicate_note"] = (
                                "Raw carving produced content identical to "
                                "an existing inode recovery result."
                            )

                        recovery_results.append(carving_result)

                        record_custody_event(
                            evidence_id=evidence_id,
                            action="RAW_H264_CARVING",
                            description=(
                                "Raw H.264 carving candidate recovered and "
                                "validated from forensic image."
                            ),
                            details={
                                "filename": carving_result.get("filename"),
                                "image_offset": carving_result.get(
                                    "image_offset"
                                ),
                                "size": carving_result.get("size"),
                                "sha256": carving_result.get("sha256"),
                                "recovery_status": carving_result.get(
                                    "recovery_status"
                                ),
                                "boundary": carving_result.get("boundary"),
                                "source": "raw_h264_carving",
                            },
                        )

                except (
                    FileNotFoundError,
                    ValueError,
                    OSError,
                ) as error:
                    carving_failures.append({
                        "reason": "carving_pipeline_error",
                        "detail": str(error),
                    })

                    record_custody_event(
                        evidence_id=evidence_id,
                        action="RAW_H264_CARVING",
                        description=(
                            "Raw H.264 carving was attempted but no "
                            "validated candidate was recovered."
                        ),
                        details={
                            "attempted": True,
                            "recovered": False,
                            "error": str(error),
                            "source": "raw_h264_carving",
                        },
                    )

            # -------------------------------------------------
            # Vendor detection
            # -------------------------------------------------

            try:
                vendor_result = detect_vendor(
                    raw_image,
                    "forensic-image/raw",
                    filesystem_metadata,
                )

                vendor_analysis = {
                    "vendor": vendor_result.vendor,
                    "confidence": vendor_result.confidence,
                    "detection_method": vendor_result.detection_method,
                    "evidence": vendor_result.evidence,
                }
            except Exception as error:
                analysis_errors.append({
                    "stage": "vendor_detection",
                    "error": str(error),
                })

            # -------------------------------------------------
            # Parser discovery and parsing
            # -------------------------------------------------

            loaded_plugins = []
            plugin_errors = []

            try:
                parser_registry = create_default_registry()

                plugins_directory = (
                    Path(__file__).resolve().parents[2] / "plugins"
                )

                plugin_discovery = parser_registry.discover_plugins(
                    plugins_directory
                )

                loaded_plugins = plugin_discovery["loaded_plugins"]
                plugin_errors = plugin_discovery["plugin_errors"]

                parser = parser_registry.find_parser(
                    raw_image,
                    filesystem_metadata,
                )

                if parser:
                    parsed_evidence = parser.parse(
                        raw_image,
                        filesystem_metadata,
                    )

                    timeline = build_timeline(
                        parsed_evidence.recordings
                    )

                    cross_camera_correlations = correlate_recordings(
                        parsed_evidence.recordings
                    )

                    parser_result = {
                        "status": "parsed",
                        "parser": parser.__class__.__name__,
                        "vendor": parser.vendor_name,
                        "dvr_evidence": {
                            "vendor": parsed_evidence.vendor,
                            "model": parsed_evidence.model,
                            "firmware": parsed_evidence.firmware,
                            "cameras": [
                                {
                                    "camera_id": camera.camera_id,
                                    "name": camera.name,
                                    "source": camera.source,
                                }
                                for camera in parsed_evidence.cameras
                            ],
                            "recordings": [
                                {
                                    "filename": recording.filename,
                                    "camera_id": recording.camera_id,
                                    "start_time": recording.start_time,
                                    "end_time": recording.end_time,
                                    "format": recording.format,
                                    "deleted": recording.deleted,
                                }
                                for recording in parsed_evidence.recordings
                            ],
                            "metadata": parsed_evidence.metadata,
                            "parser": parsed_evidence.parser,
                        },
                    }

                else:
                    parser_result = {
                        "status": "no_matching_parser",
                        "parser": None,
                        "vendor": (
                            vendor_analysis["vendor"]
                            if vendor_analysis
                            else "unknown"
                        ),
                    }

            except Exception as error:
                analysis_errors.append({
                    "stage": "vendor_parsing",
                    "error": str(error),
                })

                if parser_result is None:
                    parser_result = {
                        "status": "parsing_error",
                        "parser": None,
                        "vendor": (
                            vendor_analysis["vendor"]
                            if vendor_analysis
                            else "unknown"
                        ),
                        "error": str(error),
                    }

            parser_result["loaded_plugins"] = loaded_plugins
            parser_result["plugin_errors"] = plugin_errors

            raw_carving_summary = {
                "attempted": carving_attempted if 'carving_attempted' in dir() else False,
                "candidates_recovered": sum(
                    1
                    for r in recovery_results
                    if r.get("method") == "contiguous_carving"
                    and r.get("source") == "raw_h264_carving"
                    and r.get("recovered")
                ),
                "candidates_rejected": len(carving_failures),
                "failures": carving_failures,
            }

            parser_result["raw_carving"] = raw_carving_summary

        finally:
            if temp_directory is not None:
                temp_directory.cleanup()

    recovery_summary = _build_recovery_summary(
        recovery_results, classification
    )

    record_custody_event(
        evidence_id=evidence_id,
        action="ANALYSIS_COMPLETED",
        description="Evidence processing and analysis completed.",
        details={
            "classification": classification,
            "vendor": (
                vendor_analysis["vendor"]
                if vendor_analysis
                else vendor_result.vendor
            ),
            "sha256": sha256,
            "recovery_attempted": classification == "FORENSIC_IMAGE",
            "recovered_files": recovery_summary["recovered"],
            "analysis_errors": len(analysis_errors),
        },
    )

    created_at = datetime.now(timezone.utc).isoformat()

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO evidence (
            evidence_id,
            filename,
            stored_path,
            file_size,
            sha256,
            file_type,
            vendor,
            vendor_confidence,
            detection_method,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence_id,
            original_name,
            str(destination),
            destination.stat().st_size,
            sha256,
            file_type,
            vendor_analysis["vendor"]
            if vendor_analysis
            else vendor_result.vendor,
            vendor_analysis["confidence"]
            if vendor_analysis
            else vendor_result.confidence,
            vendor_analysis["detection_method"]
            if vendor_analysis
            else vendor_result.detection_method,
            "IMPORTED",
            created_at,
        ),
    )

    connection.commit()
    connection.close()

    custody_history = get_custody_history(evidence_id)

    evidence_data = {
        "evidence_id": evidence_id,
        "filename": original_name,
        "stored_as": destination.name,
        "size": destination.stat().st_size,
        "sha256": sha256,
        "file_type": file_type,
        "classification": classification,
        "vendor": (
            vendor_analysis["vendor"]
            if vendor_analysis
            else vendor_result.vendor
        ),
        "vendor_confidence": (
            vendor_analysis["confidence"]
            if vendor_analysis
            else vendor_result.confidence
        ),
        "detection_method": (
            vendor_analysis["detection_method"]
            if vendor_analysis
            else vendor_result.detection_method
        ),
        "working_copy": str(working_copy),
        "integrity_verified": integrity_validation["valid"],
        "integrity_validation": integrity_validation,
        "media_metadata": media_metadata,
        "forensic_image_metadata": forensic_image_metadata,
        "filesystem_metadata": filesystem_metadata,
        "dvr_structure": dvr_structure,
        "vendor_analysis": vendor_analysis,
        "parser_result": parser_result,
        "timeline": timeline,
        "cross_camera_correlations": cross_camera_correlations,
        "recovery_results": recovery_results,
        "recovery_summary": recovery_summary,
        "chain_of_custody": custody_history,
        "analysis_errors": analysis_errors,
        "status": "IMPORTED",
    }

    report_result = generate_forensic_report(
        evidence_data
    )

    record_custody_event(
        evidence_id=evidence_id,
        action="REPORT_GENERATED",
        description="Standardized forensic report generated.",
        details={
            "pdf_path": report_result["pdf_path"],
            "json_path": report_result["json_path"],
            "generated_at": report_result["generated_at"],
        },
    )

    return {
        "evidence_id": evidence_id,
        "filename": original_name,
        "stored_as": destination.name,
        "size": destination.stat().st_size,
        "sha256": sha256,
        "file_type": file_type,
        "classification": classification,
        "vendor": (
            vendor_analysis["vendor"]
            if vendor_analysis
            else vendor_result.vendor
        ),
        "vendor_confidence": (
            vendor_analysis["confidence"]
            if vendor_analysis
            else vendor_result.confidence
        ),
        "detection_method": (
            vendor_analysis["detection_method"]
            if vendor_analysis
            else vendor_result.detection_method
        ),
        "working_copy": str(working_copy),
        "integrity_verified": integrity_validation["valid"],
        "integrity_validation": integrity_validation,
        "media_metadata": media_metadata,
        "forensic_image_metadata": forensic_image_metadata,
        "filesystem_metadata": filesystem_metadata,
        "dvr_structure": dvr_structure,
        "vendor_analysis": vendor_analysis,
        "parser_result": parser_result,
        "timeline": timeline,
        "cross_camera_correlations": cross_camera_correlations,
        "recovery_results": recovery_results,
        "recovery_summary": recovery_summary,
        "chain_of_custody": get_custody_history(evidence_id),
        "analysis_errors": analysis_errors,
        "report": report_result,
        "status": "IMPORTED",
    }


def _build_recovery_summary(
    recovery_results: list[dict],
    classification: str,
) -> dict:
    attempted = classification == "FORENSIC_IMAGE"

    unique_recovered = 0
    unique_inode = 0
    unique_carving = 0
    duplicates_suppressed = 0
    failed = 0

    seen_sha256s: set[str] = set()

    for result in recovery_results:
        status = result.get("recovery_status")

        if status == "DUPLICATE_OF_EXISTING":
            duplicates_suppressed += 1
            continue

        if not result.get("recovered"):
            if status in {"FAILED", "EMPTY"}:
                failed += 1
            continue

        sha = result.get("sha256")

        if sha and sha in seen_sha256s:
            duplicates_suppressed += 1
            continue

        if sha:
            seen_sha256s.add(sha)

        unique_recovered += 1

        if result.get("method") == "inode_recovery":
            unique_inode += 1
        elif (
            result.get("method") == "contiguous_carving"
            and result.get("source") == "raw_h264_carving"
        ):
            unique_carving += 1

    return {
        "attempted": attempted,
        "recovered": unique_recovered,
        "failed": failed,
        "inode_recovery": unique_inode,
        "raw_h264_carving": unique_carving,
        "duplicates_suppressed": duplicates_suppressed,
    }


@router.get("/{evidence_id}/chain-of-custody")
def get_evidence_chain_of_custody(evidence_id: str):
    return {
        "evidence_id": evidence_id,
        "chain_of_custody": get_custody_history(evidence_id),
    }
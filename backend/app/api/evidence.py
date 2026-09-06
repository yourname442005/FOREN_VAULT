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
    parser_result = None
    timeline = []
    cross_camera_correlations = []

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
            filesystem_metadata = analyze_filesystem(raw_image)

            dvr_structure = detect_dvr_structure(
                filesystem_metadata["files"]
            )

            record_custody_event(
                evidence_id=evidence_id,
                action="FILESYSTEM_ANALYZED",
                description="Filesystem analysis completed.",
                details={
                    "filesystem": filesystem_metadata["filesystem"],
                    "partition": filesystem_metadata["partition"],
                    "file_count": len(filesystem_metadata["files"]),
                },
            )

            filesystem_code = filesystem_metadata["filesystem"]["code"]
            filesystem_offset = filesystem_metadata["partition"]["start_sector"]

            recovery_directory = (
                EVIDENCE_DIR.parent / "recovered" / evidence_id
            )

            # ---------------------------------------------------------
            # Phase 1: Filesystem-level deleted file recovery
            # ---------------------------------------------------------

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
                            "sha256": recovery_result.get("sha256"),
                            "method": recovery_result.get("method"),
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
                        "error": stderr,
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
                            "error": stderr,
                        },
                    )

            # ---------------------------------------------------------
            # Phase 2: Raw H.264 carving
            #
            # Secondary recovery technique for recordings that no
            # longer exist in filesystem metadata.
            # ---------------------------------------------------------

            inode_recovered_media = any(
                result.get("recovered")
                and result.get("method") == "inode_recovery"
                for result in recovery_results
            )

            carving_attempted = False
            carving_results = []
            carving_error = None

            try:
                carving_attempted = True

                carving_results = carve_h264_candidates(
                    image_path=raw_image,
                    output_directory=recovery_directory / "carved",
                )

                for carving_result in carving_results:
                    carving_result["filename"] = Path(
                        carving_result["output_path"]
                    ).name

                    carving_result["source"] = "raw_h264_carving"

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
                            "validation": carving_result.get("validation"),
                            "boundary": carving_result.get("boundary"),
                            "recovery_status": carving_result.get(
                                "recovery_status"
                            ),
                            "source": "raw_h264_carving",
                        },
                    )

            except (FileNotFoundError, ValueError, OSError) as error:
                carving_error = str(error)

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
                        "error": carving_error,
                        "source": "raw_h264_carving",
                    },
                )

            raw_carving_summary = {
                "attempted": carving_attempted,
                "candidates_recovered": len(carving_results),
                "error": carving_error,
                "inode_recovery_found_media": inode_recovered_media,
            }

            # ---------------------------------------------------------
            # Vendor detection and DVR parsing
            # ---------------------------------------------------------

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
                    "vendor": vendor_result.vendor,
                }

            parser_result["loaded_plugins"] = loaded_plugins
            parser_result["plugin_errors"] = plugin_errors
            parser_result["raw_carving"] = raw_carving_summary

        finally:
            if temp_directory is not None:
                temp_directory.cleanup()

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
            "recovered_files": sum(
                1
                for result in recovery_results
                if result.get("recovered")
            ),
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
        "recovery_summary": {
            "attempted": classification == "FORENSIC_IMAGE",
            "recovered": sum(
                1
                for result in recovery_results
                if result.get("recovered")
            ),
            "failed": sum(
                1
                for result in recovery_results
                if not result.get("recovered")
            ),
            "inode_recovery": sum(
                1
                for result in recovery_results
                if result.get("method") == "inode_recovery"
                and result.get("recovered")
            ),
            "raw_h264_carving": sum(
                1
                for result in recovery_results
                if result.get("method") == "contiguous_carving"
                and result.get("source") == "raw_h264_carving"
                and result.get("recovered")
            ),
        },
        "chain_of_custody": custody_history,
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
        "recovery_summary": {
            "attempted": classification == "FORENSIC_IMAGE",
            "recovered": sum(
                1
                for result in recovery_results
                if result.get("recovered")
            ),
            "failed": sum(
                1
                for result in recovery_results
                if not result.get("recovered")
            ),
            "inode_recovery": sum(
                1
                for result in recovery_results
                if result.get("method") == "inode_recovery"
                and result.get("recovered")
            ),
            "raw_h264_carving": sum(
                1
                for result in recovery_results
                if result.get("method") == "contiguous_carving"
                and result.get("source") == "raw_h264_carving"
                and result.get("recovered")
            ),
        },
        "chain_of_custody": get_custody_history(evidence_id),
        "report": report_result,
        "status": "IMPORTED",
    }


@router.get("/{evidence_id}/chain-of-custody")
def get_evidence_chain_of_custody(evidence_id: str):
    return {
        "evidence_id": evidence_id,
        "chain_of_custody": get_custody_history(evidence_id),
    }
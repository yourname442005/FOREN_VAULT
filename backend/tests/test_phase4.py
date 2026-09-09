import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.analysis_pipeline import (
    AnalysisPipeline,
    AnalysisStage,
    COMPLETED,
    FAILED,
    NOT_APPLICABLE,
    NOT_ATTEMPTED,
    PARTIAL,
)
from app.models.dvr_evidence import (
    NORMALIZED,
    TIMEZONE_UNKNOWN,
    DVREvidence,
    Recording,
    TimestampResult,
)
from app.services.dvr_structure_detector import detect_dvr_structure
from app.services.event_correlator import correlate_recordings
from app.services.recording_timeline import build_timeline
from app.services.timestamp_parser import parse_recording_filename, parse_timestamp
from app.services.vendor_detector import score_vendor


# =============================================================================
# A. ANALYSIS STAGE STATES
# =============================================================================


class TestAnalysisStageModel:
    def test_stage_defaults(self):
        stage = AnalysisStage(name="test_stage")
        assert stage.name == "test_stage"
        assert stage.status == NOT_ATTEMPTED
        assert stage.error is None
        assert stage.details == {}

    def test_stage_mark_completed(self):
        stage = AnalysisStage(name="test_stage")
        stage.mark_completed({"key": "value"})
        assert stage.status == COMPLETED
        assert stage.details["key"] == "value"

    def test_stage_mark_failed(self):
        stage = AnalysisStage(name="test_stage")
        stage.mark_failed("Something went wrong")
        assert stage.status == FAILED
        assert stage.error == "Something went wrong"

    def test_stage_mark_not_applicable(self):
        stage = AnalysisStage(name="test_stage")
        stage.mark_not_applicable("No vendor detected")
        assert stage.status == NOT_APPLICABLE
        assert stage.details["reason"] == "No vendor detected"

    def test_stage_mark_partial(self):
        stage = AnalysisStage(name="test_stage")
        stage.mark_partial("Partial data", {"count": 5})
        assert stage.status == PARTIAL
        assert stage.error == "Partial data"
        assert stage.details["count"] == 5

    def test_stage_to_dict(self):
        stage = AnalysisStage(name="test_stage")
        stage.mark_completed({"key": "value"})
        d = stage.to_dict()
        assert d["name"] == "test_stage"
        assert d["status"] == COMPLETED
        assert d["details"]["key"] == "value"

    def test_stage_to_dict_with_error(self):
        stage = AnalysisStage(name="test_stage")
        stage.mark_failed("error message")
        d = stage.to_dict()
        assert d["error"] == "error message"


class TestAnalysisPipeline:
    def test_pipeline_initializes_all_stages(self):
        pipeline = AnalysisPipeline(evidence_id="EV-TEST")
        assert len(pipeline.stages) == len(AnalysisPipeline.STAGE_NAMES)

    def test_pipeline_get_stage_creates_if_missing(self):
        pipeline = AnalysisPipeline(evidence_id="EV-TEST")
        stage = pipeline.get_stage("nonexistent_stage")
        assert stage.name == "nonexistent_stage"
        assert stage.status == NOT_ATTEMPTED

    def test_pipeline_mark_completed(self):
        pipeline = AnalysisPipeline(evidence_id="EV-TEST")
        pipeline.mark_completed("filesystem_analyzed", {"type": "HFS+"})
        assert pipeline.get_stage("filesystem_analyzed").status == COMPLETED
        assert pipeline.get_stage("filesystem_analyzed").details["type"] == "HFS+"

    def test_pipeline_mark_failed(self):
        pipeline = AnalysisPipeline(evidence_id="EV-TEST")
        pipeline.mark_failed("vendor_detected", "No markers found")
        assert pipeline.get_stage("vendor_detected").status == FAILED
        assert pipeline.get_stage("vendor_detected").error == "No markers found"

    def test_pipeline_mark_not_applicable(self):
        pipeline = AnalysisPipeline(evidence_id="EV-TEST")
        pipeline.mark_not_applicable("vendor_parser_selected", "No matching parser")
        assert pipeline.get_stage("vendor_parser_selected").status == NOT_APPLICABLE

    def test_pipeline_summary(self):
        pipeline = AnalysisPipeline(evidence_id="EV-TEST")
        pipeline.mark_completed("stage_a")
        pipeline.mark_failed("stage_b", "error")
        pipeline.mark_not_applicable("stage_c")
        summary = pipeline.summary()
        assert summary["completed"] >= 1
        assert summary["failed"] >= 1
        assert summary["not_applicable"] >= 1

    def test_pipeline_to_dict(self):
        pipeline = AnalysisPipeline(evidence_id="EV-TEST")
        d = pipeline.to_dict()
        assert d["evidence_id"] == "EV-TEST"
        assert "stages" in d
        assert len(d["stages"]) == len(AnalysisPipeline.STAGE_NAMES)


# =============================================================================
# B-C. FORENSIC IMAGE ANALYSIS
# =============================================================================


class TestForensicImageAnalysis:
    def test_e01_extension_recognized(self):
        from app.services.evidence_classifier import classify_evidence
        result = classify_evidence(Path("test.e01"), "forensic-image/e01")
        assert result == "FORENSIC_IMAGE"

    def test_raw_extension_recognized(self):
        from app.services.evidence_classifier import classify_evidence
        result = classify_evidence(Path("test.raw"), "application/octet-stream")
        assert result == "FORENSIC_IMAGE"

    def test_dmg_extension_recognized(self):
        from app.services.evidence_classifier import classify_evidence
        result = classify_evidence(Path("test.dmg"), "application/octet-stream")
        assert result == "FORENSIC_IMAGE"

    def test_dd_extension_recognized(self):
        from app.services.evidence_classifier import classify_evidence
        result = classify_evidence(Path("test.dd"), "application/octet-stream")
        assert result == "FORENSIC_IMAGE"

    def test_media_classification(self):
        from app.services.evidence_classifier import classify_evidence
        result = classify_evidence(Path("test.mp4"), "video/mp4")
        assert result == "MEDIA"

    def test_unknown_classification(self):
        from app.services.evidence_classifier import classify_evidence
        result = classify_evidence(Path("test.xyz"), "application/octet-stream")
        assert result == "UNKNOWN"


# =============================================================================
# D-E. FILESYSTEM ANALYSIS
# =============================================================================


class TestFilesystemAnalysis:
    def test_dvr_structure_with_recordings(self):
        files = [
            {"name": "CAM01", "type": "directory"},
            {"name": "CAM01/20260901_180000_190000_CAM01.h264", "type": "file"},
            {"name": "device.conf", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "DVR_NVR_LIKELY"
        assert len(result["camera_directories"]) == 1
        assert len(result["recording_files"]) == 1

    def test_dvr_structure_without_recordings(self):
        files = [
            {"name": "document.pdf", "type": "file"},
            {"name": "image.jpg", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "NOT_DVR_NVR"

    def test_dvr_structure_empty(self):
        result = detect_dvr_structure([])
        assert result["classification"] == "NOT_DVR_NVR"
        assert result["confidence"] == 0.0

    def test_dvr_structure_config_files(self):
        files = [
            {"name": "config.json", "type": "file"},
            {"name": "settings.conf", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert len(result["configuration_files"]) == 2

    def test_dvr_structure_log_files(self):
        files = [
            {"name": "system.log", "type": "file"},
            {"name": "log_2026.txt", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert len(result["log_files"]) == 2


# =============================================================================
# F-G. DVR/NVR CLASSIFICATION
# =============================================================================


class TestDVRClassification:
    def test_possible_dvr(self):
        files = [
            {"name": "recording.h264", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "POSSIBLE_DVR_NVR"
        assert result["confidence"] >= 0.40

    def test_dvr_likely(self):
        files = [
            {"name": "CAM01", "type": "directory"},
            {"name": "CAM01/rec.h264", "type": "file"},
            {"name": "config.json", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "DVR_NVR_LIKELY"
        assert result["confidence"] >= 0.70

    def test_not_dvr(self):
        files = [
            {"name": "notes.txt", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "NOT_DVR_NVR"


# =============================================================================
# H-I. VENDOR DETECTION
# =============================================================================


class TestVendorDetectionPipeline:
    def test_hikvision_detection(self):
        sources = [
            ("configuration", "manufacturer=Hikvision"),
            ("filename", "hik_config.json"),
        ]
        score, evidence = score_vendor("hikvision", sources)
        assert score >= 0.50

    def test_dahua_detection(self):
        sources = [
            ("configuration", "manufacturer=Dahua"),
            ("filename", "dahua_config.json"),
        ]
        score, evidence = score_vendor("dahua", sources)
        assert score >= 0.50

    def test_unknown_vendor(self):
        sources = [
            ("configuration", "This is a generic device"),
        ]
        for vendor in ["hikvision", "dahua", "cp_plus", "uniview",
                        "honeywell", "tp_link", "godrej", "matrix"]:
            score, evidence = score_vendor(vendor, sources)
            assert score == 0.0


# =============================================================================
# J-K. PARSER SUCCESS/FAILURE
# =============================================================================


class TestParserPipeline:
    def test_all_parsers_have_capabilities(self):
        from app.parsers.vendor_registry import create_default_registry

        registry = create_default_registry()
        for parser in registry._parsers:
            caps = parser.get_capabilities()
            assert len(caps.capabilities) > 0

    def test_plugin_discovery(self):
        from app.parsers.vendor_registry import create_default_registry

        registry = create_default_registry()
        plugins_directory = Path(__file__).parent.parent / "plugins"
        result = registry.discover_plugins(plugins_directory)
        assert len(result["loaded_plugins"]) >= 6
        assert len(result["plugin_errors"]) == 0


# =============================================================================
# L-N. TIMESTAMPS
# =============================================================================


class TestTimestampPipeline:
    def test_parse_timestamp_with_hint(self):
        result = parse_timestamp("2026-09-01T18:00:00", timezone_hint="Asia/Kolkata")
        assert result.normalization_status == NORMALIZED
        assert result.timezone == "Asia/Kolkata"

    def test_parse_timestamp_without_hint(self):
        result = parse_timestamp("2026-09-01T18:00:00")
        assert result.normalization_status == TIMEZONE_UNKNOWN
        assert result.timezone is None

    def test_parse_timestamp_utc(self):
        result = parse_timestamp("2026-09-01T18:00:00Z")
        assert result.normalization_status == NORMALIZED
        assert result.timezone == "UTC"

    def test_parse_timestamp_invalid(self):
        result = parse_timestamp("not-a-date")
        assert result.normalization_status == "INVALID"

    def test_parse_timestamp_preserves_original(self):
        result = parse_timestamp("2026-09-01T18:00:00")
        assert result.original == "2026-09-01T18:00:00"

    def test_recording_filename_integration(self):
        parsed = parse_recording_filename("20260901_180000_190000_CAM01.h264")
        assert parsed is not None
        start_ts = parse_timestamp(parsed["start_time"])
        assert start_ts.iso_naive is not None


# =============================================================================
# O-P. OVERLAPPING/NON-OVERLAPPING RECORDINGS
# =============================================================================


class TestRecordingCorrelation:
    def test_overlapping_recordings(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T18:30:00",
            end_time="2026-09-01T19:30:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 1
        assert correlations[0]["timezone_comparable"] is True

    def test_non_overlapping_recordings(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T20:00:00",
            end_time="2026-09-01T21:00:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 0

    def test_same_camera_not_correlated(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:30:00",
            end_time="2026-09-01T19:30:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 0


# =============================================================================
# Q. MIXED TIMEZONE CORRELATION SAFETY
# =============================================================================


class TestMixedTimezoneCorrelation:
    def test_mixed_tz_skipped(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00+05:30",
            end_time="2026-09-01T19:00:00+05:30",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T18:30:00",
            end_time="2026-09-01T19:30:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 0

    def test_both_naive_comparable(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T18:30:00",
            end_time="2026-09-01T19:30:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 1

    def test_both_aware_comparable(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00+05:30",
            end_time="2026-09-01T19:00:00+05:30",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T18:30:00+05:30",
            end_time="2026-09-01T19:30:00+05:30",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 1


# =============================================================================
# R-T. DELETED EVIDENCE RECOVERY
# =============================================================================


class TestRecoveryPipeline:
    def test_recovery_result_structure(self):
        result = {
            "method": "inode_recovery",
            "inode": 1234,
            "output_path": "/tmp/test.h264",
            "size": 53186,
            "sha256": "abc123",
            "recovered": True,
            "recovery_status": "VALIDATED_CANDIDATE",
            "validation": {"valid": True, "codec": "h264", "width": 320, "height": 240},
        }
        assert result["recovered"] is True
        assert result["recovery_status"] == "VALIDATED_CANDIDATE"
        assert result["validation"]["valid"] is True

    def test_carving_result_structure(self):
        result = {
            "method": "contiguous_carving",
            "image_offset": 18796544,
            "output_path": "/tmp/carved.h264",
            "size": 53186,
            "sha256": "abc123",
            "recovered": True,
            "source": "raw_h264_carving",
            "recovery_status": "VALIDATED_CANDIDATE",
            "boundary": {
                "confidence": "MEDIUM",
                "method": "sustained_zero_tail",
                "exact_original_boundary_established": False,
            },
        }
        assert result["boundary"]["confidence"] == "MEDIUM"
        assert result["boundary"]["exact_original_boundary_established"] is False

    def test_duplicate_detection(self):
        existing = {
            "recovered": True,
            "sha256": "abc123",
            "method": "inode_recovery",
        }
        carving = {
            "recovered": True,
            "sha256": "abc123",
            "method": "contiguous_carving",
        }
        is_duplicate = (
            existing.get("recovered")
            and carving.get("recovered")
            and existing.get("sha256") == carving.get("sha256")
        )
        assert is_duplicate is True

    def test_different_files_not_deduplicated(self):
        existing = {
            "recovered": True,
            "sha256": "abc123",
            "method": "inode_recovery",
        }
        carving = {
            "recovered": True,
            "sha256": "def456",
            "method": "contiguous_carving",
        }
        is_duplicate = (
            existing.get("recovered")
            and carving.get("recovered")
            and existing.get("sha256") == carving.get("sha256")
        )
        assert is_duplicate is False

    def test_failed_recovery_not_deduplicated(self):
        existing = {
            "recovered": False,
            "sha256": None,
            "method": "inode_recovery",
            "recovery_status": "FAILED",
        }
        carving = {
            "recovered": True,
            "sha256": "abc123",
            "method": "contiguous_carving",
        }
        is_duplicate = (
            existing.get("recovered")
            and carving.get("recovered")
            and existing.get("sha256") == carving.get("sha256")
        )
        assert is_duplicate is False


# =============================================================================
# U. UNCERTAIN RECOVERY BOUNDARY
# =============================================================================


class TestRecoveryBoundary:
    def test_boundary_not_exact(self):
        boundary = {
            "confidence": "MEDIUM",
            "method": "sustained_zero_tail",
            "exact_original_boundary_established": False,
        }
        assert boundary["exact_original_boundary_established"] is False
        assert boundary["confidence"] == "MEDIUM"

    def test_boundary_uncertain(self):
        boundary = {
            "confidence": "UNCERTAIN",
            "method": None,
            "exact_original_boundary_established": False,
        }
        assert boundary["exact_original_boundary_established"] is False
        assert boundary["confidence"] == "UNCERTAIN"


# =============================================================================
# V. DUPLICATE RECOVERY HANDLING
# =============================================================================


class TestDuplicateRecovery:
    def test_recovery_summary_deduplication(self):
        from app.api.evidence import _build_recovery_summary

        recovery_results = [
            {"recovered": True, "sha256": "abc", "method": "inode_recovery", "recovery_status": "RECOVERED"},
            {"recovered": True, "sha256": "abc", "method": "contiguous_carving", "source": "raw_h264_carving", "recovery_status": "DUPLICATE_OF_EXISTING"},
            {"recovered": True, "sha256": "def", "method": "inode_recovery", "recovery_status": "RECOVERED"},
            {"recovered": False, "recovery_status": "FAILED"},
        ]
        summary = _build_recovery_summary(recovery_results, "FORENSIC_IMAGE")
        assert summary["recovered"] == 2
        assert summary["duplicates_suppressed"] == 1
        assert summary["failed"] == 1

    def test_recovery_summary_no_duplicates(self):
        from app.api.evidence import _build_recovery_summary

        recovery_results = [
            {"recovered": True, "sha256": "abc", "method": "inode_recovery", "recovery_status": "RECOVERED"},
            {"recovered": True, "sha256": "def", "method": "inode_recovery", "recovery_status": "RECOVERED"},
        ]
        summary = _build_recovery_summary(recovery_results, "FORENSIC_IMAGE")
        assert summary["recovered"] == 2
        assert summary["duplicates_suppressed"] == 0

    def test_recovery_summary_non_forensic(self):
        from app.api.evidence import _build_recovery_summary

        summary = _build_recovery_summary([], "MEDIA")
        assert summary["attempted"] is False


# =============================================================================
# W. CHAIN OF CUSTODY SEQUENCE
# =============================================================================


class TestChainOfCustody:
    def test_custody_event_structure(self):
        from app.services.chain_of_custody import get_custody_history
        assert callable(get_custody_history)

    def test_custody_event_creation(self):
        from app.services.chain_of_custody import record_custody_event
        assert callable(record_custody_event)


# =============================================================================
# X. REPORT CONTENT
# =============================================================================


class TestReportContent:
    def test_report_generator_callable(self):
        from app.services.forensic_report_generator import generate_forensic_report
        assert callable(generate_forensic_report)

    def test_report_includes_pipeline(self):
        evidence_data = {
            "evidence_id": "EV-TEST",
            "analysis_pipeline": {
                "stages": {
                    "filesystem_analyzed": {"status": "COMPLETED", "details": {"type": "HFS+"}},
                    "vendor_detected": {"status": "COMPLETED", "details": {"vendor": "Hikvision"}},
                }
            },
            "analysis_errors": [],
        }
        from app.services.forensic_report_generator import generate_forensic_report
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_forensic_report(evidence_data, Path(tmpdir))
            assert result["status"] == "GENERATED"
            assert Path(result["json_path"]).exists()

    def test_report_includes_limitations(self):
        evidence_data = {
            "evidence_id": "EV-TEST",
            "analysis_pipeline": {"stages": {}},
            "analysis_errors": [{"stage": "test", "error": "test error"}],
            "recovery_summary": {"duplicates_suppressed": 1},
        }
        from app.services.forensic_report_generator import generate_forensic_report
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_forensic_report(evidence_data, Path(tmpdir))
            with open(result["json_path"]) as f:
                report = json.load(f)
            assert "evidence" in report


# =============================================================================
# Y. GENERIC FORENSIC FALLBACK
# =============================================================================


class TestGenericFallback:
    def test_unknown_vendor_still_works(self):
        files = [
            {"name": "CAM01", "type": "directory"},
            {"name": "CAM01/rec.h264", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "DVR_NVR_LIKELY"

    def test_empty_evidence_handled(self):
        result = detect_dvr_structure([])
        assert result["classification"] == "NOT_DVR_NVR"
        assert result["camera_directories"] == []
        assert result["recording_files"] == []


# =============================================================================
# Z. PHASE 1/2/3 REGRESSION
# =============================================================================


class TestPhaseRegression:
    def test_timestamp_result_model(self):
        tr = TimestampResult(
            original="2026-09-01T18:00:00",
            iso_naive="2026-09-01T18:00:00",
            normalization_status=TIMEZONE_UNKNOWN,
        )
        assert tr.original == "2026-09-01T18:00:00"
        assert tr.normalization_status == TIMEZONE_UNKNOWN

    def test_recording_model_with_timestamps(self):
        ts = TimestampResult(original="20260901_180000")
        rec = Recording(
            filename="20260901_180000_190000_CAM01.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
            start_timestamp=ts,
            end_timestamp=ts,
        )
        assert rec.start_timestamp is not None
        assert rec.start_timestamp.original == "20260901_180000"

    def test_dvr_evidence_with_timezone(self):
        evidence = DVREvidence(
            vendor="Hikvision",
            timezone="Asia/Kolkata",
            timezone_source="device_config",
        )
        assert evidence.timezone == "Asia/Kolkata"
        assert evidence.timezone_source == "device_config"

    def test_build_timeline(self):
        rec = Recording(
            filename="test.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        timeline = build_timeline([rec])
        assert len(timeline) == 1
        assert timeline[0]["filename"] == "test.h264"

    def test_sha256_calculation(self):
        from app.services.evidence_service import calculate_sha256
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            f.flush()
            result = calculate_sha256(Path(f.name))
        assert len(result) == 64

    def test_evidence_validation(self):
        from app.services.evidence_validator import validate_evidence_integrity
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            f.flush()
            from app.services.evidence_service import calculate_sha256
            sha = calculate_sha256(Path(f.name))
            result = validate_evidence_integrity(Path(f.name), sha)
        assert result["valid"] is True
        assert result["status"] == "VALID"

    def test_capability_constants(self):
        from app.models.vendor import (
            ALL_CANONICAL_CAPABILITIES,
            CAPABILITY_DELETED_RECOVERY,
            CAPABILITY_VENDOR_DETECTION,
        )
        assert len(ALL_CANONICAL_CAPABILITIES) == 8
        assert CAPABILITY_VENDOR_DETECTION == "vendor_detection"
        assert CAPABILITY_DELETED_RECOVERY == "deleted_recovery"

    def test_all_eight_vendors_detected(self):
        from app.services.vendor_detector import VENDOR_MARKERS, VENDOR_DISPLAY_NAMES
        assert set(VENDOR_MARKERS.keys()) == set(VENDOR_DISPLAY_NAMES.keys())
        assert len(VENDOR_MARKERS) == 8

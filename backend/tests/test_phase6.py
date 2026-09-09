import pytest
from pathlib import Path

from app.models.ai_observation import (
    AIAnalysisResult,
    AICapabilitiesResult,
    AIObservation,
    BoundingBox,
    TimeMapping,
    MOTION,
    OBJECT_DETECTION,
    FACE_DETECTION,
    NOT_REQUESTED,
    NOT_AVAILABLE,
    COMPLETED,
    FAILED,
    VALID_ANALYSIS_TYPES,
)
from app.services.ai_providers import (
    MotionProvider,
    ObjectDetectionProvider,
    FaceDetectionProvider,
    get_provider,
    get_available_providers,
    PROVIDER_REGISTRY,
)
from app.services.ai_analysis import (
    get_capabilities,
    analyze_video_file,
    filter_observations,
    _normalize_observation,
)


# =============================================================================
# A. AI CAPABILITY DISCOVERY
# =============================================================================


class TestAICapabilityDiscovery:
    def test_get_capabilities(self):
        caps = get_capabilities()
        assert isinstance(caps, AICapabilitiesResult)
        assert caps.motion is not None
        assert caps.object_detection is not None
        assert caps.face_detection is not None

    def test_motion_available(self):
        caps = get_capabilities()
        assert caps.motion.available is True
        assert caps.motion.provider == "opencv_motion"

    def test_object_detection_available(self):
        caps = get_capabilities()
        assert caps.object_detection.available is True
        assert caps.object_detection.provider == "nanodet"

    def test_face_detection_available(self):
        caps = get_capabilities()
        assert caps.face_detection.available is True
        assert caps.face_detection.provider == "yunet"

    def test_capabilities_to_dict(self):
        caps = get_capabilities()
        d = caps.to_dict()
        assert "motion" in d
        assert "object_detection" in d
        assert "face_detection" in d
        assert d["motion"]["available"] is True
        assert d["object_detection"]["available"] is True
        assert d["face_detection"]["available"] is True


# =============================================================================
# B. PROVIDER UNAVAILABLE BEHAVIOR
# =============================================================================


class TestProviderAvailability:
    def test_object_detector_available(self):
        provider = ObjectDetectionProvider()
        assert provider.is_available is True

    def test_face_detector_available(self):
        provider = FaceDetectionProvider()
        assert provider.is_available is True


# =============================================================================
# C. PROVIDER SELECTION
# =============================================================================


class TestProviderSelection:
    def test_get_motion_provider(self):
        provider = get_provider("motion")
        assert provider is not None
        assert isinstance(provider, MotionProvider)

    def test_get_object_provider(self):
        provider = get_provider("object_detection")
        assert provider is not None
        assert isinstance(provider, ObjectDetectionProvider)

    def test_get_face_provider(self):
        provider = get_provider("face_detection")
        assert provider is not None
        assert isinstance(provider, FaceDetectionProvider)

    def test_get_unknown_provider(self):
        provider = get_provider("unknown_type")
        assert provider is None

    def test_provider_registry_complete(self):
        assert "motion" in PROVIDER_REGISTRY
        assert "object_detection" in PROVIDER_REGISTRY
        assert "face_detection" in PROVIDER_REGISTRY

    def test_available_providers(self):
        available = get_available_providers()
        assert available["motion"] is True
        assert available["object_detection"] is True
        assert available["face_detection"] is True


# =============================================================================
# D. OBSERVATION MODEL VALIDATION
# =============================================================================


class TestObservationModel:
    def test_observation_creation(self):
        obs = AIObservation(
            observation_id="OBS-001",
            evidence_id="EV-001",
            camera_id="CAM01",
            analysis_type="motion",
            label="motion_detected",
            confidence=0.85,
        )
        assert obs.observation_id == "OBS-001"
        assert obs.evidence_id == "EV-001"
        assert obs.confidence == 0.85

    def test_observation_to_dict(self):
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            label="motion_detected",
            confidence=0.85,
        )
        d = obs.to_dict()
        assert d["observation_id"] == "OBS-001"
        assert d["confidence"] == 0.85

    def test_observation_with_time_mapping(self):
        tm = TimeMapping(
            media_offset_seconds=13.5,
            mapping_status="MAPPED",
        )
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            time_mapping=tm,
        )
        d = obs.to_dict()
        assert d["time_mapping"]["media_offset_seconds"] == 13.5
        assert d["time_mapping"]["mapping_status"] == "MAPPED"

    def test_observation_with_bounding_box(self):
        bb = BoundingBox(x=10, y=20, width=100, height=200)
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="object_detection",
            bounding_box=bb,
        )
        d = obs.to_dict()
        assert d["bounding_box"]["x"] == 10
        assert d["bounding_box"]["width"] == 100


# =============================================================================
# E. CONFIDENCE VALIDATION
# =============================================================================


class TestConfidenceValidation:
    def test_valid_confidence(self):
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            confidence=0.91,
        )
        assert obs.confidence == 0.91

    def test_confidence_boundary_zero(self):
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            confidence=0.0,
        )
        assert obs.confidence == 0.0

    def test_confidence_boundary_one(self):
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            confidence=1.0,
        )
        assert obs.confidence == 1.0

    def test_invalid_confidence_too_high(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            AIObservation(
                observation_id="OBS-001",
                analysis_type="motion",
                confidence=1.5,
            )

    def test_invalid_confidence_negative(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            AIObservation(
                observation_id="OBS-001",
                analysis_type="motion",
                confidence=-0.1,
            )

    def test_none_confidence_allowed(self):
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            confidence=None,
        )
        assert obs.confidence is None


# =============================================================================
# F. MOTION OBSERVATION NORMALIZATION
# =============================================================================


class TestMotionNormalization:
    def test_motion_observation_normalized(self):
        raw = {
            "label": "motion_detected",
            "confidence": 0.85,
            "frame_number": 100,
            "media_timestamp": "3.333",
            "time_mapping": {
                "media_offset_seconds": 3.333,
                "mapping_status": "MAPPED",
                "normalized_timestamp": "2026-09-03T18:20:13.333",
            },
        }
        obs = _normalize_observation(
            raw,
            evidence_id="EV-001",
            recording_filename="test.h264",
            camera_id="CAM01",
            provider_name="opencv_motion",
            analysis_type="motion",
        )
        assert obs is not None
        assert obs.label == "motion_detected"
        assert obs.confidence == 0.85
        assert obs.evidence_id == "EV-001"
        assert obs.camera_id == "CAM01"
        assert obs.time_mapping.mapping_status == "MAPPED"

    def test_motion_no_motion_normalized(self):
        raw = {
            "label": "no_motion",
            "confidence": 0.0,
            "frame_number": 50,
        }
        obs = _normalize_observation(raw, analysis_type="motion")
        assert obs is not None
        assert obs.label == "no_motion"

    def test_motion_not_available_returns_none(self):
        raw = {"status": "NOT_AVAILABLE", "error": "OpenCV not installed"}
        obs = _normalize_observation(raw)
        assert obs is None


# =============================================================================
# G. OBJECT OBSERVATION NORMALIZATION
# =============================================================================


class TestObjectNormalization:
    def test_object_provider_works(self):
        import numpy as np
        provider = ObjectDetectionProvider()
        assert provider.is_available is True
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_object_with_mock_data(self):
        raw = {
            "label": "person",
            "confidence": 0.91,
            "bounding_box": {"x": 100, "y": 200, "width": 50, "height": 120},
        }
        obs = _normalize_observation(
            raw,
            analysis_type="object_detection",
            evidence_id="EV-001",
        )
        assert obs is not None
        assert obs.label == "person"
        assert obs.bounding_box.width == 50


# =============================================================================
# H. FACE OBSERVATION NORMALIZATION
# =============================================================================


class TestFaceNormalization:
    def test_face_provider_works(self):
        import numpy as np
        provider = FaceDetectionProvider()
        assert provider.is_available is True
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_face_with_mock_data(self):
        raw = {
            "label": "face_detected",
            "confidence": 0.88,
            "bounding_box": {"x": 50, "y": 60, "width": 80, "height": 80},
        }
        obs = _normalize_observation(
            raw,
            analysis_type="face_detection",
            evidence_id="EV-001",
        )
        assert obs is not None
        assert obs.label == "face_detected"
        assert obs.bounding_box.height == 80


# =============================================================================
# I. PROVENANCE PRESERVATION
# =============================================================================


class TestProvenancePreservation:
    def test_provenance_fields(self):
        obs = AIObservation(
            observation_id="OBS-001",
            evidence_id="EV-001",
            recording_filename="test.h264",
            camera_id="CAM01",
            analysis_type="motion",
            provider="opencv_motion",
            model_name="OpenCV",
        )
        d = obs.to_dict()
        assert d["evidence_id"] == "EV-001"
        assert d["recording_filename"] == "test.h264"
        assert d["camera_id"] == "CAM01"
        assert d["provider"] == "opencv_motion"
        assert d["model_name"] == "OpenCV"


# =============================================================================
# J. RECORDING REFERENCE PRESERVATION
# =============================================================================


class TestRecordingReference:
    def test_recording_filename_preserved(self):
        obs = AIObservation(
            observation_id="OBS-001",
            recording_filename="recording_18_20_30.h264",
            analysis_type="motion",
        )
        assert obs.recording_filename == "recording_18_20_30.h264"


# =============================================================================
# K. CAMERA REFERENCE PRESERVATION
# =============================================================================


class TestCameraReference:
    def test_camera_id_preserved(self):
        obs = AIObservation(
            observation_id="OBS-001",
            camera_id="CAM02",
            analysis_type="motion",
        )
        assert obs.camera_id == "CAM02"


# =============================================================================
# L. EVIDENCE REFERENCE PRESERVATION
# =============================================================================


class TestEvidenceReference:
    def test_evidence_id_preserved(self):
        obs = AIObservation(
            observation_id="OBS-001",
            evidence_id="EV-123",
            analysis_type="motion",
        )
        assert obs.evidence_id == "EV-123"


# =============================================================================
# M. MEDIA-RELATIVE TIMESTAMP
# =============================================================================


class TestMediaTimestamp:
    def test_media_timestamp_preserved(self):
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            media_timestamp="13.500",
        )
        assert obs.media_timestamp == "13.500"


# =============================================================================
# N. TIMESTAMP MAPPING
# =============================================================================


class TestTimestampMapping:
    def test_mapped_timestamp(self):
        tm = TimeMapping(
            media_offset_seconds=13.5,
            source_recording_start="2026-09-03T18:20:00",
            normalized_timestamp="2026-09-03T18:20:13.500",
            mapping_status="MAPPED",
        )
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            time_mapping=tm,
        )
        d = obs.to_dict()
        assert d["time_mapping"]["mapping_status"] == "MAPPED"
        assert d["time_mapping"]["normalized_timestamp"] == "2026-09-03T18:20:13.500"


# =============================================================================
# O. UNCERTAIN TIMESTAMP MAPPING
# =============================================================================


class TestUncertainTimestampMapping:
    def test_uncertain_mapping(self):
        tm = TimeMapping(
            media_offset_seconds=5.0,
            mapping_status="MAPPING_UNCERTAIN",
        )
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            time_mapping=tm,
        )
        d = obs.to_dict()
        assert d["time_mapping"]["mapping_status"] == "MAPPING_UNCERTAIN"


# =============================================================================
# P. UNKNOWN TIMEZONE
# =============================================================================


class TestUnknownTimezone:
    def test_unknown_timezone_preserved(self):
        tm = TimeMapping(
            timezone=None,
            mapping_status="MAPPING_UNCERTAIN",
        )
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            time_mapping=tm,
        )
        d = obs.to_dict()
        assert d["time_mapping"]["timezone"] is None


# =============================================================================
# Q. RECOVERED MEDIA PROVENANCE
# =============================================================================


class TestRecoveredMediaProvenance:
    def test_recovered_media_observation(self):
        obs = AIObservation(
            observation_id="OBS-001",
            evidence_id="EV-001",
            recording_filename="recovered_001.h264",
            analysis_type="motion",
            uncertainty="source media boundary not proven exact",
        )
        d = obs.to_dict()
        assert d["uncertainty"] == "source media boundary not proven exact"
        assert "recovered" in d["recording_filename"]


# =============================================================================
# R. UNCERTAIN RECOVERY BOUNDARY
# =============================================================================


class TestUncertainRecoveryBoundary:
    def test_boundary_uncertainty_propagated(self):
        obs = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            uncertainty="recovery boundary uncertain, exact_original_boundary_established=false",
        )
        assert "uncertain" in obs.uncertainty.lower()


# =============================================================================
# S. AI PIPELINE FAILURE ISOLATION
# =============================================================================


class TestAIPipelineFailure:
    def test_failed_provider_returns_result(self):
        result = analyze_video_file(
            video_path=Path("nonexistent.mp4"),
            analysis_type="motion",
            evidence_id="EV-001",
        )
        assert result.status == FAILED
        assert result.error is not None
        assert result.observations == []

    def test_unknown_analysis_type(self):
        result = analyze_video_file(
            video_path=Path("test.mp4"),
            analysis_type="unknown_type",
        )
        assert result.status == NOT_AVAILABLE

    def test_object_detection_provider_available(self):
        result = analyze_video_file(
            video_path=Path("nonexistent.mp4"),
            analysis_type="object_detection",
        )
        assert result.status == FAILED
        assert result.error is not None


# =============================================================================
# T. AI DISABLED WHILE FORENSIC PIPELINE STILL WORKS
# =============================================================================


class TestAIDisabledForensicWorks:
    def test_forensic_models_unaffected(self):
        from app.models.dvr_evidence import Recording
        rec = Recording(filename="test.h264")
        assert rec.filename == "test.h264"

    def test_investigative_service_unaffected(self):
        from app.services.investigative_service import InvestigativeIndex
        index = InvestigativeIndex()
        assert index.all_recordings() == []

    def test_analysis_pipeline_unaffected(self):
        from app.models.analysis_pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline(evidence_id="EV-001")
        assert pipeline.evidence_id == "EV-001"


# =============================================================================
# U. OBSERVATION FILTERING
# =============================================================================


class TestObservationFiltering:
    def test_filter_by_camera(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            camera_id="CAM01",
            analysis_type="motion",
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            camera_id="CAM02",
            analysis_type="motion",
        )
        filtered = filter_observations([obs1, obs2], camera_id="CAM01")
        assert len(filtered) == 1
        assert filtered[0].camera_id == "CAM01"

    def test_filter_by_analysis_type(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            analysis_type="object_detection",
        )
        filtered = filter_observations([obs1, obs2], analysis_type="motion")
        assert len(filtered) == 1

    def test_filter_by_label(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            label="motion_detected",
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            analysis_type="motion",
            label="no_motion",
        )
        filtered = filter_observations([obs1, obs2], label="motion_detected")
        assert len(filtered) == 1

    def test_filter_by_min_confidence(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            confidence=0.9,
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            analysis_type="motion",
            confidence=0.3,
        )
        filtered = filter_observations([obs1, obs2], min_confidence=0.5)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_filter_by_evidence(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            evidence_id="EV-001",
            analysis_type="motion",
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            evidence_id="EV-002",
            analysis_type="motion",
        )
        filtered = filter_observations([obs1, obs2], evidence_id="EV-001")
        assert len(filtered) == 1

    def test_filter_combined(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            camera_id="CAM01",
            analysis_type="motion",
            confidence=0.9,
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            camera_id="CAM02",
            analysis_type="motion",
            confidence=0.3,
        )
        filtered = filter_observations(
            [obs1, obs2],
            camera_id="CAM01",
            min_confidence=0.5,
        )
        assert len(filtered) == 1


# =============================================================================
# V. DETERMINISTIC ORDERING
# =============================================================================


class TestDeterministicOrdering:
    def test_sorted_by_media_timestamp(self):
        obs1 = AIObservation(
            observation_id="OBS-002",
            analysis_type="motion",
            media_timestamp="5.000",
        )
        obs2 = AIObservation(
            observation_id="OBS-001",
            analysis_type="motion",
            media_timestamp="2.000",
        )
        filtered = filter_observations([obs1, obs2])
        assert filtered[0].media_timestamp == "2.000"
        assert filtered[1].media_timestamp == "5.000"


# =============================================================================
# W. API RESPONSE SHAPE
# =============================================================================


class TestAPIResponseShape:
    def test_analysis_result_structure(self):
        result = AIAnalysisResult(
            evidence_id="EV-001",
            analysis_type="motion",
            status=COMPLETED,
        )
        d = result.to_dict()
        assert "evidence_id" in d
        assert "analysis_type" in d
        assert "status" in d
        assert "observations" in d
        assert "observation_count" in d

    def test_capabilities_result_structure(self):
        caps = get_capabilities()
        d = caps.to_dict()
        assert "motion" in d
        assert "object_detection" in d
        assert "face_detection" in d
        for cap_type in ["motion", "object_detection", "face_detection"]:
            assert "available" in d[cap_type]


# =============================================================================
# X. REPORT OUTPUT
# =============================================================================


class TestReportOutput:
    def test_report_with_ai_results(self):
        from app.services.forensic_report_generator import generate_forensic_report
        import tempfile

        evidence_data = {
            "evidence_id": "EV-TEST",
            "filename": "test.E01",
            "size": 1000,
            "sha256": "abc123",
            "file_type": "forensic-image/e01",
            "classification": "FORENSIC_IMAGE",
            "vendor": "Hikvision",
            "vendor_confidence": 0.95,
            "detection_method": "config_markers",
            "ai_results": {
                "motion": {
                    "status": "COMPLETED",
                    "provider": "opencv_motion",
                    "observations": [
                        {
                            "camera_id": "CAM01",
                            "recording_filename": "test.h264",
                            "label": "motion_detected",
                            "confidence": 0.85,
                            "frame_number": 100,
                            "media_timestamp": "3.333",
                            "time_mapping": {"mapping_status": "MAPPED"},
                        }
                    ],
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_forensic_report(evidence_data, Path(tmpdir))
            assert result["status"] == "GENERATED"
            assert Path(result["pdf_path"]).exists()
            assert Path(result["json_path"]).exists()


# =============================================================================
# Y. CHAIN-OF-CUSTODY INTEGRATION
# =============================================================================


class TestChainOfCustodyIntegration:
    def test_custody_event_types(self):
        from app.services.chain_of_custody import record_custody_event
        assert callable(record_custody_event)

    def test_ai_analysis_result_has_timestamps(self):
        result = AIAnalysisResult(
            evidence_id="EV-001",
            started_at="2026-09-03T18:20:00+00:00",
            completed_at="2026-09-03T18:20:05+00:00",
        )
        assert result.started_at is not None
        assert result.completed_at is not None


# =============================================================================
# Z. NO-OVERCLAIMING CHECKS
# =============================================================================


class TestNoOverclaiming:
    def test_no_incident_claims(self):
        import app.models.ai_observation as mod
        import inspect
        all_code = inspect.getsource(mod)
        forbidden = ["incident", "suspect", "crime", "confirmed identity"]
        for term in forbidden:
            assert term.lower() not in all_code.lower(), f"Overclaiming term '{term}' found"

    def test_no_incident_in_providers(self):
        import app.services.ai_providers as mod
        import inspect
        all_code = inspect.getsource(mod)
        forbidden = ["incident", "suspect", "crime", "confirmed identity"]
        for term in forbidden:
            assert term.lower() not in all_code.lower(), f"Overclaiming term '{term}' found"

    def test_no_incident_in_analysis(self):
        import app.services.ai_analysis as mod
        import inspect
        all_code = inspect.getsource(mod)
        forbidden = ["incident", "suspect", "crime", "confirmed identity"]
        for term in forbidden:
            assert term.lower() not in all_code.lower(), f"Overclaiming term '{term}' found"

    def test_valid_analysis_types(self):
        assert MOTION == "motion"
        assert OBJECT_DETECTION == "object_detection"
        assert FACE_DETECTION == "face_detection"
        assert len(VALID_ANALYSIS_TYPES) == 3

    def test_invalid_analysis_type_rejected(self):
        with pytest.raises(ValueError, match="Invalid analysis_type"):
            AIObservation(
                observation_id="OBS-001",
                analysis_type="incident_detection",
            )

    def test_analysis_result_status_values(self):
        assert NOT_REQUESTED == "NOT_REQUESTED"
        assert NOT_AVAILABLE == "NOT_AVAILABLE"
        assert COMPLETED == "COMPLETED"
        assert FAILED == "FAILED"


# =============================================================================
# REGRESSION: Phase 1-5 FUNCTIONALITY
# =============================================================================


class TestPhaseRegression:
    def test_timestamp_result_model(self):
        from app.models.dvr_evidence import TimestampResult
        ts = TimestampResult(original="2026-09-03 18:00:00")
        assert ts.original == "2026-09-03 18:00:00"

    def test_recording_model(self):
        from app.models.dvr_evidence import Recording
        rec = Recording(filename="test.h264")
        assert rec.filename == "test.h264"

    def test_dvr_evidence_model(self):
        from app.models.dvr_evidence import DVREvidence
        evidence = DVREvidence(vendor="Hikvision")
        assert evidence.vendor == "Hikvision"

    def test_investigative_models(self):
        from app.models.investigative import InvestigativeResult
        result = InvestigativeResult(query_type="test")
        assert result.query_type == "test"

    def test_investigative_service(self):
        from app.services.investigative_service import InvestigativeIndex
        index = InvestigativeIndex()
        assert index.all_recordings() == []

    def test_analysis_pipeline(self):
        from app.models.analysis_pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline(evidence_id="EV-001")
        assert pipeline.evidence_id == "EV-001"

    def test_vendor_capabilities(self):
        from app.models.vendor import ALL_CANONICAL_CAPABILITIES
        assert len(ALL_CANONICAL_CAPABILITIES) == 8

    def test_chain_of_custody(self):
        from app.services.chain_of_custody import record_custody_event, get_custody_history
        assert callable(record_custody_event)
        assert callable(get_custody_history)

    def test_timestamp_parser(self):
        from app.services.timestamp_parser import parse_timestamp
        ts = parse_timestamp("2026-09-03T18:00:00Z", None)
        assert ts.original == "2026-09-03T18:00:00Z"

    def test_motion_provider_available(self):
        provider = MotionProvider()
        assert provider.is_available is True
        assert provider.provider_name == "opencv_motion"

    def test_motion_provider_capabilities(self):
        provider = MotionProvider()
        caps = provider.get_capabilities()
        assert caps["available"] is True
        assert caps["analysis_type"] == "motion"

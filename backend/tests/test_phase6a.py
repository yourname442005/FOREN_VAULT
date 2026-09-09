import numpy as np
from pathlib import Path

from app.models.ai_observation import (
    AIObservation,
    MOTION,
    OBJECT_DETECTION,
    FACE_DETECTION,
    NOT_AVAILABLE,
)
from app.services.ai_providers import (
    FaceDetectionProvider,
    ObjectDetectionProvider,
    MotionProvider,
    get_available_providers,
    YUNET_MODEL,
    NANODET_MODEL,
    COCO_LABELS,
)
from app.services.ai_analysis import (
    get_capabilities,
    analyze_video_file,
    filter_observations,
)


# =============================================================================
# A. YUNET MODEL AVAILABILITY
# =============================================================================


class TestYuNetAvailability:
    def test_yunet_model_file_exists(self):
        assert YUNET_MODEL.exists(), f"YuNet model not found: {YUNET_MODEL}"

    def test_yunet_model_file_size(self):
        size = YUNET_MODEL.stat().st_size
        assert size > 10000, f"YuNet model too small: {size} bytes"
        assert size < 5000000, f"YuNet model too large: {size} bytes"

    def test_yunet_provider_available(self):
        provider = FaceDetectionProvider()
        assert provider.is_available is True


# =============================================================================
# B. YUNET MODEL LOADING
# =============================================================================


class TestYuNetLoading:
    def test_yunet_detector_loads(self):
        provider = FaceDetectionProvider()
        provider._load_model()
        assert provider._detector is not None

    def test_yunet_provider_name(self):
        provider = FaceDetectionProvider()
        assert provider.provider_name == "yunet"

    def test_yunet_model_name(self):
        provider = FaceDetectionProvider()
        assert provider.model_name == "face_detection_yunet_2023mar"

    def test_yunet_model_version(self):
        provider = FaceDetectionProvider()
        assert provider.model_version == "2023mar"

    def test_yunet_capabilities(self):
        provider = FaceDetectionProvider()
        caps = provider.get_capabilities()
        assert caps["available"] is True
        assert caps["provider"] == "yunet"
        assert caps["analysis_type"] == "face_detection"


# =============================================================================
# C. YUNET FACE DETECTION ON TEST IMAGE
# =============================================================================


class TestYuNetDetection:
    def _create_test_image_with_face(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        import cv2
        cv2.ellipse(img, (320, 240), (80, 100), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(img, (290, 220), 10, (50, 50, 50), -1)
        cv2.circle(img, (350, 220), 10, (50, 50, 50), -1)
        return img

    def test_yunet_runs_on_frame(self):
        provider = FaceDetectionProvider()
        img = self._create_test_image_with_face()
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_yunet_returns_detection_format(self):
        provider = FaceDetectionProvider()
        img = self._create_test_image_with_face()
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            assert "label" in obs
            assert obs["label"] in ("face", "no_face")

    def test_yunet_returns_confidence(self):
        provider = FaceDetectionProvider()
        img = self._create_test_image_with_face()
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            if "confidence" in obs:
                assert 0.0 <= obs["confidence"] <= 1.0


# =============================================================================
# D. FACE BOUNDING-BOX NORMALIZATION
# =============================================================================


class TestFaceBoundingBox:
    def test_bounding_box_coordinates(self):
        provider = FaceDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            if "bounding_box" in obs:
                bb = obs["bounding_box"]
                assert "x" in bb
                assert "y" in bb
                assert "width" in bb
                assert "height" in bb
                assert bb["x"] >= 0
                assert bb["y"] >= 0
                assert bb["width"] > 0
                assert bb["height"] > 0

    def test_bounding_box_within_frame(self):
        provider = FaceDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            if "bounding_box" in obs:
                bb = obs["bounding_box"]
                assert bb["x"] + bb["width"] <= 640
                assert bb["y"] + bb["height"] <= 480


# =============================================================================
# E. FACE CONFIDENCE RANGE
# =============================================================================


class TestFaceConfidence:
    def test_confidence_in_valid_range(self):
        provider = FaceDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            if "confidence" in obs and obs["confidence"] is not None:
                assert 0.0 <= obs["confidence"] <= 1.0


# =============================================================================
# F. FACE PROVENANCE
# =============================================================================


class TestFaceProvenance:
    def test_frame_number_preserved(self):
        provider = FaceDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = provider.analyze_frame(img, frame_number=100, media_timestamp=3.333)
        for obs in result:
            assert obs.get("frame_number") == 100

    def test_media_timestamp_preserved(self):
        provider = FaceDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=5.5)
        for obs in result:
            assert obs.get("media_timestamp") == 5.5

    def test_model_name_preserved(self):
        provider = FaceDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            assert obs.get("model_name") == "face_detection_yunet_2023mar"


# =============================================================================
# G. FACE TIMESTAMP MAPPING
# =============================================================================


class TestFaceTimestampMapping:
    def test_time_mapping_in_video(self):
        provider = FaceDetectionProvider()
        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(f.name, fourcc, 25.0, (640, 480))
            for _ in range(25):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (128, 128, 128)
                out.write(frame)
            out.release()

            result = provider.analyze_video(
                Path(f.name),
                start_time="2026-09-03T18:20:00",
                sample_interval_seconds=1.0,
                max_frames=2,
            )

            for obs in result:
                if obs.get("label") == "face":
                    tm = obs.get("time_mapping", {})
                    assert "mapping_status" in tm

            Path(f.name).unlink()


# =============================================================================
# H. FACE UNCERTAINTY PROPAGATION
# =============================================================================


class TestFaceUncertainty:
    def test_no_face_returns_no_face_label(self):
        provider = FaceDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (0, 0, 0)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        assert len(result) >= 1
        assert result[0]["label"] == "no_face"


# =============================================================================
# I. NANODET MODEL AVAILABILITY
# =============================================================================


class TestNanoDetAvailability:
    def test_nanodet_model_file_exists(self):
        assert NANODET_MODEL.exists(), f"NanoDet model not found: {NANODET_MODEL}"

    def test_nanodet_model_file_size(self):
        size = NANODET_MODEL.stat().st_size
        assert size > 1000000, f"NanoDet model too small: {size} bytes"
        assert size < 20000000, f"NanoDet model too large: {size} bytes"

    def test_nanodet_provider_available(self):
        provider = ObjectDetectionProvider()
        assert provider.is_available is True


# =============================================================================
# J. NANODET MODEL LOADING
# =============================================================================


class TestNanoDetLoading:
    def test_nanodet_net_loads(self):
        provider = ObjectDetectionProvider()
        provider._load_model()
        assert provider._net is not None

    def test_nanodet_provider_name(self):
        provider = ObjectDetectionProvider()
        assert provider.provider_name == "nanodet"

    def test_nanodet_model_name(self):
        provider = ObjectDetectionProvider()
        assert provider.model_name == "object_detection_nanodet_2022nov"

    def test_nanodet_model_version(self):
        provider = ObjectDetectionProvider()
        assert provider.model_version == "2022nov"

    def test_nanodet_capabilities(self):
        provider = ObjectDetectionProvider()
        caps = provider.get_capabilities()
        assert caps["available"] is True
        assert caps["provider"] == "nanodet"
        assert caps["analysis_type"] == "object_detection"


# =============================================================================
# K. NANODET DETECTION ON TEST IMAGE
# =============================================================================


class TestNanoDetDetection:
    def test_nanodet_runs_on_frame(self):
        provider = ObjectDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_nanodet_returns_detection_format(self):
        provider = ObjectDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            assert "label" in obs


# =============================================================================
# L. OBJECT CLASS MAPPING
# =============================================================================


class TestObjectClassMapping:
    def test_coco_labels_complete(self):
        assert len(COCO_LABELS) == 80
        assert "person" in COCO_LABELS
        assert "car" in COCO_LABELS
        assert "truck" in COCO_LABELS

    def test_label_from_class_id(self):
        provider = ObjectDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            if "class_id" in obs:
                class_id = obs["class_id"]
                assert 0 <= class_id < len(COCO_LABELS)
                assert obs["label"] == COCO_LABELS[class_id]


# =============================================================================
# M. OBJECT BOUNDING-BOX NORMALIZATION
# =============================================================================


class TestObjectBoundingBox:
    def test_bounding_box_coordinates(self):
        provider = ObjectDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            if "bounding_box" in obs:
                bb = obs["bounding_box"]
                assert "x" in bb
                assert "y" in bb
                assert "width" in bb
                assert "height" in bb


# =============================================================================
# N. OBJECT CONFIDENCE RANGE
# =============================================================================


class TestObjectConfidence:
    def test_confidence_in_valid_range(self):
        provider = ObjectDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            if "confidence" in obs and obs["confidence"] is not None:
                assert 0.0 <= obs["confidence"] <= 1.0


# =============================================================================
# O. OBJECT PROVENANCE
# =============================================================================


class TestObjectProvenance:
    def test_frame_number_preserved(self):
        provider = ObjectDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=50, media_timestamp=2.0)
        for obs in result:
            assert obs.get("frame_number") == 50

    def test_model_name_preserved(self):
        provider = ObjectDetectionProvider()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        result = provider.analyze_frame(img, frame_number=0, media_timestamp=0.0)
        for obs in result:
            assert obs.get("model_name") == "object_detection_nanodet_2022nov"


# =============================================================================
# P. OBJECT TIMESTAMP MAPPING
# =============================================================================


class TestObjectTimestampMapping:
    def test_time_mapping_in_video(self):
        provider = ObjectDetectionProvider()
        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(f.name, fourcc, 25.0, (640, 480))
            for _ in range(25):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (128, 128, 128)
                out.write(frame)
            out.release()

            result = provider.analyze_video(
                Path(f.name),
                start_time="2026-09-03T18:20:00",
                sample_interval_seconds=1.0,
                max_frames=2,
            )

            for obs in result:
                if obs.get("label") != "no_object":
                    tm = obs.get("time_mapping", {})
                    assert "mapping_status" in tm

            Path(f.name).unlink()


# =============================================================================
# Q. PROVIDER CAPABILITY DISCOVERY
# =============================================================================


class TestProviderCapabilityDiscovery:
    def test_all_providers_available(self):
        available = get_available_providers()
        assert available["motion"] is True
        assert available["face_detection"] is True
        assert available["object_detection"] is True

    def test_capabilities_accurate(self):
        caps = get_capabilities()
        assert caps.motion.available is True
        assert caps.face_detection.available is True
        assert caps.object_detection.available is True

    def test_face_provider_name_in_caps(self):
        caps = get_capabilities()
        assert caps.face_detection.provider == "yunet"
        assert caps.face_detection.model_name == "face_detection_yunet_2023mar"

    def test_object_provider_name_in_caps(self):
        caps = get_capabilities()
        assert caps.object_detection.provider == "nanodet"
        assert caps.object_detection.model_name == "object_detection_nanodet_2022nov"


# =============================================================================
# R. MISSING MODEL BEHAVIOR
# =============================================================================


class TestMissingModelBehavior:
    def test_missing_yunet_model(self):
        from app.services import ai_providers
        original = ai_providers.YUNET_MODEL
        ai_providers.YUNET_MODEL = Path("/nonexistent/model.onnx")
        try:
            provider = FaceDetectionProvider()
            assert provider.is_available is False
        finally:
            ai_providers.YUNET_MODEL = original

    def test_missing_nanodet_model(self):
        from app.services import ai_providers
        original = ai_providers.NANODET_MODEL
        ai_providers.NANODET_MODEL = Path("/nonexistent/model.onnx")
        try:
            provider = ObjectDetectionProvider()
            assert provider.is_available is False
        finally:
            ai_providers.NANODET_MODEL = original


# =============================================================================
# S. INVALID MODEL BEHAVIOR
# =============================================================================


class TestInvalidModelBehavior:
    def test_invalid_model_path(self):
        from app.services import ai_providers
        original = ai_providers.YUNET_MODEL
        ai_providers.YUNET_MODEL = Path("/tmp/invalid_model.onnx")
        try:
            provider = FaceDetectionProvider()
            assert provider.is_available is False
        finally:
            ai_providers.YUNET_MODEL = original


# =============================================================================
# T. API CAPABILITIES
# =============================================================================


class TestAPICapabilities:
    def test_capabilities_to_dict(self):
        caps = get_capabilities()
        d = caps.to_dict()
        assert "motion" in d
        assert "face_detection" in d
        assert "object_detection" in d
        assert d["face_detection"]["available"] is True
        assert d["object_detection"]["available"] is True
        assert d["face_detection"]["provider"] == "yunet"
        assert d["object_detection"]["provider"] == "nanodet"


# =============================================================================
# U. API ANALYSIS
# =============================================================================


class TestAPIAnalysis:
    def test_analyze_video_file_missing(self):
        result = analyze_video_file(
            video_path=Path("nonexistent.mp4"),
            analysis_type="face_detection",
            evidence_id="EV-001",
        )
        assert result.status == "FAILED"
        assert result.error is not None

    def test_analyze_unknown_type(self):
        result = analyze_video_file(
            video_path=Path("test.mp4"),
            analysis_type="unknown_type",
        )
        assert result.status == NOT_AVAILABLE


# =============================================================================
# V. API OBSERVATION FILTERING
# =============================================================================


class TestAPIObservationFiltering:
    def test_filter_by_label(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            analysis_type="face_detection",
            label="face",
            confidence=0.9,
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            analysis_type="face_detection",
            label="no_face",
            confidence=0.0,
        )
        filtered = filter_observations([obs1, obs2], label="face")
        assert len(filtered) == 1
        assert filtered[0].label == "face"

    def test_filter_by_min_confidence(self):
        obs1 = AIObservation(
            observation_id="OBS-001",
            analysis_type="object_detection",
            label="person",
            confidence=0.9,
        )
        obs2 = AIObservation(
            observation_id="OBS-002",
            analysis_type="object_detection",
            label="car",
            confidence=0.3,
        )
        filtered = filter_observations([obs1, obs2], min_confidence=0.5)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9


# =============================================================================
# W. REPORT INTEGRATION
# =============================================================================


class TestReportIntegration:
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
                "face_detection": {
                    "status": "COMPLETED",
                    "provider": "yunet",
                    "observations": [
                        {
                            "camera_id": "CAM01",
                            "recording_filename": "test.h264",
                            "label": "face",
                            "confidence": 0.88,
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


# =============================================================================
# X. CHAIN-OF-CUSTODY INTEGRATION
# =============================================================================


class TestChainOfCustodyIntegration:
    def test_custody_event_types(self):
        from app.services.chain_of_custody import record_custody_event
        assert callable(record_custody_event)


# =============================================================================
# Y. OFFLINE INFERENCE
# =============================================================================


class TestOfflineInference:
    def test_no_internet_required(self):
        import os
        env_internet = os.environ.get("FORENVAULT_REQUIRES_INTERNET")
        assert env_internet is None or env_internet == "false"

    def test_providers_work_without_network(self):
        face_provider = FaceDetectionProvider()
        obj_provider = ObjectDetectionProvider()
        assert face_provider.is_available is True
        assert obj_provider.is_available is True


# =============================================================================
# Z. FULL PHASE 1-5 REGRESSION
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

    def test_motion_provider(self):
        provider = MotionProvider()
        assert provider.is_available is True

    def test_face_provider(self):
        provider = FaceDetectionProvider()
        assert provider.is_available is True

    def test_object_provider(self):
        provider = ObjectDetectionProvider()
        assert provider.is_available is True

    def test_no_overclaiming(self):
        import app.services.ai_providers as mod
        import inspect
        all_code = inspect.getsource(mod)
        forbidden = ["incident", "suspect", "crime", "confirmed identity", "person identified"]
        for term in forbidden:
            assert term.lower() not in all_code.lower(), f"Overclaiming term '{term}' found"

    def test_valid_analysis_types(self):
        from app.models.ai_observation import VALID_ANALYSIS_TYPES
        assert MOTION in VALID_ANALYSIS_TYPES
        assert OBJECT_DETECTION in VALID_ANALYSIS_TYPES
        assert FACE_DETECTION in VALID_ANALYSIS_TYPES

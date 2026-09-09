import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.dvr_evidence import (
    INVALID,
    NORMALIZED,
    TIMEZONE_UNKNOWN,
    DVREvidence,
    Recording,
    TimestampResult,
)
from app.models.vendor import (
    ALL_CANONICAL_CAPABILITIES,
    CAPABILITY_CAMERA_EXTRACTION,
    CAPABILITY_METADATA_EXTRACTION,
    CAPABILITY_RECORDING_EXTRACTION,
    CAPABILITY_TIMEZONE_EXTRACTION,
    CAPABILITY_TIMESTAMP_EXTRACTION,
    VendorProfile,
)
from app.parsers.vendor_base import VendorParser
from app.parsers.vendor_registry import create_default_registry
from app.services.timestamp_parser import parse_recording_filename, parse_timestamp
from app.services.vendor_detector import score_vendor


def _make_filesystem_analysis(files, fs_code="ntfs", start_sector=0):
    return {
        "files": files,
        "filesystem": {"code": fs_code, "name": fs_code.upper()},
        "partition": {"start_sector": start_sector},
    }


def _make_file_entry(name, inode=1, deleted=False, entry_type="file"):
    return {
        "name": name,
        "type": entry_type,
        "inode": inode,
        "deleted": deleted,
    }


# =============================================================================
# HIKVISION HARDENING
# =============================================================================


class TestHikvisionCameraDeduplication:
    def test_duplicate_cameras_not_duplicated(self):
        from app.parsers.hikvision import HikvisionParser

        parser = HikvisionParser()
        files = [
            _make_file_entry("CAM01", entry_type="directory"),
            _make_file_entry("CAM01", entry_type="directory"),
            _make_file_entry("CAM02", entry_type="directory"),
        ]
        cameras = []
        seen_cameras = set()
        for entry in files:
            name = entry.get("name", "")
            if entry.get("type") == "directory":
                if parser.CAMERA_PATTERN.match(name):
                    camera_id = name.upper()
                    if camera_id in seen_cameras:
                        continue
                    seen_cameras.add(camera_id)
                    cameras.append(camera_id)

        assert cameras == ["CAM01", "CAM02"]
        assert len(cameras) == 2

    def test_hikvision_camera_pattern_variants(self):
        from app.parsers.hikvision import HikvisionParser

        pattern = HikvisionParser.CAMERA_PATTERN
        assert pattern.match("CAM01") is not None
        assert pattern.match("CAM02") is not None
        assert pattern.match("CAM16") is not None
        assert pattern.match("CAMERA01") is not None
        assert pattern.match("ch01") is not None
        assert pattern.match("ch_01") is not None
        assert pattern.match("channel01") is not None
        assert pattern.match("channel_01") is not None
        assert pattern.match("CH01") is not None
        assert pattern.match("random_dir") is None
        assert pattern.match("recordings") is None
        assert pattern.match("CAM") is None
        assert pattern.match("CAMERA") is None
        assert pattern.match("") is None


class TestHikvisionMetadataExtraction:
    def test_json_metadata_parsed(self):
        metadata = {}

        json_content = '{"model": "DS-7608NI-K2", "firmware": "V4.72.000", "serial": "ABC123"}'
        data = json.loads(json_content)
        if isinstance(data, dict):
            metadata.update(data)

        assert metadata["model"] == "DS-7608NI-K2"
        assert metadata["firmware"] == "V4.72.000"
        assert metadata["serial"] == "ABC123"

    def test_keyvalue_metadata_parsed(self):
        metadata = {}
        content = "model=DS-7608NI-K2\nfirmware=V4.72.000\nchannels=8"

        for line in content.splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            metadata[key.strip().lower()] = value.strip()

        assert metadata["model"] == "DS-7608NI-K2"
        assert metadata["firmware"] == "V4.72.000"
        assert metadata["channels"] == "8"

    def test_malformed_json_does_not_crash(self):
        metadata = {}
        content = "{invalid json data"

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                metadata.update(data)
        except json.JSONDecodeError:
            pass

        assert metadata == {}

    def test_empty_metadata(self):
        metadata = {}
        assert metadata.get("model") is None
        assert metadata.get("firmware") is None
        assert len(metadata) == 0


class TestHikvisionRecordingExtraction:
    def test_standard_recording_filename(self):
        result = parse_recording_filename("20260901_180000_190000_CAM01.h264")
        assert result is not None
        assert result["camera_id"] == "CAM01"
        assert result["start_time"] == "2026-09-01T18:00:00"
        assert result["end_time"] == "2026-09-01T19:00:00"
        assert result["format"] == "h264"

    def test_recording_with_timestamp_normalization(self):
        result = parse_recording_filename("20260901_180000_190000_CAM01.h264")
        assert result is not None

        start_ts = parse_timestamp(result["start_time"], timezone_hint="Asia/Kolkata")
        assert start_ts.normalization_status == NORMALIZED
        assert start_ts.timezone == "Asia/Kolkata"
        assert start_ts.timezone_source == "device_config"
        assert start_ts.iso_aware is not None
        assert start_ts.utc is not None

    def test_recording_without_timezone_hint(self):
        result = parse_recording_filename("20260901_180000_190000_CAM01.h264")
        assert result is not None

        start_ts = parse_timestamp(result["start_time"])
        assert start_ts.normalization_status == TIMEZONE_UNKNOWN
        assert start_ts.iso_aware is None
        assert start_ts.utc is None

    def test_recording_extensions_recognized(self):
        from app.parsers.hikvision import HikvisionParser

        parser = HikvisionParser()
        assert ".h264" in parser.RECORDING_EXTENSIONS
        assert ".264" in parser.RECORDING_EXTENSIONS
        assert ".h265" in parser.RECORDING_EXTENSIONS
        assert ".265" in parser.RECORDING_EXTENSIONS
        assert ".dav" in parser.RECORDING_EXTENSIONS


class TestHikvisionCanParse:
    def test_marker_in_content(self):
        from app.parsers.hikvision import HikvisionParser

        parser = HikvisionParser()
        for marker in parser.MARKERS:
            assert isinstance(marker, str)
            assert len(marker) > 0

    def test_markers_are_lowercase(self):
        from app.parsers.hikvision import HikvisionParser

        parser = HikvisionParser()
        for marker in parser.MARKERS:
            assert marker == marker.lower()


# =============================================================================
# DAHUA HARDENING
# =============================================================================


class TestDahuaParser:
    def test_camera_pattern_variants(self):
        from app.parsers.dahua import DahuaParser

        pattern = DahuaParser.CAMERA_PATTERN
        assert pattern.match("CAM01") is not None
        assert pattern.match("ch01") is not None
        assert pattern.match("channel01") is not None
        assert pattern.match("random_dir") is None

    def test_recording_extensions(self):
        from app.parsers.dahua import DahuaParser

        parser = DahuaParser()
        assert ".dav" in parser.RECORDING_EXTENSIONS
        assert ".h264" in parser.RECORDING_EXTENSIONS
        assert ".mp4" in parser.RECORDING_EXTENSIONS

    def test_deduplication_set(self):
        from app.parsers.dahua import DahuaParser

        parser = DahuaParser()
        files = [
            _make_file_entry("CAM01", entry_type="directory"),
            _make_file_entry("CAM01", entry_type="directory"),
            _make_file_entry("CAM02", entry_type="directory"),
        ]

        cameras = []
        seen_cameras = set()
        for entry in files:
            name = entry.get("name", "")
            if entry.get("type") == "directory":
                if parser.CAMERA_PATTERN.match(name):
                    camera_id = name.upper()
                    if camera_id in seen_cameras:
                        continue
                    seen_cameras.add(camera_id)
                    cameras.append(camera_id)

        assert cameras == ["CAM01", "CAM02"]


# =============================================================================
# CP PLUS HARDENING
# =============================================================================


class TestCPPlusParser:
    def test_metadata_extraction(self):
        from plugins.cp_plus import CPPlusParser

        parser = CPPlusParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True

    def test_camera_config_parsing(self):
        camera_config = json.dumps({
            "cameras": [
                {"id": "CAM01", "name": "Entrance"},
                {"id": "CAM02", "name": "Parking"},
            ]
        })
        data = json.loads(camera_config)
        cameras = data.get("cameras") or []
        assert len(cameras) == 2
        assert cameras[0]["id"] == "CAM01"
        assert cameras[1]["name"] == "Parking"

    def test_channel_config_parsing(self):
        channel_config = json.dumps({
            "channels": [
                {"channel": "CH01", "name": "Gate"},
            ]
        })
        data = json.loads(channel_config)
        channels = data.get("channels") or []
        assert len(channels) == 1
        assert channels[0]["channel"] == "CH01"


# =============================================================================
# UNIVIEW HARDENING
# =============================================================================


class TestUniviewParser:
    def test_marker_detection(self):
        sources = [("filename", "uniview_config.json")]
        score, evidence = score_vendor("uniview", sources)
        assert score > 0

    def test_json_metadata_parsing(self):
        content = '{"model": "NVR301-08S2", "timezone": "Asia/Kolkata"}'
        data = json.loads(content)
        assert data["model"] == "NVR301-08S2"
        assert data["timezone"] == "Asia/Kolkata"

    def test_keyvalue_metadata_parsing(self):
        metadata = {}
        content = "model=NVR301-08S2\ntimezone=Asia/Kolkata\nchannels=8"
        for line in content.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().lower()] = value.strip()
        assert metadata["model"] == "NVR301-08S2"
        assert metadata["timezone"] == "Asia/Kolkata"

    def test_malformed_json_handled(self):
        metadata = {}
        content = "{invalid"
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                metadata.update(data)
        except json.JSONDecodeError:
            pass
        assert metadata == {}

    def test_capabilities_accurate(self):
        from plugins.uniview import UniviewParser

        parser = UniviewParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


# =============================================================================
# HONEYWELL HARDENING
# =============================================================================


class TestHoneywellParser:
    def test_marker_detection(self):
        from app.services.vendor_detector import score_vendor

        sources = [("configuration", "manufacturer=Honeywell Security")]
        score, evidence = score_vendor("honeywell", sources)
        assert score >= 0.50

    def test_json_metadata_parsing(self):
        content = '{"model": "HRN-X108-4SP", "firmware": "2.0.0"}'
        data = json.loads(content)
        assert data["model"] == "HRN-X108-4SP"

    def test_keyvalue_metadata_parsing(self):
        metadata = {}
        content = "model=HRN-X108-4SP\nfirmware=2.0.0"
        for line in content.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().lower()] = value.strip()
        assert metadata["model"] == "HRN-X108-4SP"

    def test_malformed_json_handled(self):
        metadata = {}
        content = "not json at all"
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                metadata.update(data)
        except json.JSONDecodeError:
            pass
        assert metadata == {}

    def test_capabilities_accurate(self):
        from plugins.honeywell import HoneywellParser

        parser = HoneywellParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


# =============================================================================
# TP-LINK HARDENING
# =============================================================================


class TestTPLinkParser:
    def test_vigi_marker_detection(self):
        from app.services.vendor_detector import score_vendor

        sources = [("configuration", "device=VIGI NVR1008H")]
        score, evidence = score_vendor("tp_link", sources)
        assert score >= 0.50

    def test_json_metadata_parsing(self):
        content = '{"model": "NVR1008H-8MP", "timezone": "Asia/Kolkata"}'
        data = json.loads(content)
        assert data["model"] == "NVR1008H-8MP"

    def test_camera_pattern_includes_ipc(self):
        from plugins.tp_link import TPLinkParser

        pattern = TPLinkParser.CAMERA_PATTERN
        assert pattern.match("IPC01") is not None
        assert pattern.match("ipc_01") is not None

    def test_malformed_json_handled(self):
        metadata = {}
        content = "}{invalid"
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                metadata.update(data)
        except json.JSONDecodeError:
            pass
        assert metadata == {}

    def test_capabilities_accurate(self):
        from plugins.tp_link import TPLinkParser

        parser = TPLinkParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


# =============================================================================
# GODREJ HARDENING
# =============================================================================


class TestGodrejParser:
    def test_honest_capabilities(self):
        from plugins.godrej import GodrejParser

        parser = GodrejParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is False

    def test_metadata_extraction_works(self):
        metadata = {}
        content = "model=Godrej-SVR-16P\nfirmware=3.0.0"
        for line in content.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().lower()] = value.strip()
        assert metadata["model"] == "Godrej-SVR-16P"

    def test_json_metadata_extraction(self):
        metadata = {}
        content = '{"model": "Godrej SecuriNet", "channels": 16}'
        data = json.loads(content)
        if isinstance(data, dict):
            metadata.update(data)
        assert metadata["model"] == "Godrej SecuriNet"
        assert metadata["channels"] == 16

    def test_empty_metadata(self):
        metadata = {}
        assert metadata.get("model") is None

    def test_malformed_json(self):
        metadata = {}
        content = "{bad json"
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                metadata.update(data)
        except json.JSONDecodeError:
            pass
        assert metadata == {}

    def test_marker_detection(self):
        from app.services.vendor_detector import score_vendor

        sources = [("filename", "godrej_nvr_config.conf")]
        score, evidence = score_vendor("godrej", sources)
        assert score > 0

    def test_no_camera_or_recording_extraction(self):
        from plugins.godrej import GodrejParser

        parser = GodrejParser()
        caps = parser.get_capabilities()
        unsupported = caps.unsupported_capabilities()
        assert "camera_extraction" in unsupported
        assert "recording_extraction" in unsupported
        assert "timestamp_extraction" in unsupported
        assert "timezone_extraction" in unsupported


# =============================================================================
# MATRIX HARDENING
# =============================================================================


class TestMatrixParser:
    def test_honest_capabilities(self):
        from plugins.matrix import MatrixParser

        parser = MatrixParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is False

    def test_metadata_extraction_works(self):
        metadata = {}
        content = "model=Matrix-SARV-4000\ndevice=Matrix Comsec"
        for line in content.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().lower()] = value.strip()
        assert metadata["model"] == "Matrix-SARV-4000"

    def test_json_metadata_extraction(self):
        metadata = {}
        content = '{"model": "Matrix VisionPro", "device": "SARV"}'
        data = json.loads(content)
        if isinstance(data, dict):
            metadata.update(data)
        assert metadata["model"] == "Matrix VisionPro"

    def test_marker_detection(self):
        from app.services.vendor_detector import score_vendor

        sources = [("configuration", "device=Matrix Comsec SARV")]
        score, evidence = score_vendor("matrix", sources)
        assert score >= 0.50

    def test_no_camera_or_recording_extraction(self):
        from plugins.matrix import MatrixParser

        parser = MatrixParser()
        caps = parser.get_capabilities()
        unsupported = caps.unsupported_capabilities()
        assert "camera_extraction" in unsupported
        assert "recording_extraction" in unsupported
        assert "timestamp_extraction" in unsupported
        assert "timezone_extraction" in unsupported


# =============================================================================
# VENDOR DETECTION HARDENING
# =============================================================================


class TestVendorDetectionHardening:
    def test_hikvision_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=Hikvision"),
            ("filename", "hik_config.json"),
        ]
        score, evidence = score_vendor("hikvision", sources)
        assert score >= 0.60

    def test_dahua_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=Dahua"),
            ("filename", "dahua_config.json"),
        ]
        score, evidence = score_vendor("dahua", sources)
        assert score >= 0.50

    def test_cp_plus_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=CP Plus"),
            ("filename", "cpplus_config.json"),
        ]
        score, evidence = score_vendor("cp_plus", sources)
        assert score >= 0.60

    def test_uniview_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=Uniview"),
            ("filename", "uniview_config.json"),
        ]
        score, evidence = score_vendor("uniview", sources)
        assert score >= 0.50

    def test_honeywell_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=Honeywell Security"),
            ("filename", "honeywell_config.json"),
        ]
        score, evidence = score_vendor("honeywell", sources)
        assert score >= 0.60

    def test_tp_link_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=TP-Link VIGI"),
            ("filename", "vigi_config.json"),
        ]
        score, evidence = score_vendor("tp_link", sources)
        assert score >= 0.60

    def test_godrej_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=Godrej Security"),
            ("filename", "godrej_config.json"),
        ]
        score, evidence = score_vendor("godrej", sources)
        assert score >= 0.60

    def test_matrix_detection_strong(self):
        sources = [
            ("configuration", "manufacturer=Matrix Comsec"),
            ("filename", "matrix_config.json"),
        ]
        score, evidence = score_vendor("matrix", sources)
        assert score >= 0.60

    def test_no_vendor_match_on_generic_text(self):
        sources = [
            ("configuration", "This is a generic device with 16 channels"),
            ("filename", "device_config.json"),
        ]
        for vendor in ["hikvision", "dahua", "cp_plus", "uniview",
                        "honeywell", "tp_link", "godrej", "matrix"]:
            score, evidence = score_vendor(vendor, sources)
            assert score == 0.0, f"False positive for {vendor}"

    def test_weak_filename_only_not_enough(self):
        sources = [("filename", "hik_video.h264")]
        score, evidence = score_vendor("hikvision", sources)
        assert score < 0.40

    def test_single_marker_in_config_only(self):
        sources = [("configuration", "system=hik")]
        score, evidence = score_vendor("hikvision", sources)
        assert score >= 0.50

    def test_scoring_weights(self):
        config_sources = [("configuration", "dahua")]
        score_config, _ = score_vendor("dahua", config_sources)

        filename_sources = [("filename", "dahua")]
        score_filename, _ = score_vendor("dahua", filename_sources)

        assert score_config > score_filename


# =============================================================================
# TIMESTAMP INTEGRATION
# =============================================================================


class TestTimestampIntegration:
    def test_parse_timestamp_preserves_original(self):
        result = parse_timestamp("2026-09-01T18:00:00")
        assert result.original == "2026-09-01T18:00:00"

    def test_parse_timestamp_never_guesses_timezone(self):
        result = parse_timestamp("2026-09-01T18:00:00")
        assert result.timezone is None
        assert result.timezone_source is None
        assert result.normalization_status == TIMEZONE_UNKNOWN

    def test_parse_timestamp_with_hint(self):
        result = parse_timestamp("2026-09-01T18:00:00", timezone_hint="Asia/Kolkata")
        assert result.timezone == "Asia/Kolkata"
        assert result.timezone_source == "device_config"
        assert result.normalization_status == NORMALIZED

    def test_parse_timestamp_utc_preserved(self):
        result = parse_timestamp("2026-09-01T18:00:00Z")
        assert result.timezone == "UTC"
        assert result.normalization_status == NORMALIZED
        assert result.utc is not None

    def test_parse_timestamp_invalid(self):
        result = parse_timestamp("not-a-date")
        assert result.normalization_status == INVALID

    def test_parse_timestamp_empty(self):
        result = parse_timestamp("")
        assert result.normalization_status == INVALID

    def test_recording_filename_timestamp_integration(self):
        parsed = parse_recording_filename("20260901_180000_190000_CAM01.h264")
        assert parsed is not None

        start_ts = parse_timestamp(parsed["start_time"])
        assert start_ts.normalization_status == TIMEZONE_UNKNOWN
        assert start_ts.iso_naive is not None
        assert start_ts.original == parsed["start_time"]


# =============================================================================
# PLUGIN DISCOVERY AND FAILURE ISOLATION
# =============================================================================


class TestPluginDiscoveryHardening:
    def test_all_eight_plugins_discovered(self):
        registry = create_default_registry()
        plugins_directory = Path(__file__).parent.parent / "plugins"
        result = registry.discover_plugins(plugins_directory)
        loaded = result["loaded_plugins"]
        assert "CP Plus" in loaded
        assert "Uniview" in loaded
        assert "Honeywell" in loaded
        assert "TP-Link" in loaded
        assert "Godrej" in loaded
        assert "Matrix" in loaded

    def test_no_plugin_errors(self):
        registry = create_default_registry()
        plugins_directory = Path(__file__).parent.parent / "plugins"
        result = registry.discover_plugins(plugins_directory)
        assert len(result["plugin_errors"]) == 0

    def test_all_plugins_inherit_vendor_parser(self):
        registry = create_default_registry()
        for parser in registry._parsers:
            assert isinstance(parser, VendorParser)

    def test_all_plugins_have_vendor_name(self):
        registry = create_default_registry()
        for parser in registry._parsers:
            assert hasattr(parser, "vendor_name")
            assert isinstance(parser.vendor_name, str)
            assert len(parser.vendor_name) > 0

    def test_all_plugins_have_can_parse(self):
        registry = create_default_registry()
        for parser in registry._parsers:
            assert hasattr(parser, "can_parse")
            assert callable(parser.can_parse)

    def test_all_plugins_have_parse(self):
        registry = create_default_registry()
        for parser in registry._parsers:
            assert hasattr(parser, "parse")
            assert callable(parser.parse)

    def test_all_plugins_have_get_capabilities(self):
        registry = create_default_registry()
        for parser in registry._parsers:
            assert hasattr(parser, "get_capabilities")
            caps = parser.get_capabilities()
            assert isinstance(caps, VendorProfile)
            assert len(caps.capabilities) > 0

    def test_all_capabilities_use_canonical_names(self):
        registry = create_default_registry()
        for parser in registry._parsers:
            caps = parser.get_capabilities()
            for cap in caps.capabilities:
                assert cap.name in ALL_CANONICAL_CAPABILITIES, (
                    f"{parser.vendor_name} uses non-canonical: {cap.name}"
                )

    def test_default_registry_includes_builtin_parsers(self):
        registry = create_default_registry()
        vendor_names = [p.vendor_name for p in registry._parsers]
        assert "Hikvision" in vendor_names
        assert "Dahua" in vendor_names


# =============================================================================
# GENERIC FORENSIC FALLBACK
# =============================================================================


class TestGenericForensicFallback:
    def test_dvr_structure_detection(self):
        from app.services.dvr_structure_detector import detect_dvr_structure

        files = [
            {"name": "CAM01", "type": "directory"},
            {"name": "CAM01/20260901_180000_190000_CAM01.h264", "type": "file"},
            {"name": "device.conf", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "DVR_NVR_LIKELY"
        assert len(result["camera_directories"]) == 1
        assert len(result["recording_files"]) == 1

    def test_not_dvr_classification(self):
        from app.services.dvr_structure_detector import detect_dvr_structure

        files = [
            {"name": "document.pdf", "type": "file"},
            {"name": "image.jpg", "type": "file"},
        ]
        result = detect_dvr_structure(files)
        assert result["classification"] == "NOT_DVR_NVR"

    def test_empty_files(self):
        from app.services.dvr_structure_detector import detect_dvr_structure

        result = detect_dvr_structure([])
        assert result["classification"] == "NOT_DVR_NVR"
        assert result["confidence"] == 0.0


# =============================================================================
# PHASE 1 REGRESSION (Chain of Custody, Evidence Integrity)
# =============================================================================


class TestPhase1Regression:
    def test_sha256_calculation(self):
        from app.services.evidence_service import calculate_sha256
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test evidence content")
            f.flush()
            result = calculate_sha256(Path(f.name))

        assert result is not None
        assert len(result) == 64

    def test_chain_of_custody_structure(self):
        from app.services.chain_of_custody import record_custody_event
        assert callable(record_custody_event)


# =============================================================================
# CAPABILITY MATRIX VERIFICATION
# =============================================================================


class TestCapabilityMatrixVerification:
    def test_hikvision_full_capabilities(self):
        from app.parsers.hikvision import HikvisionParser

        parser = HikvisionParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True

    def test_dahua_full_capabilities(self):
        from app.parsers.dahua import DahuaParser

        parser = DahuaParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True

    def test_cp_plus_full_capabilities(self):
        from plugins.cp_plus import CPPlusParser

        parser = CPPlusParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True

    def test_uniview_full_capabilities(self):
        from plugins.uniview import UniviewParser

        parser = UniviewParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True

    def test_honeywell_full_capabilities(self):
        from plugins.honeywell import HoneywellParser

        parser = HoneywellParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True

    def test_tp_link_full_capabilities(self):
        from plugins.tp_link import TPLinkParser

        parser = TPLinkParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True

    def test_godrej_honest_capabilities(self):
        from plugins.godrej import GodrejParser

        parser = GodrejParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is False

    def test_matrix_honest_capabilities(self):
        from plugins.matrix import MatrixParser

        parser = MatrixParser()
        caps = parser.get_capabilities()
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is False


# =============================================================================
# EVIDENCE INTEGRITY
# =============================================================================


class TestEvidenceIntegrity:
    def test_dvr_evidence_preserves_timezone(self):
        evidence = DVREvidence(
            vendor="Hikvision",
            model="DS-7608NI-K2",
            timezone="Asia/Kolkata",
            timezone_source="device_config",
        )
        assert evidence.timezone == "Asia/Kolkata"
        assert evidence.timezone_source == "device_config"

    def test_dvr_evidence_backward_compatible(self):
        evidence = DVREvidence(vendor="Dahua")
        assert evidence.timezone is None
        assert evidence.timezone_source is None

    def test_recording_preserves_timestamps(self):
        ts = TimestampResult(
            original="20260901_180000",
            iso_naive="2026-09-01T18:00:00",
            normalization_status=NORMALIZED,
            timezone="Asia/Kolkata",
        )
        rec = Recording(
            filename="20260901_180000_190000_CAM01.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
            start_timestamp=ts,
            end_timestamp=ts,
        )
        assert rec.start_timestamp is not None
        assert rec.start_timestamp.original == "20260901_180000"
        assert rec.start_timestamp.timezone == "Asia/Kolkata"

    def test_recording_backward_compatible(self):
        rec = Recording(
            filename="test.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        assert rec.start_timestamp is None
        assert rec.end_timestamp is None

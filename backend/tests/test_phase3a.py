import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.vendor import (
    ALL_CANONICAL_CAPABILITIES,
    CAPABILITY_CAMERA_EXTRACTION,
    CAPABILITY_DELETED_RECOVERY,
    CAPABILITY_FILESYSTEM_ANALYSIS,
    CAPABILITY_METADATA_EXTRACTION,
    CAPABILITY_RECORDING_EXTRACTION,
    CAPABILITY_TIMEZONE_EXTRACTION,
    CAPABILITY_TIMESTAMP_EXTRACTION,
    CAPABILITY_VENDOR_DETECTION,
    VendorCapability,
    VendorProfile,
)
from app.parsers.vendor_base import VendorParser
from app.parsers.vendor_registry import (
    create_default_registry,
)


class TestCapabilityModel:
    def test_canonical_capabilities_exist(self):
        assert CAPABILITY_VENDOR_DETECTION == "vendor_detection"
        assert CAPABILITY_METADATA_EXTRACTION == "metadata_extraction"
        assert CAPABILITY_CAMERA_EXTRACTION == "camera_extraction"
        assert CAPABILITY_RECORDING_EXTRACTION == "recording_extraction"
        assert CAPABILITY_TIMESTAMP_EXTRACTION == "timestamp_extraction"
        assert CAPABILITY_TIMEZONE_EXTRACTION == "timezone_extraction"
        assert CAPABILITY_FILESYSTEM_ANALYSIS == "filesystem_analysis"
        assert CAPABILITY_DELETED_RECOVERY == "deleted_recovery"

    def test_all_canonical_set_has_eight(self):
        assert len(ALL_CANONICAL_CAPABILITIES) == 8

    def test_vendor_capability_construction(self):
        cap = VendorCapability(
            name="metadata_extraction",
            supported=True,
            detail="Extracts from config files",
        )
        assert cap.name == "metadata_extraction"
        assert cap.supported is True
        assert cap.detail == "Extracts from config files"

    def test_vendor_profile_has_capability(self):
        profile = VendorProfile(
            vendor_name="Test",
            capabilities=[
                VendorCapability(
                    name="metadata_extraction",
                    supported=True,
                ),
                VendorCapability(
                    name="camera_extraction",
                    supported=False,
                ),
            ],
        )
        assert profile.has_capability("metadata_extraction") is True
        assert profile.has_capability("camera_extraction") is False
        assert profile.has_capability("nonexistent") is False

    def test_vendor_profile_unsupported(self):
        profile = VendorProfile(
            vendor_name="Test",
            capabilities=[
                VendorCapability(
                    name="metadata_extraction",
                    supported=True,
                ),
                VendorCapability(
                    name="camera_extraction",
                    supported=False,
                ),
                VendorCapability(
                    name="recording_extraction",
                    supported=False,
                ),
            ],
        )
        unsupported = profile.unsupported_capabilities()
        assert "camera_extraction" in unsupported
        assert "recording_extraction" in unsupported
        assert "metadata_extraction" not in unsupported

    def test_vendor_profile_parser_class(self):
        profile = VendorProfile(
            vendor_name="Test",
            parser_class="TestParser",
        )
        assert profile.parser_class == "TestParser"


class TestVendorParserBaseCapabilities:
    def test_base_returns_detection_only(self):
        class DummyParser(VendorParser):
            vendor_name = "Dummy"
            def can_parse(self, image_path, filesystem_analysis):
                return False
            def parse(self, image_path, filesystem_analysis):
                pass

        parser = DummyParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "Dummy"
        assert len(caps.capabilities) == 2
        assert caps.has_capability(CAPABILITY_VENDOR_DETECTION) is True
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is False


class TestHikvisionCapabilities:
    def test_hikvision_has_expected_capabilities(self):
        from app.parsers.hikvision import HikvisionParser
        parser = HikvisionParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "Hikvision"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True

    def test_hikvision_camera_pattern(self):
        from app.parsers.hikvision import HikvisionParser
        pattern = HikvisionParser.CAMERA_PATTERN
        assert pattern.match("CAM01") is not None
        assert pattern.match("CAMERA01") is not None
        assert pattern.match("ch01") is not None
        assert pattern.match("channel01") is not None
        assert pattern.match("random_dir") is None


class TestDahuaCapabilities:
    def test_dahua_has_expected_capabilities(self):
        from app.parsers.dahua import DahuaParser
        parser = DahuaParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "Dahua"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


class TestCPPlusCapabilities:
    def test_cp_plus_has_expected_capabilities(self):
        from plugins.cp_plus import CPPlusParser
        parser = CPPlusParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "CP Plus"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


class TestUniviewCapabilities:
    def test_uniview_has_expected_capabilities(self):
        from plugins.uniview import UniviewParser
        parser = UniviewParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "Uniview"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


class TestHoneywellCapabilities:
    def test_honeywell_has_expected_capabilities(self):
        from plugins.honeywell import HoneywellParser
        parser = HoneywellParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "Honeywell"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


class TestTPLinkCapabilities:
    def test_tp_link_has_expected_capabilities(self):
        from plugins.tp_link import TPLinkParser
        parser = TPLinkParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "TP-Link"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is True


class TestGodrejCapabilities:
    def test_godrej_has_honest_capabilities(self):
        from plugins.godrej import GodrejParser
        parser = GodrejParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "Godrej"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is False

    def test_godrej_unsupported_count(self):
        from plugins.godrej import GodrejParser
        parser = GodrejParser()
        caps = parser.get_capabilities()
        unsupported = caps.unsupported_capabilities()
        assert len(unsupported) == 4


class TestMatrixCapabilities:
    def test_matrix_has_honest_capabilities(self):
        from plugins.matrix import MatrixParser
        parser = MatrixParser()
        caps = parser.get_capabilities()
        assert caps.vendor_name == "Matrix"
        assert caps.has_capability(CAPABILITY_METADATA_EXTRACTION) is True
        assert caps.has_capability(CAPABILITY_CAMERA_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_RECORDING_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMESTAMP_EXTRACTION) is False
        assert caps.has_capability(CAPABILITY_TIMEZONE_EXTRACTION) is False

    def test_matrix_unsupported_count(self):
        from plugins.matrix import MatrixParser
        parser = MatrixParser()
        caps = parser.get_capabilities()
        unsupported = caps.unsupported_capabilities()
        assert len(unsupported) == 4


class TestPluginDiscovery:
    def test_default_registry_loads_builtins(self):
        registry = create_default_registry()
        parser_names = [
            p.vendor_name for p in registry._parsers
        ]
        assert "Hikvision" in parser_names
        assert "Dahua" in parser_names

    def test_plugin_discovery_loads_all_six(self):
        registry = create_default_registry()
        plugins_directory = (
            Path(__file__).parent.parent / "plugins"
        )
        result = registry.discover_plugins(plugins_directory)
        loaded = result["loaded_plugins"]
        assert "CP Plus" in loaded
        assert "Uniview" in loaded
        assert "Honeywell" in loaded
        assert "TP-Link" in loaded
        assert "Godrej" in loaded
        assert "Matrix" in loaded

    def test_plugin_discovery_no_errors(self):
        registry = create_default_registry()
        plugins_directory = (
            Path(__file__).parent.parent / "plugins"
        )
        result = registry.discover_plugins(plugins_directory)
        assert len(result["plugin_errors"]) == 0


class TestVendorDetectorMarkers:
    def test_all_eight_vendors_have_markers(self):
        from app.services.vendor_detector import VENDOR_MARKERS
        expected = {
            "hikvision", "dahua", "cp_plus",
            "uniview", "honeywell", "tp_link",
            "godrej", "matrix",
        }
        assert set(VENDOR_MARKERS.keys()) == expected

    def test_all_eight_have_display_names(self):
        from app.services.vendor_detector import VENDOR_DISPLAY_NAMES
        expected = {
            "hikvision", "dahua", "cp_plus",
            "uniview", "honeywell", "tp_link",
            "godrej", "matrix",
        }
        assert set(VENDOR_DISPLAY_NAMES.keys()) == expected


class TestGodrejDetection:
    def test_godrej_marker_in_filename(self):
        from app.services.vendor_detector import score_vendor
        sources = [("filename", "godrej_nvr_config.conf")]
        score, evidence = score_vendor("godrej", sources)
        assert score > 0
        assert len(evidence) > 0

    def test_godrej_marker_in_config(self):
        from app.services.vendor_detector import score_vendor
        sources = [("configuration", "manufacturer=Godrej Security")]
        score, _evidence = score_vendor("godrej", sources)
        assert score >= 0.50


class TestMatrixDetection:
    def test_matrix_marker_in_filename(self):
        from app.services.vendor_detector import score_vendor
        sources = [("filename", "matrix_visionpro_config.xml")]
        score, _evidence = score_vendor("matrix", sources)
        assert score > 0

    def test_matrix_marker_in_config(self):
        from app.services.vendor_detector import score_vendor
        sources = [("configuration", "device=Matrix Comsec SARV")]
        score, _evidence = score_vendor("matrix", sources)
        assert score >= 0.50


class TestFalsePositiveResistance:
    def test_generic_text_does_not_match_vendor(self):
        from app.services.vendor_detector import score_vendor
        sources = [
            ("configuration", "This is a generic device\nchannels=8"),
        ]
        for vendor in ["hikvision", "dahua", "cp_plus", "godrej", "matrix"]:
            score, _evidence = score_vendor(vendor, sources)
            assert score == 0.0

    def test_weak_signal_returns_unknown(self):
        from pathlib import Path
        import tempfile

        from app.services.vendor_detector import detect_vendor

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_dir = Path(tmpdir) / "recordings"
            evidence_dir.mkdir()
            (evidence_dir / "CAM01").mkdir()

            filesystem_analysis = {
                "files": [
                    {
                        "name": "random_hikvision_mention.txt",
                        "type": "file",
                        "inode": 1,
                    },
                ],
                "filesystem": {"code": "ntfs", "name": "NTFS"},
                "partition": {"start_sector": 0},
            }

            result = detect_vendor(
                Path(tmpdir),
                "forensic-image/raw",
                filesystem_analysis,
            )
            assert result.vendor == "unknown"


class TestCapabilityVocabularyEnforcement:
    def test_all_parser_capabilities_use_canonical_names(self):
        from app.parsers.dahua import DahuaParser
        from app.parsers.hikvision import HikvisionParser
        from plugins.cp_plus import CPPlusParser
        from plugins.godrej import GodrejParser
        from plugins.honeywell import HoneywellParser
        from plugins.matrix import MatrixParser
        from plugins.tp_link import TPLinkParser
        from plugins.uniview import UniviewParser

        parsers = [
            HikvisionParser(),
            DahuaParser(),
            CPPlusParser(),
            UniviewParser(),
            HoneywellParser(),
            TPLinkParser(),
            GodrejParser(),
            MatrixParser(),
        ]

        for parser in parsers:
            caps = parser.get_capabilities()
            for cap in caps.capabilities:
                assert cap.name in ALL_CANONICAL_CAPABILITIES, (
                    f"{parser.vendor_name} uses non-canonical "
                    f"capability: {cap.name}"
                )


class TestGenericForensicFallback:
    def test_unknown_vendor_still_has_dvr_structure(self):
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

from dataclasses import dataclass, field

CAPABILITY_VENDOR_DETECTION = "vendor_detection"
CAPABILITY_METADATA_EXTRACTION = "metadata_extraction"
CAPABILITY_CAMERA_EXTRACTION = "camera_extraction"
CAPABILITY_RECORDING_EXTRACTION = "recording_extraction"
CAPABILITY_TIMESTAMP_EXTRACTION = "timestamp_extraction"
CAPABILITY_TIMEZONE_EXTRACTION = "timezone_extraction"
CAPABILITY_FILESYSTEM_ANALYSIS = "filesystem_analysis"
CAPABILITY_DELETED_RECOVERY = "deleted_recovery"

ALL_CANONICAL_CAPABILITIES = {
    CAPABILITY_VENDOR_DETECTION,
    CAPABILITY_METADATA_EXTRACTION,
    CAPABILITY_CAMERA_EXTRACTION,
    CAPABILITY_RECORDING_EXTRACTION,
    CAPABILITY_TIMESTAMP_EXTRACTION,
    CAPABILITY_TIMEZONE_EXTRACTION,
    CAPABILITY_FILESYSTEM_ANALYSIS,
    CAPABILITY_DELETED_RECOVERY,
}


@dataclass
class VendorCapability:
    name: str
    supported: bool
    detail: str | None = None


@dataclass
class VendorProfile:
    vendor_name: str
    capabilities: list[VendorCapability] = field(
        default_factory=list
    )
    parser_class: str | None = None

    def has_capability(self, name: str) -> bool:
        for cap in self.capabilities:
            if cap.name == name and cap.supported:
                return True
        return False

    def unsupported_capabilities(self) -> list[str]:
        return [
            cap.name
            for cap in self.capabilities
            if not cap.supported
        ]

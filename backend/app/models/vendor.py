from dataclasses import dataclass, field


@dataclass
class VendorDetectionResult:
    vendor: str
    confidence: float
    detection_method: str
    evidence: list[str] = field(default_factory=list)
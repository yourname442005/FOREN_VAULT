from dataclasses import dataclass, field

NORMALIZED = "NORMALIZED"
TIMEZONE_UNKNOWN = "TIMEZONE_UNKNOWN"
INVALID = "INVALID"
AMBIGUOUS = "AMBIGUOUS"


@dataclass
class TimestampResult:
    original: str
    iso_naive: str | None = None
    iso_aware: str | None = None
    utc: str | None = None
    timezone: str | None = None
    timezone_source: str | None = None
    normalization_status: str = TIMEZONE_UNKNOWN
    format_used: str | None = None


@dataclass
class Camera:
    camera_id: str
    name: str | None = None
    source: str | None = None


@dataclass
class Recording:
    filename: str
    camera_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    format: str | None = None
    deleted: bool = False
    start_timestamp: TimestampResult | None = None
    end_timestamp: TimestampResult | None = None


@dataclass
class DVREvidence:
    vendor: str
    model: str | None = None
    firmware: str | None = None
    timezone: str | None = None
    timezone_source: str | None = None

    cameras: list[Camera] = field(
        default_factory=list
    )

    recordings: list[Recording] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )

    parser: str | None = None

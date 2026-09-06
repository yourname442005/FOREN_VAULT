from dataclasses import dataclass, field


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


@dataclass
class DVREvidence:
    vendor: str
    model: str | None = None
    firmware: str | None = None

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

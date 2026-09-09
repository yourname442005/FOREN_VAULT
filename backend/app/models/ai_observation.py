from dataclasses import dataclass, field


MOTION = "motion"
OBJECT_DETECTION = "object_detection"
FACE_DETECTION = "face_detection"

NOT_REQUESTED = "NOT_REQUESTED"
NOT_AVAILABLE = "NOT_AVAILABLE"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
PARTIAL = "PARTIAL"
FAILED = "FAILED"

VALID_ANALYSIS_TYPES = {MOTION, OBJECT_DETECTION, FACE_DETECTION}


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class TimeMapping:
    media_offset_seconds: float | None = None
    source_recording_start: str | None = None
    normalized_timestamp: str | None = None
    timezone: str | None = None
    mapping_method: str | None = None
    mapping_status: str = "MAPPING_UNCERTAIN"


@dataclass
class AIObservation:
    observation_id: str
    evidence_id: str | None = None
    recording_filename: str | None = None
    camera_id: str | None = None
    analysis_type: str = ""
    label: str | None = None
    confidence: float | None = None
    frame_number: int | None = None
    media_timestamp: str | None = None
    time_mapping: TimeMapping | None = None
    bounding_box: BoundingBox | None = None
    source_media_path: str | None = None
    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    status: str = "COMPLETED"
    uncertainty: str | None = None
    created_at: str | None = None

    def __post_init__(self):
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(
                    f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
                )
        if self.analysis_type and self.analysis_type not in VALID_ANALYSIS_TYPES:
            raise ValueError(
                f"Invalid analysis_type: {self.analysis_type}. "
                f"Must be one of: {VALID_ANALYSIS_TYPES}"
            )

    def to_dict(self) -> dict:
        result = {
            "observation_id": self.observation_id,
            "evidence_id": self.evidence_id,
            "recording_filename": self.recording_filename,
            "camera_id": self.camera_id,
            "analysis_type": self.analysis_type,
            "label": self.label,
            "confidence": self.confidence,
            "frame_number": self.frame_number,
            "media_timestamp": self.media_timestamp,
            "provider": self.provider,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status,
            "uncertainty": self.uncertainty,
            "created_at": self.created_at,
        }
        if self.time_mapping:
            result["time_mapping"] = {
                "media_offset_seconds": self.time_mapping.media_offset_seconds,
                "source_recording_start": self.time_mapping.source_recording_start,
                "normalized_timestamp": self.time_mapping.normalized_timestamp,
                "timezone": self.time_mapping.timezone,
                "mapping_method": self.time_mapping.mapping_method,
                "mapping_status": self.time_mapping.mapping_status,
            }
        if self.bounding_box:
            result["bounding_box"] = {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            }
        return result


@dataclass
class AIAnalysisResult:
    evidence_id: str | None = None
    recording_filename: str | None = None
    camera_id: str | None = None
    analysis_type: str = ""
    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    status: str = NOT_REQUESTED
    observations: list[AIObservation] = field(default_factory=list)
    observation_count: int = 0
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    analysis_duration_seconds: float | None = None
    uncertainty_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "recording_filename": self.recording_filename,
            "camera_id": self.camera_id,
            "analysis_type": self.analysis_type,
            "provider": self.provider,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status,
            "observation_count": self.observation_count,
            "observations": [obs.to_dict() for obs in self.observations],
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "analysis_duration_seconds": self.analysis_duration_seconds,
            "uncertainty_notes": self.uncertainty_notes,
        }


@dataclass
class AICapability:
    analysis_type: str
    available: bool
    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    reason: str | None = None


@dataclass
class AICapabilitiesResult:
    motion: AICapability | None = None
    object_detection: AICapability | None = None
    face_detection: AICapability | None = None

    def to_dict(self) -> dict:
        result = {}
        for name, cap in [("motion", self.motion), ("object_detection", self.object_detection), ("face_detection", self.face_detection)]:
            if cap:
                result[name] = {
                    "available": cap.available,
                    "provider": cap.provider,
                    "model_name": cap.model_name,
                    "model_version": cap.model_version,
                    "reason": cap.reason,
                }
            else:
                result[name] = {"available": False, "reason": "not_configured"}
        return result

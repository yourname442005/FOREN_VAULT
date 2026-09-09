from dataclasses import dataclass, field


@dataclass
class EvidenceSource:
    evidence_id: str
    filename: str | None = None
    source_image: str | None = None
    sha256: str | None = None
    vendor: str | None = None


@dataclass
class CameraInvestigation:
    camera_id: str
    name: str | None = None
    source: str | None = None
    evidence_id: str | None = None
    recording_count: int = 0
    deleted_count: int = 0
    recovered_count: int = 0


@dataclass
class RecordingLookup:
    filename: str
    camera_id: str | None = None
    evidence_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    deleted: bool = False
    recovered: bool = False
    validated: bool = False
    vendor: str | None = None
    format: str | None = None
    timestamp_status: str | None = None


@dataclass
class TemporalCorrelation:
    correlation_id: str
    camera_ids: list[str] = field(default_factory=list)
    recording_references: list[dict] = field(default_factory=list)
    overlap_start: str | None = None
    overlap_end: str | None = None
    timezone_comparable: bool = False
    evidence_ids: list[str] = field(default_factory=list)
    confidence: str | None = None


@dataclass
class TemporalCluster:
    cluster_id: str
    recording_references: list[dict] = field(default_factory=list)
    camera_ids: list[str] = field(default_factory=list)
    cluster_start: str | None = None
    cluster_end: str | None = None
    timezone_comparable: bool = False
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class CoverageInterval:
    start: str
    end: str
    status: str = "covered"
    recording_filename: str | None = None
    camera_id: str | None = None
    evidence_id: str | None = None


@dataclass
class CameraCoverage:
    camera_id: str
    evidence_id: str | None = None
    covered_intervals: list[CoverageInterval] = field(default_factory=list)
    gap_intervals: list[CoverageInterval] = field(default_factory=list)
    uncertain_intervals: list[CoverageInterval] = field(default_factory=list)
    total_covered_seconds: float = 0.0
    total_gap_seconds: float = 0.0


@dataclass
class RecoveryIntelligence:
    recovery_id: str
    evidence_id: str | None = None
    method: str | None = None
    image_offset: int | None = None
    size: int = 0
    sha256: str | None = None
    validation_status: str | None = None
    boundary_confidence: str | None = None
    exact_boundary: bool = False
    camera_id: str | None = None
    filename: str | None = None
    is_active_recording: bool = False
    is_recovered_candidate: bool = False
    is_validated_candidate: bool = False


@dataclass
class InvestigativeResult:
    query_type: str
    evidence_ids: list[str] = field(default_factory=list)
    recordings: list[RecordingLookup] = field(default_factory=list)
    cameras: list[CameraInvestigation] = field(default_factory=list)
    correlations: list[TemporalCorrelation] = field(default_factory=list)
    clusters: list[TemporalCluster] = field(default_factory=list)
    coverage: list[CameraCoverage] = field(default_factory=list)
    recovery_items: list[RecoveryIntelligence] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)

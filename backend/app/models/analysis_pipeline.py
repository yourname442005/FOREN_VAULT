from dataclasses import dataclass, field


NOT_ATTEMPTED = "NOT_ATTEMPTED"
COMPLETED = "COMPLETED"
PARTIAL = "PARTIAL"
FAILED = "FAILED"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class AnalysisStage:
    name: str
    status: str = NOT_ATTEMPTED
    error: str | None = None
    details: dict = field(default_factory=dict)

    def mark_completed(self, details: dict | None = None):
        self.status = COMPLETED
        if details:
            self.details.update(details)

    def mark_failed(self, error: str):
        self.status = FAILED
        self.error = error

    def mark_not_applicable(self, reason: str = ""):
        self.status = NOT_APPLICABLE
        if reason:
            self.details["reason"] = reason

    def mark_partial(self, error: str | None = None, details: dict | None = None):
        self.status = PARTIAL
        if error:
            self.error = error
        if details:
            self.details.update(details)

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "status": self.status,
        }
        if self.error:
            result["error"] = self.error
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class AnalysisPipeline:
    evidence_id: str
    stages: dict[str, AnalysisStage] = field(default_factory=dict)

    STAGE_NAMES = (
        "evidence_detected",
        "evidence_classified",
        "filesystem_analyzed",
        "dvr_structure_detected",
        "vendor_detected",
        "vendor_parser_selected",
        "metadata_extracted",
        "cameras_extracted",
        "recordings_extracted",
        "timestamps_normalized",
        "deleted_evidence_recovered",
        "recovery_validated",
        "cross_camera_correlation",
        "chain_of_custody_recorded",
        "report_generated",
    )

    def __post_init__(self):
        for name in self.STAGE_NAMES:
            if name not in self.stages:
                self.stages[name] = AnalysisStage(name=name)

    def get_stage(self, name: str) -> AnalysisStage:
        if name not in self.stages:
            self.stages[name] = AnalysisStage(name=name)
        return self.stages[name]

    def mark_completed(self, stage_name: str, details: dict | None = None):
        self.get_stage(stage_name).mark_completed(details)

    def mark_failed(self, stage_name: str, error: str):
        self.get_stage(stage_name).mark_failed(error)

    def mark_not_applicable(self, stage_name: str, reason: str = ""):
        self.get_stage(stage_name).mark_not_applicable(reason)

    def mark_partial(self, stage_name: str, error: str | None = None, details: dict | None = None):
        self.get_stage(stage_name).mark_partial(error, details)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "stages": {
                name: stage.to_dict()
                for name, stage in self.stages.items()
            },
        }

    def summary(self) -> dict:
        completed = sum(1 for s in self.stages.values() if s.status == COMPLETED)
        failed = sum(1 for s in self.stages.values() if s.status == FAILED)
        partial = sum(1 for s in self.stages.values() if s.status == PARTIAL)
        not_applicable = sum(1 for s in self.stages.values() if s.status == NOT_APPLICABLE)
        not_attempted = sum(1 for s in self.stages.values() if s.status == NOT_ATTEMPTED)

        return {
            "total_stages": len(self.STAGE_NAMES),
            "completed": completed,
            "failed": failed,
            "partial": partial,
            "not_applicable": not_applicable,
            "not_attempted": not_attempted,
        }

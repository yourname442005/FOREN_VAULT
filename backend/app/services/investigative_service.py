from datetime import datetime
from uuid import uuid4

from app.models.dvr_evidence import Recording
from app.models.investigative import (
    CameraCoverage,
    CameraInvestigation,
    CoverageInterval,
    EvidenceSource,
    InvestigativeResult,
    RecoveryIntelligence,
    RecordingLookup,
    TemporalCluster,
    TemporalCorrelation,
)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _intervals_overlap(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> bool:
    return a_start <= b_end and b_start <= a_end


def _compute_overlap(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> tuple[datetime, datetime] | None:
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_start <= overlap_end:
        return (overlap_start, overlap_end)
    return None


def _timestamps_comparable(
    rec_a: Recording,
    rec_b: Recording,
) -> bool:
    a_has_tz = rec_a.start_timestamp and rec_a.start_timestamp.iso_aware
    b_has_tz = rec_b.start_timestamp and rec_b.start_timestamp.iso_aware

    if a_has_tz and b_has_tz:
        return True

    a_has_meta = rec_a.start_timestamp is not None
    b_has_meta = rec_b.start_timestamp is not None

    if not a_has_meta and not b_has_meta:
        return True

    a_naive = (
        rec_a.start_timestamp
        and rec_a.start_timestamp.normalization_status == "TIMEZONE_UNKNOWN"
    )
    b_naive = (
        rec_b.start_timestamp
        and rec_b.start_timestamp.normalization_status == "TIMEZONE_UNKNOWN"
    )

    return bool(a_naive and b_naive)


class InvestigativeIndex:
    def __init__(self):
        self._evidence_sources: dict[str, EvidenceSource] = {}
        self._recordings: list[Recording] = []
        self._by_camera: dict[str, list[int]] = {}
        self._by_evidence: dict[str, list[int]] = {}
        self._deleted_indices: list[int] = []
        self._recovered_indices: list[int] = []

    def add_evidence(
        self,
        evidence_id: str,
        filename: str | None = None,
        source_image: str | None = None,
        sha256: str | None = None,
        vendor: str | None = None,
    ) -> None:
        self._evidence_sources[evidence_id] = EvidenceSource(
            evidence_id=evidence_id,
            filename=filename,
            source_image=source_image,
            sha256=sha256,
            vendor=vendor,
        )

    def add_recordings(
        self,
        recordings: list[Recording],
        evidence_id: str | None = None,
        vendor: str | None = None,
    ) -> None:
        for rec in recordings:
            if evidence_id and not rec.evidence_id:
                rec.evidence_id = evidence_id
            if vendor and not rec.vendor:
                rec.vendor = vendor

            idx = len(self._recordings)
            self._recordings.append(rec)

            if rec.camera_id:
                self._by_camera.setdefault(rec.camera_id, []).append(idx)

            if rec.evidence_id:
                self._by_evidence.setdefault(rec.evidence_id, []).append(idx)

            if rec.deleted:
                self._deleted_indices.append(idx)

            if rec.recovered:
                self._recovered_indices.append(idx)

    def all_recordings(self) -> list[Recording]:
        return list(self._recordings)

    def recordings_by_camera(self, camera_id: str) -> list[Recording]:
        indices = self._by_camera.get(camera_id, [])
        return [self._recordings[i] for i in indices]

    def recordings_by_evidence(self, evidence_id: str) -> list[Recording]:
        indices = self._by_evidence.get(evidence_id, [])
        return [self._recordings[i] for i in indices]

    def deleted_recordings(self) -> list[Recording]:
        return [self._recordings[i] for i in self._deleted_indices]

    def recovered_recordings(self) -> list[Recording]:
        return [self._recordings[i] for i in self._recovered_indices]

    def recordings_with_uncertain_timestamps(self) -> list[Recording]:
        result = []
        for rec in self._recordings:
            if not rec.start_timestamp:
                result.append(rec)
                continue
            ts = rec.start_timestamp
            if ts.normalization_status in ("TIMEZONE_UNKNOWN", "INVALID", "AMBIGUOUS"):
                result.append(rec)
        return result

    def recordings_in_time_window(
        self,
        start: datetime,
        end: datetime,
    ) -> list[Recording]:
        result = []
        for rec in self._recordings:
            rec_start = _parse_dt(rec.start_time)
            rec_end = _parse_dt(rec.end_time)
            if rec_start is None or rec_end is None:
                continue
            if _intervals_overlap(rec_start, rec_end, start, end):
                result.append(rec)
        return result

    def recordings_at_timestamp(
        self,
        timestamp: datetime,
    ) -> list[Recording]:
        result = []
        for rec in self._recordings:
            rec_start = _parse_dt(rec.start_time)
            rec_end = _parse_dt(rec.end_time)
            if rec_start is None or rec_end is None:
                continue
            if rec_start <= timestamp <= rec_end:
                result.append(rec)
        return result

    def get_evidence_source(self, evidence_id: str) -> EvidenceSource | None:
        return self._evidence_sources.get(evidence_id)

    def all_cameras(self) -> dict[str, list[Recording]]:
        return {
            cam: [self._recordings[i] for i in indices]
            for cam, indices in sorted(self._by_camera.items())
        }


def build_index(
    evidence_id: str,
    recordings: list[Recording],
    vendor: str | None = None,
    filename: str | None = None,
    source_image: str | None = None,
    sha256: str | None = None,
) -> InvestigativeIndex:
    index = InvestigativeIndex()
    index.add_evidence(
        evidence_id=evidence_id,
        filename=filename,
        source_image=source_image,
        sha256=sha256,
        vendor=vendor,
    )
    index.add_recordings(recordings, evidence_id=evidence_id, vendor=vendor)
    return index


def merge_indexes(*indexes: InvestigativeIndex) -> InvestigativeIndex:
    merged = InvestigativeIndex()
    for idx in indexes:
        for ev_id, src in idx._evidence_sources.items():
            merged._evidence_sources[ev_id] = src
        for rec in idx._recordings:
            new_idx = len(merged._recordings)
            merged._recordings.append(rec)
            if rec.camera_id:
                merged._by_camera.setdefault(rec.camera_id, []).append(new_idx)
            if rec.evidence_id:
                merged._by_evidence.setdefault(rec.evidence_id, []).append(new_idx)
            if rec.deleted:
                merged._deleted_indices.append(new_idx)
            if rec.recovered:
                merged._recovered_indices.append(new_idx)
    return merged


def query_recordings(
    index: InvestigativeIndex,
    camera_id: str | None = None,
    evidence_id: str | None = None,
    deleted: bool | None = None,
    recovered: bool | None = None,
    vendor: str | None = None,
) -> list[RecordingLookup]:
    if camera_id:
        recordings = index.recordings_by_camera(camera_id)
    elif evidence_id:
        recordings = index.recordings_by_evidence(evidence_id)
    else:
        recordings = index.all_recordings()

    results = []
    for rec in recordings:
        if deleted is not None and rec.deleted != deleted:
            continue
        if recovered is not None and rec.recovered != recovered:
            continue
        if vendor and rec.vendor != vendor:
            continue

        ts_status = None
        if rec.start_timestamp:
            ts_status = rec.start_timestamp.normalization_status

        results.append(RecordingLookup(
            filename=rec.filename,
            camera_id=rec.camera_id,
            evidence_id=rec.evidence_id,
            start_time=rec.start_time,
            end_time=rec.end_time,
            deleted=rec.deleted,
            recovered=rec.recovered,
            validated=rec.recovery_status == "VALIDATED_CANDIDATE",
            vendor=rec.vendor,
            format=rec.format,
            timestamp_status=ts_status,
        ))

    results.sort(key=lambda r: r.start_time or "")
    return results


def query_time_window(
    index: InvestigativeIndex,
    start: datetime,
    end: datetime,
) -> InvestigativeResult:
    recordings = index.recordings_in_time_window(start, end)
    lookups = []
    uncertainty_notes = []

    for rec in recordings:
        ts_status = None
        if rec.start_timestamp:
            ts_status = rec.start_timestamp.normalization_status

        lookups.append(RecordingLookup(
            filename=rec.filename,
            camera_id=rec.camera_id,
            evidence_id=rec.evidence_id,
            start_time=rec.start_time,
            end_time=rec.end_time,
            deleted=rec.deleted,
            recovered=rec.recovered,
            validated=rec.recovery_status == "VALIDATED_CANDIDATE",
            vendor=rec.vendor,
            format=rec.format,
            timestamp_status=ts_status,
        ))

        if ts_status in ("TIMEZONE_UNKNOWN", "INVALID", "AMBIGUOUS"):
            uncertainty_notes.append(
                f"Recording {rec.filename} has uncertain timestamp status: {ts_status}"
            )

    lookups.sort(key=lambda r: r.start_time or "")

    evidence_ids = sorted({r.evidence_id for r in recordings if r.evidence_id})

    return InvestigativeResult(
        query_type="time_window",
        evidence_ids=evidence_ids,
        recordings=lookups,
        uncertainty_notes=uncertainty_notes,
    )


def query_point_in_time(
    index: InvestigativeIndex,
    timestamp: datetime,
) -> InvestigativeResult:
    recordings = index.recordings_at_timestamp(timestamp)
    lookups = []
    cameras = []
    uncertainty_notes = []

    seen_cameras: set[str] = set()

    for rec in recordings:
        ts_status = None
        if rec.start_timestamp:
            ts_status = rec.start_timestamp.normalization_status

        lookups.append(RecordingLookup(
            filename=rec.filename,
            camera_id=rec.camera_id,
            evidence_id=rec.evidence_id,
            start_time=rec.start_time,
            end_time=rec.end_time,
            deleted=rec.deleted,
            recovered=rec.recovered,
            vendor=rec.vendor,
            format=rec.format,
            timestamp_status=ts_status,
        ))

        if rec.camera_id and rec.camera_id not in seen_cameras:
            seen_cameras.add(rec.camera_id)
            cameras.append(CameraInvestigation(
                camera_id=rec.camera_id,
                evidence_id=rec.evidence_id,
                recording_count=1,
            ))

        if ts_status in ("TIMEZONE_UNKNOWN", "INVALID", "AMBIGUOUS"):
            uncertainty_notes.append(
                f"Recording {rec.filename} has uncertain timestamp: {ts_status}"
            )

    evidence_ids = sorted({r.evidence_id for r in recordings if r.evidence_id})

    return InvestigativeResult(
        query_type="point_in_time",
        evidence_ids=evidence_ids,
        recordings=lookups,
        cameras=cameras,
        uncertainty_notes=uncertainty_notes,
    )


def find_temporal_correlations(
    index: InvestigativeIndex,
) -> list[TemporalCorrelation]:
    recordings = index.all_recordings()
    correlations = []

    valid = [
        r for r in recordings
        if r.start_time and r.end_time and r.camera_id
    ]

    for i, first in enumerate(valid):
        for second in valid[i + 1:]:
            if first.camera_id == second.camera_id:
                continue

            fs = _parse_dt(first.start_time)
            fe = _parse_dt(first.end_time)
            ss = _parse_dt(second.start_time)
            se = _parse_dt(second.end_time)

            if not all([fs, fe, ss, se]):
                continue

            if not _intervals_overlap(fs, fe, ss, se):
                continue

            if first.start_timestamp and second.start_timestamp:
                if first.start_timestamp.iso_aware and not second.start_timestamp.iso_aware:
                    continue
                if not first.start_timestamp.iso_aware and second.start_timestamp.iso_aware:
                    continue

            overlap = _compute_overlap(fs, fe, ss, se)
            if not overlap:
                continue

            tz_comparable = _timestamps_comparable(first, second)

            evidence_ids = sorted({
                e for e in [first.evidence_id, second.evidence_id] if e
            })

            correlations.append(TemporalCorrelation(
                correlation_id=f"CORR-{uuid4().hex[:8].upper()}",
                camera_ids=sorted({first.camera_id, second.camera_id}),
                recording_references=[
                    {"filename": first.filename, "camera_id": first.camera_id, "evidence_id": first.evidence_id},
                    {"filename": second.filename, "camera_id": second.camera_id, "evidence_id": second.evidence_id},
                ],
                overlap_start=overlap[0].isoformat(),
                overlap_end=overlap[1].isoformat(),
                timezone_comparable=tz_comparable,
                evidence_ids=evidence_ids,
                confidence="TEMPORAL_OVERLAP" if tz_comparable else "TEMPORAL_OVERLAP_UNCERTAIN_TZ",
            ))

    correlations.sort(key=lambda c: c.overlap_start or "")
    return correlations


def build_temporal_clusters(
    correlations: list[TemporalCorrelation],
) -> list[TemporalCluster]:
    if not correlations:
        return []

    events: list[tuple[str, int, str]] = []
    for corr in correlations:
        if corr.overlap_start and corr.overlap_end:
            events.append((corr.overlap_start, 1, corr.correlation_id))
            events.append((corr.overlap_end, -1, corr.correlation_id))

    events.sort(key=lambda e: (e[0], -e[1]))

    clusters: list[TemporalCluster] = []
    active: set[str] = set()
    cluster_start: str | None = None

    for ts, delta, corr_id in events:
        if delta == 1:
            if not active:
                cluster_start = ts
            active.add(corr_id)
        else:
            active.discard(corr_id)
            if not active and cluster_start:
                all_cameras: set[str] = set()
                all_recordings: list[dict] = []
                all_evidence: set[str] = set()
                all_tz_comparable = True

                for c in correlations:
                    if c.overlap_start and c.overlap_end:
                        c_start = _parse_dt(c.overlap_start)
                        c_end = _parse_dt(c.overlap_end)
                        cl_start = _parse_dt(cluster_start)
                        cl_end = _parse_dt(ts)
                        if c_start and c_end and cl_start and cl_end:
                            if c_start <= cl_end and c_end >= cl_start:
                                all_cameras.update(c.camera_ids)
                                all_recordings.extend(c.recording_references)
                                all_evidence.update(c.evidence_ids)
                                if not c.timezone_comparable:
                                    all_tz_comparable = False

                if all_cameras:
                    clusters.append(TemporalCluster(
                        cluster_id=f"CLUSTER-{uuid4().hex[:8].upper()}",
                        recording_references=all_recordings,
                        camera_ids=sorted(all_cameras),
                        cluster_start=cluster_start,
                        cluster_end=ts,
                        timezone_comparable=all_tz_comparable,
                        evidence_ids=sorted(all_evidence),
                    ))

                cluster_start = None

    if active and cluster_start:
        all_cameras: set[str] = set()
        all_recordings_list: list[dict] = []
        all_evidence: set[str] = set()
        all_tz_comparable = True

        for cid in active:
            for c in correlations:
                if c.correlation_id == cid:
                    all_cameras.update(c.camera_ids)
                    all_recordings_list.extend(c.recording_references)
                    all_evidence.update(c.evidence_ids)
                    if not c.timezone_comparable:
                        all_tz_comparable = False

        if all_cameras:
            clusters.append(TemporalCluster(
                cluster_id=f"CLUSTER-{uuid4().hex[:8].upper()}",
                recording_references=all_recordings_list,
                camera_ids=sorted(all_cameras),
                cluster_start=cluster_start,
                cluster_end=None,
                timezone_comparable=all_tz_comparable,
                evidence_ids=sorted(all_evidence),
            ))

    return clusters


def analyze_camera_coverage(
    index: InvestigativeIndex,
    camera_id: str,
    query_start: datetime | None = None,
    query_end: datetime | None = None,
) -> CameraCoverage:
    recordings = index.recordings_by_camera(camera_id)

    intervals: list[tuple[datetime, datetime, Recording]] = []
    for rec in recordings:
        rs = _parse_dt(rec.start_time)
        re_ = _parse_dt(rec.end_time)
        if rs and re_:
            intervals.append((rs, re_, rec))

    intervals.sort(key=lambda x: x[0])

    covered: list[CoverageInterval] = []
    gaps: list[CoverageInterval] = []
    uncertain: list[CoverageInterval] = []

    for rs, re_, rec in intervals:
        ts_status = None
        if rec.start_timestamp:
            ts_status = rec.start_timestamp.normalization_status

        ci = CoverageInterval(
            start=rs.isoformat(),
            end=re_.isoformat(),
            status="covered",
            recording_filename=rec.filename,
            camera_id=camera_id,
            evidence_id=rec.evidence_id,
        )

        if ts_status in ("TIMEZONE_UNKNOWN", "INVALID", "AMBIGUOUS"):
            ci.status = "uncertain"
            uncertain.append(ci)
        else:
            covered.append(ci)

    for i in range(len(intervals) - 1):
        _, prev_end, _ = intervals[i]
        next_start, _, _ = intervals[i + 1]

        if prev_end < next_start:
            gaps.append(CoverageInterval(
                start=prev_end.isoformat(),
                end=next_start.isoformat(),
                status="gap",
                camera_id=camera_id,
            ))

    total_covered = sum(
        (re_ - rs).total_seconds()
        for rs, re_, _ in intervals
    )
    total_gap = sum(
        (gaps[i].end and gaps[i].start and (datetime.fromisoformat(gaps[i].end) - datetime.fromisoformat(gaps[i].start)).total_seconds() or 0)
        for i in range(len(gaps))
        if gaps[i].start and gaps[i].end
    )

    evidence_ids = sorted({r.evidence_id for r in recordings if r.evidence_id})

    return CameraCoverage(
        camera_id=camera_id,
        evidence_id=evidence_ids[0] if evidence_ids else None,
        covered_intervals=covered,
        gap_intervals=gaps,
        uncertain_intervals=uncertain,
        total_covered_seconds=total_covered,
        total_gap_seconds=total_gap,
    )


def build_recovery_intelligence(
    recovery_results: list[dict],
    evidence_id: str | None = None,
) -> list[RecoveryIntelligence]:
    items = []
    for i, result in enumerate(recovery_results):
        boundary = result.get("boundary") or {}
        boundary_confidence = boundary.get("confidence")
        exact_boundary = boundary.get("exact_original_boundary_established", False)

        validation = result.get("validation") or {}
        validation_status = "UNKNOWN"
        if result.get("recovered"):
            if validation.get("codec"):
                validation_status = "VALIDATED"
            else:
                validation_status = "RECOVERED_UNVALIDATED"
        elif result.get("recovery_status") == "FAILED":
            validation_status = "FAILED"
        elif result.get("recovery_status") == "VALIDATED_CANDIDATE":
            validation_status = "VALIDATED_CANDIDATE"
        elif result.get("recovery_status") == "DUPLICATE_OF_EXISTING":
            validation_status = "DUPLICATE"

        is_active = not result.get("deleted", False) and not result.get("recovered", False)
        is_recovered = result.get("recovered", False) or result.get("recovery_status") == "VALIDATED_CANDIDATE"
        is_validated = result.get("recovery_status") == "VALIDATED_CANDIDATE" or (
            result.get("recovered") and validation.get("codec")
        )

        items.append(RecoveryIntelligence(
            recovery_id=f"REC-{i:04d}-{evidence_id or 'UNKNOWN'}",
            evidence_id=result.get("evidence_id", evidence_id),
            method=result.get("method"),
            image_offset=result.get("image_offset"),
            size=result.get("size", 0),
            sha256=result.get("sha256"),
            validation_status=validation_status,
            boundary_confidence=boundary_confidence,
            exact_boundary=exact_boundary,
            camera_id=result.get("camera_id"),
            filename=result.get("filename"),
            is_active_recording=is_active,
            is_recovered_candidate=is_recovered,
            is_validated_candidate=is_validated,
        ))

    return items


def build_provenance(
    index: InvestigativeIndex,
) -> list[dict]:
    provenance = []
    for ev_id, src in index._evidence_sources.items():
        provenance.append({
            "evidence_id": ev_id,
            "filename": src.filename,
            "source_image": src.source_image,
            "sha256": src.sha256,
            "vendor": src.vendor,
            "recording_count": len(index.recordings_by_evidence(ev_id)),
        })
    return provenance

from datetime import datetime

from app.models.dvr_evidence import Camera, Recording, TimestampResult, DVREvidence
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
from app.services.investigative_service import (
    InvestigativeIndex,
    analyze_camera_coverage,
    build_index,
    build_provenance,
    build_recovery_intelligence,
    build_temporal_clusters,
    find_temporal_correlations,
    merge_indexes,
    query_point_in_time,
    query_recordings,
    query_time_window,
)


# =============================================================================
# A. RECORDING INDEX CREATION
# =============================================================================


class TestRecordingIndex:
    def test_index_creation(self):
        index = InvestigativeIndex()
        assert index.all_recordings() == []

    def test_add_recordings(self):
        index = InvestigativeIndex()
        rec = Recording(filename="test.h264", camera_id="CAM01")
        index.add_recordings([rec], evidence_id="EV-001")
        assert len(index.all_recordings()) == 1

    def test_index_with_provenance(self):
        index = InvestigativeIndex()
        index.add_evidence(
            evidence_id="EV-001",
            filename="test.E01",
            sha256="abc123",
            vendor="Hikvision",
        )
        rec = Recording(
            filename="test.h264",
            camera_id="CAM01",
            evidence_id="EV-001",
        )
        index.add_recordings([rec])
        assert index.all_recordings()[0].evidence_id == "EV-001"


# =============================================================================
# B. LOOKUP BY CAMERA
# =============================================================================


class TestLookupByCamera:
    def test_lookup_by_camera(self):
        index = InvestigativeIndex()
        rec1 = Recording(filename="a.h264", camera_id="CAM01")
        rec2 = Recording(filename="b.h264", camera_id="CAM02")
        index.add_recordings([rec1, rec2])
        results = index.recordings_by_camera("CAM01")
        assert len(results) == 1
        assert results[0].filename == "a.h264"

    def test_lookup_nonexistent_camera(self):
        index = InvestigativeIndex()
        rec = Recording(filename="a.h264", camera_id="CAM01")
        index.add_recordings([rec])
        results = index.recordings_by_camera("CAM99")
        assert len(results) == 0


# =============================================================================
# C. LOOKUP BY EVIDENCE
# =============================================================================


class TestLookupByEvidence:
    def test_lookup_by_evidence(self):
        index = InvestigativeIndex()
        rec1 = Recording(filename="a.h264", evidence_id="EV-001")
        rec2 = Recording(filename="b.h264", evidence_id="EV-002")
        index.add_recordings([rec1, rec2])
        results = index.recordings_by_evidence("EV-001")
        assert len(results) == 1
        assert results[0].filename == "a.h264"


# =============================================================================
# D. EXACT TIMESTAMP QUERY
# =============================================================================


class TestExactTimestampQuery:
    def test_point_in_time(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec])
        ts = datetime(2026, 9, 3, 18, 30, 0)
        result = query_point_in_time(index, ts)
        assert len(result.recordings) == 1
        assert result.recordings[0].camera_id == "CAM01"

    def test_point_in_time_no_match(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec])
        ts = datetime(2026, 9, 3, 20, 0, 0)
        result = query_point_in_time(index, ts)
        assert len(result.recordings) == 0


# =============================================================================
# E. TIME-WINDOW QUERY
# =============================================================================


class TestTimeWindowQuery:
    def test_time_window_overlap(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec])
        start = datetime(2026, 9, 3, 18, 30, 0)
        end = datetime(2026, 9, 3, 18, 45, 0)
        result = query_time_window(index, start, end)
        assert len(result.recordings) == 1

    def test_time_window_no_overlap(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec])
        start = datetime(2026, 9, 3, 19, 30, 0)
        end = datetime(2026, 9, 3, 20, 0, 0)
        result = query_time_window(index, start, end)
        assert len(result.recordings) == 0


# =============================================================================
# F. EXACT-BOUNDARY OVERLAP
# =============================================================================


class TestExactBoundaryOverlap:
    def test_exact_boundary(self):
        index = InvestigativeIndex()
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec1, rec2])
        start = datetime(2026, 9, 3, 18, 0, 0)
        end = datetime(2026, 9, 3, 19, 0, 0)
        result = query_time_window(index, start, end)
        assert len(result.recordings) == 2


# =============================================================================
# G. PARTIAL OVERLAP
# =============================================================================


class TestPartialOverlap:
    def test_partial_overlap(self):
        index = InvestigativeIndex()
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T18:30:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:15:00",
            end_time="2026-09-03T18:45:00",
        )
        index.add_recordings([rec1, rec2])
        start = datetime(2026, 9, 3, 18, 10, 0)
        end = datetime(2026, 9, 3, 18, 20, 0)
        result = query_time_window(index, start, end)
        assert len(result.recordings) == 2


# =============================================================================
# H. CONTAINMENT
# =============================================================================


class TestContainment:
    def test_query_contained_by_recording(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec])
        start = datetime(2026, 9, 3, 18, 15, 0)
        end = datetime(2026, 9, 3, 18, 45, 0)
        result = query_time_window(index, start, end)
        assert len(result.recordings) == 1


# =============================================================================
# I. ADJACENT NON-OVERLAP
# =============================================================================


class TestAdjacentNonOverlap:
    def test_adjacent_non_overlap(self):
        index = InvestigativeIndex()
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T18:29:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:31:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec1, rec2])
        start = datetime(2026, 9, 3, 18, 0, 0)
        end = datetime(2026, 9, 3, 18, 30, 0)
        result = query_time_window(index, start, end)
        assert len(result.recordings) == 1


# =============================================================================
# J. MISSING TIMESTAMP
# =============================================================================


class TestMissingTimestamp:
    def test_missing_start_time(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time=None,
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec])
        ts = datetime(2026, 9, 3, 18, 30, 0)
        result = query_point_in_time(index, ts)
        assert len(result.recordings) == 0

    def test_missing_end_time(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time=None,
        )
        index.add_recordings([rec])
        ts = datetime(2026, 9, 3, 18, 30, 0)
        result = query_point_in_time(index, ts)
        assert len(result.recordings) == 0


# =============================================================================
# K. UNKNOWN TIMEZONE
# =============================================================================


class TestUnknownTimezone:
    def test_unknown_timezone(self):
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_timestamp=TimestampResult(
                original="2026-09-03 18:00:00",
                normalization_status="TIMEZONE_UNKNOWN",
            ),
        )
        index = InvestigativeIndex()
        index.add_recordings([rec])
        results = index.recordings_with_uncertain_timestamps()
        assert len(results) == 1


# =============================================================================
# L. MIXED TIMEZONE SAFETY
# =============================================================================


class TestMixedTimezoneSafety:
    def test_mixed_tz_correlation_skipped(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
            start_timestamp=TimestampResult(
                original="2026-09-03T18:00:00",
                iso_aware="2026-09-03T18:00:00+05:30",
                timezone="Asia/Kolkata",
            ),
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
            start_timestamp=TimestampResult(
                original="2026-09-03T18:00:00",
                normalization_status="TIMEZONE_UNKNOWN",
            ),
        )
        index = InvestigativeIndex()
        index.add_recordings([rec1, rec2])
        correlations = find_temporal_correlations(index)
        assert len(correlations) == 0


# =============================================================================
# M. TWO-CAMERA CORRELATION
# =============================================================================


class TestTwoCameraCorrelation:
    def test_two_camera_overlap(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:30:00",
            end_time="2026-09-03T19:30:00",
        )
        index = InvestigativeIndex()
        index.add_recordings([rec1, rec2])
        correlations = find_temporal_correlations(index)
        assert len(correlations) == 1
        assert correlations[0].overlap_start == "2026-09-03T18:30:00"
        assert correlations[0].overlap_end == "2026-09-03T19:00:00"
        assert "CAM01" in correlations[0].camera_ids
        assert "CAM02" in correlations[0].camera_ids


# =============================================================================
# N. MULTI-CAMERA CORRELATION
# =============================================================================


class TestMultiCameraCorrelation:
    def test_multi_camera(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        rec3 = Recording(
            filename="c.h264",
            camera_id="CAM03",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index = InvestigativeIndex()
        index.add_recordings([rec1, rec2, rec3])
        correlations = find_temporal_correlations(index)
        assert len(correlations) == 3


# =============================================================================
# O. TEMPORAL CLUSTER
# =============================================================================


class TestTemporalCluster:
    def test_cluster_building(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T18:30:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:10:00",
            end_time="2026-09-03T18:25:00",
        )
        index = InvestigativeIndex()
        index.add_recordings([rec1, rec2])
        correlations = find_temporal_correlations(index)
        clusters = build_temporal_clusters(correlations)
        assert len(clusters) >= 1
        assert clusters[0].cluster_start is not None
        assert clusters[0].cluster_end is not None


# =============================================================================
# P. CHAINED OVERLAP
# =============================================================================


class TestChainedOverlap:
    def test_chained_overlap(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T18:30:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-03T18:20:00",
            end_time="2026-09-03T18:40:00",
        )
        rec3 = Recording(
            filename="c.h264",
            camera_id="CAM03",
            start_time="2026-09-03T18:35:00",
            end_time="2026-09-03T19:00:00",
        )
        index = InvestigativeIndex()
        index.add_recordings([rec1, rec2, rec3])
        correlations = find_temporal_correlations(index)
        assert len(correlations) >= 2


# =============================================================================
# Q. COVERAGE INTERVALS
# =============================================================================


class TestCoverageIntervals:
    def test_coverage_intervals(self):
        index = InvestigativeIndex()
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T18:10:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:20:00",
            end_time="2026-09-03T18:30:00",
        )
        index.add_recordings([rec1, rec2])
        coverage = analyze_camera_coverage(index, "CAM01")
        assert len(coverage.covered_intervals) == 2
        assert len(coverage.gap_intervals) == 1
        assert coverage.gap_intervals[0].start == "2026-09-03T18:10:00"
        assert coverage.gap_intervals[0].end == "2026-09-03T18:20:00"


# =============================================================================
# R. COVERAGE GAPS
# =============================================================================


class TestCoverageGaps:
    def test_coverage_gap_detection(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T18:10:00",
        )
        index.add_recordings([rec])
        coverage = analyze_camera_coverage(index, "CAM01")
        assert len(coverage.covered_intervals) == 1
        assert len(coverage.gap_intervals) == 0


# =============================================================================
# S. DELETED RECORDING FILTERING
# =============================================================================


class TestDeletedRecordingFiltering:
    def test_deleted_recordings(self):
        index = InvestigativeIndex()
        rec1 = Recording(filename="a.h264", deleted=True)
        rec2 = Recording(filename="b.h264", deleted=False)
        index.add_recordings([rec1, rec2])
        deleted = index.deleted_recordings()
        assert len(deleted) == 1
        assert deleted[0].filename == "a.h264"


# =============================================================================
# T. RECOVERED CANDIDATE PROVENANCE
# =============================================================================


class TestRecoveredCandidateProvenance:
    def test_recovered_candidate(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="recovered.h264",
            recovered=True,
            recovery_status="VALIDATED_CANDIDATE",
            evidence_id="EV-001",
        )
        index.add_recordings([rec])
        recovered = index.recovered_recordings()
        assert len(recovered) == 1
        assert recovered[0].recovery_status == "VALIDATED_CANDIDATE"


# =============================================================================
# U. VALIDATED RECOVERED CANDIDATE
# =============================================================================


class TestValidatedRecoveredCandidate:
    def test_validated_candidate(self):
        recovery_results = [
            {
                "method": "contiguous_carving",
                "image_offset": 18796544,
                "size": 53186,
                "sha256": "5cb3e384c4107004bc77635ddbe80efcfa8797ee066d163a200b4b1a9649fc35",
                "recovered": True,
                "recovery_status": "VALIDATED_CANDIDATE",
                "validation": {"codec": "h264", "width": 320, "height": 240, "fps": 20},
                "boundary": {"confidence": "MEDIUM", "method": "sustained_zero_tail", "exact_original_boundary_established": False},
                "filename": "carved_001.h264",
            }
        ]
        items = build_recovery_intelligence(recovery_results, evidence_id="EV-001")
        assert len(items) == 1
        assert items[0].is_validated_candidate is True
        assert items[0].image_offset == 18796544
        assert items[0].sha256 == "5cb3e384c4107004bc77635ddbe80efcfa8797ee066d163a200b4b1a9649fc35"
        assert items[0].boundary_confidence == "MEDIUM"
        assert items[0].exact_boundary is False


# =============================================================================
# V. UNCERTAIN RECOVERY BOUNDARY
# =============================================================================


class TestUncertainRecoveryBoundary:
    def test_uncertain_boundary(self):
        recovery_results = [
            {
                "method": "inode_recovery",
                "size": 1000,
                "sha256": "abc123",
                "recovered": True,
                "recovery_status": "RECOVERED",
                "validation": {},
                "boundary": {"confidence": "LOW"},
                "filename": "inode_123.h264",
            }
        ]
        items = build_recovery_intelligence(recovery_results)
        assert items[0].boundary_confidence == "LOW"
        assert items[0].exact_boundary is False


# =============================================================================
# W. MULTI-EVIDENCE PROVENANCE
# =============================================================================


class TestMultiEvidenceProvenance:
    def test_multi_evidence(self):
        index1 = InvestigativeIndex()
        index1.add_evidence(evidence_id="EV-001", filename="img1.E01")
        rec1 = Recording(filename="a.h264", evidence_id="EV-001")
        index1.add_recordings([rec1])

        index2 = InvestigativeIndex()
        index2.add_evidence(evidence_id="EV-002", filename="img2.E01")
        rec2 = Recording(filename="b.h264", evidence_id="EV-002")
        index2.add_recordings([rec2])

        merged = merge_indexes(index1, index2)
        assert len(merged.all_recordings()) == 2
        provenance = build_provenance(merged)
        assert len(provenance) == 2
        evidence_ids = {p["evidence_id"] for p in provenance}
        assert "EV-001" in evidence_ids
        assert "EV-002" in evidence_ids


# =============================================================================
# X. DETERMINISTIC ORDERING
# =============================================================================


class TestDeterministicOrdering:
    def test_sorted_by_start_time(self):
        index = InvestigativeIndex()
        rec1 = Recording(
            filename="b.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:30:00",
            end_time="2026-09-03T19:00:00",
        )
        rec2 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T18:30:00",
        )
        index.add_recordings([rec1, rec2])
        start = datetime(2026, 9, 3, 17, 0, 0)
        end = datetime(2026, 9, 3, 20, 0, 0)
        result = query_time_window(index, start, end)
        assert result.recordings[0].filename == "a.h264"
        assert result.recordings[1].filename == "b.h264"


# =============================================================================
# Y. API RESPONSE SHAPE
# =============================================================================


class TestAPIResponseShape:
    def test_investigative_result_structure(self):
        result = InvestigativeResult(
            query_type="test",
            evidence_ids=["EV-001"],
            recordings=[],
            cameras=[],
            correlations=[],
            clusters=[],
            coverage=[],
            recovery_items=[],
            uncertainty_notes=[],
            provenance=[],
        )
        d = result.__dict__
        assert "query_type" in d
        assert "evidence_ids" in d
        assert "recordings" in d
        assert "cameras" in d
        assert "correlations" in d
        assert "clusters" in d
        assert "coverage" in d
        assert "recovery_items" in d
        assert "uncertainty_notes" in d
        assert "provenance" in d

    def test_recording_lookup_structure(self):
        lookup = RecordingLookup(
            filename="test.h264",
            camera_id="CAM01",
            evidence_id="EV-001",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        assert lookup.filename == "test.h264"
        assert lookup.camera_id == "CAM01"
        assert lookup.evidence_id == "EV-001"


# =============================================================================
# Z. REGRESSION AGAINST ALL EXISTING PHASE 1-4 FUNCTIONALITY
# =============================================================================


class TestPhaseRegression:
    def test_timestamp_result_model(self):
        ts = TimestampResult(
            original="2026-09-03 18:00:00",
            iso_naive="2026-09-03T18:00:00",
            timezone="UTC",
        )
        assert ts.original == "2026-09-03 18:00:00"
        assert ts.timezone == "UTC"

    def test_recording_model_with_timestamps(self):
        rec = Recording(
            filename="test.h264",
            start_timestamp=TimestampResult(original="2026-09-03 18:00:00"),
            end_timestamp=TimestampResult(original="2026-09-03 19:00:00"),
        )
        assert rec.start_timestamp.original == "2026-09-03 18:00:00"

    def test_dvr_evidence_with_timezone(self):
        evidence = DVREvidence(
            vendor="Hikvision",
            timezone="Asia/Kolkata",
            cameras=[Camera(camera_id="CAM01")],
            recordings=[Recording(filename="test.h264")],
        )
        assert evidence.vendor == "Hikvision"
        assert evidence.timezone == "Asia/Kolkata"

    def test_recording_model_backward_compatible(self):
        rec = Recording(filename="test.h264")
        assert rec.evidence_id is None
        assert rec.vendor is None
        assert rec.recovered is False
        assert rec.recovery_status is None

    def test_investigative_models_exist(self):
        assert EvidenceSource is not None
        assert CameraInvestigation is not None
        assert RecordingLookup is not None
        assert TemporalCorrelation is not None
        assert TemporalCluster is not None
        assert CoverageInterval is not None
        assert CameraCoverage is not None
        assert RecoveryIntelligence is not None
        assert InvestigativeResult is not None

    def test_investigative_service_exists(self):
        assert InvestigativeIndex is not None
        assert build_index is not None
        assert merge_indexes is not None
        assert query_recordings is not None
        assert query_time_window is not None
        assert query_point_in_time is not None
        assert find_temporal_correlations is not None
        assert build_temporal_clusters is not None
        assert analyze_camera_coverage is not None
        assert build_recovery_intelligence is not None
        assert build_provenance is not None

    def test_no_ai_capabilities(self):
        import app.models.investigative as mod
        import app.services.investigative_service as svc
        import inspect

        all_code = inspect.getsource(mod) + inspect.getsource(svc)
        forbidden = ["face", "object_detection", "person_detection", "embedding", "vector", "llm"]
        for term in forbidden:
            assert term.lower() not in all_code.lower(), f"AI term '{term}' found in investigative code"

    def test_no_unsupported_claims(self):
        import app.services.investigative_service as svc
        import inspect

        all_code = inspect.getsource(svc)
        forbidden = ["incident occurred", "person was present", "definitely the original"]
        for phrase in forbidden:
            assert phrase.lower() not in all_code.lower(), f"Unsupported claim '{phrase}' found"

    def test_timestamp_unchanged(self):
        from app.services.timestamp_parser import parse_timestamp
        ts = parse_timestamp("2026-09-03T18:00:00Z", None)
        assert ts.original == "2026-09-03T18:00:00Z"
        assert ts.utc == "2026-09-03T18:00:00+00:00"

    def test_vendor_detection_unchanged(self):
        from app.models.vendor import ALL_CANONICAL_CAPABILITIES
        assert len(ALL_CANONICAL_CAPABILITIES) == 8

    def test_chain_of_custody_unchanged(self):
        from app.services.chain_of_custody import record_custody_event, get_custody_history
        assert callable(record_custody_event)
        assert callable(get_custody_history)

    def test_evidence_model_unchanged(self):
        from app.models.analysis_pipeline import AnalysisPipeline, AnalysisStage
        assert AnalysisPipeline is not None
        assert AnalysisStage is not None

    def test_merge_preserves_provenance(self):
        index1 = InvestigativeIndex()
        index1.add_evidence(evidence_id="EV-A", filename="a.E01")
        rec1 = Recording(filename="a.h264", evidence_id="EV-A")
        index1.add_recordings([rec1])

        index2 = InvestigativeIndex()
        index2.add_evidence(evidence_id="EV-B", filename="b.E01")
        rec2 = Recording(filename="b.h264", evidence_id="EV-B")
        index2.add_recordings([rec2])

        merged = merge_indexes(index1, index2)
        provenance = build_provenance(merged)
        assert len(provenance) == 2

    def test_coverage_total_seconds(self):
        index = InvestigativeIndex()
        rec = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-03T18:00:00",
            end_time="2026-09-03T19:00:00",
        )
        index.add_recordings([rec])
        coverage = analyze_camera_coverage(index, "CAM01")
        assert coverage.total_covered_seconds == 3600.0

    def test_recovery_intelligence_structure(self):
        items = build_recovery_intelligence([])
        assert items == []

    def test_query_recordings_with_filters(self):
        index = InvestigativeIndex()
        rec1 = Recording(filename="a.h264", camera_id="CAM01", deleted=True, vendor="Hikvision")
        rec2 = Recording(filename="b.h264", camera_id="CAM02", deleted=False, vendor="Dahua")
        index.add_recordings([rec1, rec2])

        deleted_only = query_recordings(index, deleted=True)
        assert len(deleted_only) == 1

        vendor_filtered = query_recordings(index, vendor="Dahua")
        assert len(vendor_filtered) == 1

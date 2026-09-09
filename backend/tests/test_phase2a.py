import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.dvr_evidence import (
    INVALID,
    NORMALIZED,
    TIMEZONE_UNKNOWN,
    DVREvidence,
    Recording,
    TimestampResult,
)
from app.services.event_correlator import (
    _timestamps_are_comparable,
    correlate_recordings,
)
from app.services.recording_timeline import (
    build_timeline,
    recording_contains_time,
)
from app.services.timestamp_parser import (
    _detect_format,
    _is_valid_datetime,
    parse_recording_filename,
    parse_timestamp,
)
from app.services.timezone_extractor import (
    _resolve_timezone,
    extract_timezone_from_config,
    extract_timezone_from_json,
)


class TestTimestampResultModel:
    def test_timestamp_result_defaults(self):
        tr = TimestampResult(original="2026-01-01T00:00:00")
        assert tr.original == "2026-01-01T00:00:00"
        assert tr.iso_naive is None
        assert tr.iso_aware is None
        assert tr.utc is None
        assert tr.timezone is None
        assert tr.timezone_source is None
        assert tr.normalization_status == TIMEZONE_UNKNOWN
        assert tr.format_used is None

    def test_timestamp_result_with_all_fields(self):
        tr = TimestampResult(
            original="20260901_180000",
            iso_naive="2026-09-01T18:00:00",
            iso_aware="2026-09-01T18:00:00+05:30",
            utc="2026-09-01T12:30:00+00:00",
            timezone="Asia/Kolkata",
            timezone_source="device_config",
            normalization_status=NORMALIZED,
            format_used="%Y%m%d%H%M%S",
        )
        assert tr.normalization_status == NORMALIZED
        assert tr.timezone == "Asia/Kolkata"

    def test_recording_model_with_timestamps(self):
        ts = TimestampResult(original="20260901_180000")
        rec = Recording(
            filename="20260901_180000_190000_CAM01.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
            start_timestamp=ts,
            end_timestamp=ts,
        )
        assert rec.start_timestamp is not None
        assert rec.end_timestamp is not None
        assert rec.start_time == "2026-09-01T18:00:00"

    def test_recording_model_backward_compatible(self):
        rec = Recording(
            filename="test.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        assert rec.start_timestamp is None
        assert rec.end_timestamp is None
        assert rec.start_time == "2026-09-01T18:00:00"

    def test_dvr_evidence_with_timezone(self):
        evidence = DVREvidence(
            vendor="Hikvision",
            timezone="Asia/Kolkata",
            timezone_source="device_config",
        )
        assert evidence.timezone == "Asia/Kolkata"
        assert evidence.timezone_source == "device_config"

    def test_dvr_evidence_backward_compatible(self):
        evidence = DVREvidence(vendor="Dahua")
        assert evidence.timezone is None
        assert evidence.timezone_source is None


class TestParseRecordingFilename:
    def test_standard_hikvision_format(self):
        result = parse_recording_filename(
            "20260901_180000_190000_CAM01.h264"
        )
        assert result is not None
        assert result["camera_id"] == "CAM01"
        assert result["start_time"] == "2026-09-01T18:00:00"
        assert result["end_time"] == "2026-09-01T19:00:00"
        assert result["format"] == "h264"

    def test_camera_prefix_format(self):
        result = parse_recording_filename(
            "CAM01_20260901_180000_190000.dav"
        )
        assert result is not None
        assert result["camera_id"] == "CAM01"

    def test_datetime_separator_format(self):
        result = parse_recording_filename(
            "2026-09-01_18-00-00_19-00-00_CAM01.h264"
        )
        assert result is not None
        assert result["camera_id"] == "CAM01"
        assert result["start_time"] == "2026-09-01T18:00:00"

    def test_camera2_format(self):
        result = parse_recording_filename(
            "20260901_180000_190000_CAMERA02.mp4"
        )
        assert result is not None
        assert result["camera_id"] == "CAMERA02"

    def test_non_matching_filename(self):
        result = parse_recording_filename("random_file.txt")
        assert result is None

    def test_invalid_date(self):
        result = parse_recording_filename(
            "20261301_180000_190000_CAM01.h264"
        )
        assert result is None


class TestParseTimestamp:
    def test_naive_iso_format(self):
        result = parse_timestamp("2026-09-01T18:00:00")
        assert result.normalization_status == TIMEZONE_UNKNOWN
        assert result.iso_naive == "2026-09-01T18:00:00"
        assert result.iso_aware is None
        assert result.utc is None
        assert result.original == "2026-09-01T18:00:00"

    def test_utc_suffix(self):
        result = parse_timestamp("2026-09-01T18:00:00Z")
        assert result.normalization_status == NORMALIZED
        assert result.iso_aware is not None
        assert result.utc is not None
        assert result.timezone == "UTC"

    def test_positive_offset(self):
        result = parse_timestamp("2026-09-01T18:00:00+05:30")
        assert result.normalization_status == NORMALIZED
        assert result.iso_aware is not None
        assert "+05:30" in result.iso_aware

    def test_negative_offset(self):
        result = parse_timestamp("2026-09-01T18:00:00-05:00")
        assert result.normalization_status == NORMALIZED
        assert result.iso_aware is not None
        assert "-05:00" in result.iso_aware

    def test_with_timezone_hint(self):
        result = parse_timestamp(
            "2026-09-01T18:00:00",
            timezone_hint="Asia/Kolkata",
        )
        assert result.normalization_status == NORMALIZED
        assert result.timezone == "Asia/Kolkata"
        assert result.timezone_source == "device_config"
        assert result.iso_aware is not None

    def test_invalid_timestamp(self):
        result = parse_timestamp("not-a-timestamp")
        assert result.normalization_status == INVALID

    def test_empty_timestamp(self):
        result = parse_timestamp("")
        assert result.normalization_status == INVALID

    def test_none_timestamp(self):
        result = parse_timestamp(None)
        assert result.normalization_status == INVALID

    def test_invalid_date_feb_30(self):
        result = parse_timestamp("2026-02-30T12:00:00")
        assert result.normalization_status == INVALID

    def test_invalid_date_apr_31(self):
        result = parse_timestamp("2026-04-31T12:00:00")
        assert result.normalization_status == INVALID

    def test_valid_leap_year(self):
        result = parse_timestamp("2024-02-29T12:00:00")
        assert result.normalization_status == TIMEZONE_UNKNOWN

    def test_invalid_non_leap_year(self):
        result = parse_timestamp("2025-02-29T12:00:00")
        assert result.normalization_status == INVALID

    def test_space_separated_format(self):
        result = parse_timestamp("2026-09-01 18:00:00")
        assert result.normalization_status == TIMEZONE_UNKNOWN
        assert result.iso_naive == "2026-09-01T18:00:00"

    def test_slash_separated_format(self):
        result = parse_timestamp("2026/09/01 18:00:00")
        assert result.normalization_status == TIMEZONE_UNKNOWN

    def test_format_detected(self):
        result = parse_timestamp("2026-09-01T18:00:00")
        assert result.format_used is not None


class TestIsValidDatetime:
    def test_valid_date(self):
        assert _is_valid_datetime(2026, 9, 1, 18, 0, 0) is True

    def test_invalid_month(self):
        assert _is_valid_datetime(2026, 13, 1, 0, 0, 0) is False

    def test_invalid_day(self):
        assert _is_valid_datetime(2026, 1, 32, 0, 0, 0) is False

    def test_invalid_hour(self):
        assert _is_valid_datetime(2026, 1, 1, 25, 0, 0) is False

    def test_invalid_minute(self):
        assert _is_valid_datetime(2026, 1, 1, 0, 60, 0) is False

    def test_invalid_second(self):
        assert _is_valid_datetime(2026, 1, 1, 0, 0, 60) is False

    def test_april_30_days(self):
        assert _is_valid_datetime(2026, 4, 30, 0, 0, 0) is True

    def test_april_31_days(self):
        assert _is_valid_datetime(2026, 4, 31, 0, 0, 0) is False

    def test_february_29_leap(self):
        assert _is_valid_datetime(2024, 2, 29, 0, 0, 0) is True

    def test_february_29_non_leap(self):
        assert _is_valid_datetime(2025, 2, 29, 0, 0, 0) is False


class TestDetectFormat:
    def test_iso_format(self):
        assert _detect_format("2026-09-01T18:00:00") == "%Y-%m-%dT%H:%M:%S"

    def test_compact_format(self):
        assert _detect_format("20260901180000") == "%Y%m%d%H%M%S"

    def test_unknown_format(self):
        assert _detect_format("not-a-date") == "unknown"


class TestTimezoneExtractor:
    def test_config_with_timezone(self):
        config = "timezone=Asia/Kolkata\nchannels=8"
        result = extract_timezone_from_config(config)
        assert result == "Asia/Kolkata"

    def test_config_utc(self):
        config = "timezone=UTC\ndebug=true"
        result = extract_timezone_from_config(config)
        assert result == "UTC"

    def test_config_gmt(self):
        config = "time_zone=GMT"
        result = extract_timezone_from_config(config)
        assert result == "UTC"

    def test_config_ist_abbreviation(self):
        config = "tz=IST"
        result = extract_timezone_from_config(config)
        assert result == "Asia/Kolkata"

    def test_config_no_timezone(self):
        config = "channels=8\ndebug=true"
        result = extract_timezone_from_config(config)
        assert result is None

    def test_config_comment_lines(self):
        config = "# This is a comment\ntimezone=America/New_York\n# Another comment"
        result = extract_timezone_from_config(config)
        assert result == "America/New_York"

    def test_json_with_timezone(self):
        data = {"timezone": "Europe/Berlin", "debug": False}
        result = extract_timezone_from_json(data)
        assert result == "Europe/Berlin"

    def test_json_utc_offset(self):
        data = {"tz": "UTC+5"}
        result = extract_timezone_from_json(data)
        assert result is not None

    def test_json_nested_timezone(self):
        data = {"system": {"timezone": "Asia/Tokyo"}}
        result = extract_timezone_from_json(data)
        assert result == "Asia/Tokyo"

    def test_json_no_timezone(self):
        data = {"debug": True, "channels": 8}
        result = extract_timezone_from_json(data)
        assert result is None


class TestResolveTimezone:
    def test_iana_format(self):
        assert _resolve_timezone("America/New_York") == "America/New_York"

    def test_utc_abbreviation(self):
        assert _resolve_timezone("UTC") == "UTC"

    def test_gmt_abbreviation(self):
        assert _resolve_timezone("GMT") == "UTC"

    def test_ist_abbreviation(self):
        assert _resolve_timezone("IST") == "Asia/Kolkata"

    def test_utc_plus_offset(self):
        result = _resolve_timezone("UTC+5")
        assert result is not None

    def test_utc_minus_offset(self):
        result = _resolve_timezone("UTC-8")
        assert result is not None

    def test_utc_zero(self):
        result = _resolve_timezone("UTC+0")
        assert result == "UTC"

    def test_empty_string(self):
        assert _resolve_timezone("") is None

    def test_unknown_value(self):
        assert _resolve_timezone("XYZ123") is None

    def test_quoted_value(self):
        assert _resolve_timezone('"Asia/Kolkata"') == "Asia/Kolkata"

    def test_single_quoted_value(self):
        assert _resolve_timezone("'Europe/London'") == "Europe/London"


class TestRecordingTimeline:
    def test_build_timeline_with_timestamps(self):
        ts = TimestampResult(
            original="20260901_180000",
            iso_naive="2026-09-01T18:00:00",
            normalization_status=NORMALIZED,
            timezone="Asia/Kolkata",
        )
        rec = Recording(
            filename="20260901_180000_190000_CAM01.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
            format="h264",
            start_timestamp=ts,
            end_timestamp=ts,
        )
        timeline = build_timeline([rec])
        assert len(timeline) == 1
        assert timeline[0]["start_timestamp"]["timezone"] == "Asia/Kolkata"
        assert timeline[0]["start_timestamp"]["normalization_status"] == NORMALIZED

    def test_build_timeline_backward_compatible(self):
        rec = Recording(
            filename="test.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
            format="h264",
        )
        timeline = build_timeline([rec])
        assert len(timeline) == 1
        assert "start_timestamp" not in timeline[0]

    def test_build_timeline_sorted(self):
        rec1 = Recording(
            filename="b.h264",
            start_time="2026-09-01T19:00:00",
            end_time="2026-09-01T20:00:00",
        )
        rec2 = Recording(
            filename="a.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        timeline = build_timeline([rec1, rec2])
        assert timeline[0]["filename"] == "a.h264"
        assert timeline[1]["filename"] == "b.h264"

    def test_recording_contains_time(self):
        rec = Recording(
            filename="test.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        ts = datetime(2026, 9, 1, 18, 30, 0)
        assert recording_contains_time(rec, ts) is True

    def test_recording_contains_time_outside(self):
        rec = Recording(
            filename="test.h264",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        ts = datetime(2026, 9, 1, 20, 0, 0)
        assert recording_contains_time(rec, ts) is False

    def test_recording_contains_time_mixed_tz(self):
        rec = Recording(
            filename="test.h264",
            start_time="2026-09-01T18:00:00+05:30",
            end_time="2026-09-01T19:00:00+05:30",
        )
        ts = datetime(2026, 9, 1, 18, 30, 0)
        assert recording_contains_time(rec, ts) is False


class TestEventCorrelator:
    def test_correlate_recordings_overlap(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T18:30:00",
            end_time="2026-09-01T19:30:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 1
        assert correlations[0]["timezone_comparable"] is True

    def test_correlate_recordings_no_overlap(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T20:00:00",
            end_time="2026-09-01T21:00:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 0

    def test_correlate_recordings_same_camera(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00",
            end_time="2026-09-01T19:00:00",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:30:00",
            end_time="2026-09-01T19:30:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 0

    def test_correlate_recordings_mixed_tz_skipped(self):
        rec1 = Recording(
            filename="a.h264",
            camera_id="CAM01",
            start_time="2026-09-01T18:00:00+05:30",
            end_time="2026-09-01T19:00:00+05:30",
        )
        rec2 = Recording(
            filename="b.h264",
            camera_id="CAM02",
            start_time="2026-09-01T18:30:00",
            end_time="2026-09-01T19:30:00",
        )
        correlations = correlate_recordings([rec1, rec2])
        assert len(correlations) == 0

    def test_timestamps_are_comparable_both_naive(self):
        rec1 = Recording(
            filename="a.h264",
            start_timestamp=TimestampResult(
                original="2026-09-01T18:00:00",
                normalization_status=TIMEZONE_UNKNOWN,
            ),
        )
        rec2 = Recording(
            filename="b.h264",
            start_timestamp=TimestampResult(
                original="2026-09-01T18:30:00",
                normalization_status=TIMEZONE_UNKNOWN,
            ),
        )
        assert _timestamps_are_comparable(rec1, rec2) is True

    def test_timestamps_are_comparable_both_aware(self):
        rec1 = Recording(
            filename="a.h264",
            start_timestamp=TimestampResult(
                original="2026-09-01T18:00:00+05:30",
                iso_aware="2026-09-01T18:00:00+05:30",
                normalization_status=NORMALIZED,
            ),
        )
        rec2 = Recording(
            filename="b.h264",
            start_timestamp=TimestampResult(
                original="2026-09-01T18:30:00+05:30",
                iso_aware="2026-09-01T18:30:00+05:30",
                normalization_status=NORMALIZED,
            ),
        )
        assert _timestamps_are_comparable(rec1, rec2) is True

    def test_timestamps_not_comparable_mixed(self):
        rec1 = Recording(
            filename="a.h264",
            start_timestamp=TimestampResult(
                original="2026-09-01T18:00:00",
                normalization_status=TIMEZONE_UNKNOWN,
            ),
        )
        rec2 = Recording(
            filename="b.h264",
            start_timestamp=TimestampResult(
                original="2026-09-01T18:30:00+05:30",
                iso_aware="2026-09-01T18:30:00+05:30",
                normalization_status=NORMALIZED,
            ),
        )
        assert _timestamps_are_comparable(rec1, rec2) is False

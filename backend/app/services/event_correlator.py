from datetime import datetime
from itertools import combinations

from app.models.dvr_evidence import Recording
from app.services.recording_timeline import (
    parse_datetime,
    recording_contains_time,
)


def _timestamps_are_comparable(
    first: Recording,
    second: Recording,
) -> bool:

    first_has_tz = (
        first.start_timestamp
        and first.start_timestamp.iso_aware
    )
    second_has_tz = (
        second.start_timestamp
        and second.start_timestamp.iso_aware
    )

    if first_has_tz and second_has_tz:
        return True

    first_has_meta = first.start_timestamp is not None
    second_has_meta = second.start_timestamp is not None

    if not first_has_meta and not second_has_meta:
        return True

    first_is_naive = (
        first.start_timestamp
        and first.start_timestamp.normalization_status == "TIMEZONE_UNKNOWN"
    )
    second_is_naive = (
        second.start_timestamp
        and second.start_timestamp.normalization_status == "TIMEZONE_UNKNOWN"
    )

    return bool(first_is_naive and second_is_naive)


def correlate_timestamp(
    recordings: list[Recording],
    timestamp: datetime,
) -> dict:

    active = [
        recording
        for recording in recordings
        if recording_contains_time(
            recording,
            timestamp,
        )
    ]

    cameras = sorted(
        {
            recording.camera_id
            for recording in active
            if recording.camera_id
        }
    )

    return {
        "timestamp": timestamp.isoformat(),
        "camera_count": len(cameras),
        "cameras": cameras,
        "recordings": [
            {
                "filename": recording.filename,
                "camera_id": recording.camera_id,
                "start_time": recording.start_time,
                "end_time": recording.end_time,
                "deleted": recording.deleted,
            }
            for recording in active
        ],
    }


def correlate_recordings(
    recordings: list[Recording],
) -> list[dict]:

    correlations = []

    valid_recordings = [
        recording
        for recording in recordings
        if (
            recording.start_time
            and recording.end_time
            and recording.camera_id
        )
    ]

    for first, second in combinations(
        valid_recordings,
        2,
    ):

        if first.camera_id == second.camera_id:
            continue

        first_start = parse_datetime(
            first.start_time
        )

        first_end = parse_datetime(
            first.end_time
        )

        second_start = parse_datetime(
            second.start_time
        )

        second_end = parse_datetime(
            second.end_time
        )

        if first_start.tzinfo is not None and second_start.tzinfo is None:
            continue
        if first_start.tzinfo is None and second_start.tzinfo is not None:
            continue

        overlap_start = max(
            first_start,
            second_start,
        )

        overlap_end = min(
            first_end,
            second_end,
        )

        if overlap_start > overlap_end:
            continue

        timezone_comparable = _timestamps_are_comparable(
            first,
            second,
        )

        correlations.append(
            {
                "camera_a": first.camera_id,
                "camera_b": second.camera_id,
                "recording_a": first.filename,
                "recording_b": second.filename,
                "overlap_start": (
                    overlap_start.isoformat()
                ),
                "overlap_end": (
                    overlap_end.isoformat()
                ),
                "timezone_comparable": timezone_comparable,
            }
        )

    return correlations

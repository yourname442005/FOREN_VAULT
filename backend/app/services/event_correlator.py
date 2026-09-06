from datetime import datetime
from itertools import combinations

from app.models.dvr_evidence import Recording
from app.services.recording_timeline import (
    parse_datetime,
    recording_contains_time,
)


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
            }
        )

    return correlations

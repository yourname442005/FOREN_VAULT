from datetime import datetime
from typing import Iterable

from app.models.dvr_evidence import Recording


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def recording_contains_time(
    recording: Recording,
    timestamp: datetime,
) -> bool:
    if not recording.start_time or not recording.end_time:
        return False

    start = parse_datetime(recording.start_time)
    end = parse_datetime(recording.end_time)

    return start <= timestamp <= end


def recordings_at_time(
    recordings: Iterable[Recording],
    timestamp: datetime,
) -> list[Recording]:

    return [
        recording
        for recording in recordings
        if recording_contains_time(
            recording,
            timestamp,
        )
    ]


def cameras_at_time(
    recordings: Iterable[Recording],
    timestamp: datetime,
) -> list[str]:

    camera_ids = []

    for recording in recordings:
        if not recording_contains_time(
            recording,
            timestamp,
        ):
            continue

        if (
            recording.camera_id
            and recording.camera_id not in camera_ids
        ):
            camera_ids.append(
                recording.camera_id
            )

    return camera_ids


def build_timeline(
    recordings: Iterable[Recording],
) -> list[dict]:

    timeline = []

    for recording in recordings:

        if not recording.start_time:
            continue

        timeline.append(
            {
                "camera_id": recording.camera_id,
                "filename": recording.filename,
                "start_time": recording.start_time,
                "end_time": recording.end_time,
                "format": recording.format,
                "deleted": recording.deleted,
            }
        )

    return sorted(
        timeline,
        key=lambda item: item["start_time"],
    )

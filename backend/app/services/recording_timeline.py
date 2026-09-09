from collections.abc import Iterable
from datetime import datetime

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

    if start.tzinfo is not None and timestamp.tzinfo is None:
        return False
    if start.tzinfo is None and timestamp.tzinfo is not None:
        return False

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

        entry = {
            "camera_id": recording.camera_id,
            "filename": recording.filename,
            "start_time": recording.start_time,
            "end_time": recording.end_time,
            "format": recording.format,
            "deleted": recording.deleted,
        }

        if recording.start_timestamp:
            entry["start_timestamp"] = {
                "original": recording.start_timestamp.original,
                "iso_naive": recording.start_timestamp.iso_naive,
                "iso_aware": recording.start_timestamp.iso_aware,
                "utc": recording.start_timestamp.utc,
                "timezone": recording.start_timestamp.timezone,
                "timezone_source": recording.start_timestamp.timezone_source,
                "normalization_status": recording.start_timestamp.normalization_status,
                "format_used": recording.start_timestamp.format_used,
            }

        if recording.end_timestamp:
            entry["end_timestamp"] = {
                "original": recording.end_timestamp.original,
                "iso_naive": recording.end_timestamp.iso_naive,
                "iso_aware": recording.end_timestamp.iso_aware,
                "utc": recording.end_timestamp.utc,
                "timezone": recording.end_timestamp.timezone,
                "timezone_source": recording.end_timestamp.timezone_source,
                "normalization_status": recording.end_timestamp.normalization_status,
                "format_used": recording.end_timestamp.format_used,
            }

        timeline.append(entry)

    return sorted(
        timeline,
        key=lambda item: item["start_time"],
    )

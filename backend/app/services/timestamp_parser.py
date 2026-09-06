import re
from datetime import datetime


RECORDING_FILENAME_PATTERNS = [
    re.compile(
        r"^(?P<date>\d{8})_"
        r"(?P<start>\d{6})_"
        r"(?P<end>\d{6})_"
        r"(?P<camera>(?:CAM|CAMERA)[_-]?\d+)"
        r"\.(?P<extension>[A-Za-z0-9]+)$",
        re.IGNORECASE,
    ),
]


def parse_recording_filename(
    filename: str,
) -> dict | None:

    for pattern in RECORDING_FILENAME_PATTERNS:
        match = pattern.match(filename)

        if not match:
            continue

        date = match.group("date")
        start = match.group("start")
        end = match.group("end")
        camera = match.group("camera")
        extension = match.group("extension").lower()

        try:
            start_datetime = datetime.strptime(
                f"{date}{start}",
                "%Y%m%d%H%M%S",
            )

            end_datetime = datetime.strptime(
                f"{date}{end}",
                "%Y%m%d%H%M%S",
            )

        except ValueError:
            return None

        return {
            "camera_id": camera.upper(),
            "start_time": start_datetime.isoformat(),
            "end_time": end_datetime.isoformat(),
            "format": extension,
        }

    return None

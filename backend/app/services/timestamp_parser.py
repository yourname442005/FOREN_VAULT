import re
from datetime import datetime, timedelta, timezone

from app.models.dvr_evidence import (
    INVALID,
    NORMALIZED,
    TIMEZONE_UNKNOWN,
    TimestampResult,
)

RECORDING_FILENAME_PATTERNS = [
    re.compile(
        r"^(?P<date>\d{8})_"
        r"(?P<start>\d{6})_"
        r"(?P<end>\d{6})_"
        r"(?P<camera>(?:CAM|CAMERA)[_-]?\d+)"
        r"\.(?P<extension>[A-Za-z0-9]+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<camera>(?:CAM|CAMERA)[_-]?\d+)_"
        r"(?P<date>\d{8})_"
        r"(?P<start>\d{6})_"
        r"(?P<end>\d{6})"
        r"\.(?P<extension>[A-Za-z0-9]+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<date>\d{4}-\d{2}-\d{2})_"
        r"(?P<start>\d{2}-\d{2}-\d{2})_"
        r"(?P<end>\d{2}-\d{2}-\d{2})_"
        r"(?P<camera>(?:CAM|CAMERA)[_-]?\d+)"
        r"\.(?P<extension>[A-Za-z0-9]+)$",
        re.IGNORECASE,
    ),
]

DATETIME_FORMATS = [
    "%Y%m%d%H%M%S",
    "%Y-%m-%d_%H-%M-%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]


def _is_valid_datetime(year, month, day, hour, minute, second):
    if not (1 <= month <= 12):
        return False
    if not (1 <= day <= 31):
        return False
    if not (0 <= hour <= 23):
        return False
    if not (0 <= minute <= 59):
        return False
    if not (0 <= second <= 59):
        return False
    if month in (4, 6, 9, 11) and day > 30:
        return False
    if month == 2:
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        if is_leap and day > 29:
            return False
        if not is_leap and day > 28:
            return False
    return True


def _parse_datetime_string(value: str) -> datetime | None:
    value = value.strip()
    tzinfo = None

    if value.endswith(("Z", "z")):
        value = value[:-1]
        tzinfo = timezone.utc
    elif "+" in value[10:]:
        parts = value.rsplit("+", 1)
        if len(parts) == 2:
            value = parts[0]
            try:
                offset_parts = parts[1].split(":")
                hours = int(offset_parts[0])
                minutes = int(offset_parts[1]) if len(offset_parts) > 1 else 0
                tzinfo = timezone(timedelta(hours=hours, minutes=minutes))
            except (ValueError, IndexError):
                pass
    elif value.count("-") > 2:
        parts = value.rsplit("-", 1)
        if len(parts) == 2 and ":" in parts[1]:
            value = parts[0]
            try:
                offset_parts = parts[1].split(":")
                hours = int(offset_parts[0])
                minutes = int(offset_parts[1]) if len(offset_parts) > 1 else 0
                utc_offset = timedelta(hours=hours, minutes=minutes)
                tzinfo = timezone(-utc_offset)
            except (ValueError, IndexError):
                pass

    for fmt in DATETIME_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)  # noqa: DTZ007
            if tzinfo is not None:
                dt = dt.replace(tzinfo=tzinfo)
            return dt
        except ValueError:
            continue

    return None


def parse_recording_filename(
    filename: str,
) -> dict | None:

    for pattern in RECORDING_FILENAME_PATTERNS:
        match = pattern.match(filename)

        if not match:
            continue

        groups = match.groupdict()
        camera = groups["camera"]
        extension = groups["extension"].lower()

        date_str = groups["date"]
        start_str = groups["start"]
        end_str = groups["end"]

        if "-" in start_str and start_str.count("-") == 2:
            start_fmt = "%H-%M-%S"
            end_fmt = "%H-%M-%S"
        else:
            start_fmt = "%H%M%S"
            end_fmt = "%H%M%S"

        try:
            if "-" in date_str:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")  # noqa: DTZ007
            else:
                date_obj = datetime.strptime(date_str, "%Y%m%d")  # noqa: DTZ007

            start_time_obj = datetime.strptime(start_str, start_fmt)  # noqa: DTZ007
            end_time_obj = datetime.strptime(end_str, end_fmt)  # noqa: DTZ007

            start_datetime = date_obj.replace(
                hour=start_time_obj.hour,
                minute=start_time_obj.minute,
                second=start_time_obj.second,
            )

            end_datetime = date_obj.replace(
                hour=end_time_obj.hour,
                minute=end_time_obj.minute,
                second=end_time_obj.second,
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


def parse_timestamp(
    value: str,
    timezone_hint: str | None = None,
) -> TimestampResult:

    original = value

    if not value or not value.strip():
        return TimestampResult(
            original=original,
            normalization_status=INVALID,
        )

    value = value.strip()

    dt = _parse_datetime_string(value)

    if dt is None:
        return TimestampResult(
            original=original,
            normalization_status=INVALID,
        )

    has_tz = dt.tzinfo is not None

    if not _is_valid_datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second):
        return TimestampResult(
            original=original,
            normalization_status=INVALID,
        )

    if has_tz:
        utc_dt = dt.astimezone(timezone.utc)
        naive_dt = dt.replace(tzinfo=None)
        return TimestampResult(
            original=original,
            iso_naive=naive_dt.isoformat(),
            iso_aware=dt.isoformat(),
            utc=utc_dt.isoformat(),
            timezone=str(dt.tzinfo),
            timezone_source="filename",
            normalization_status=NORMALIZED,
            format_used=_detect_format(value),
        )

    naive_iso = dt.isoformat()

    if timezone_hint:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(timezone_hint)
            aware_dt = dt.replace(tzinfo=tz)
            utc_dt = aware_dt.astimezone(timezone.utc)
            return TimestampResult(
                original=original,
                iso_naive=naive_iso,
                iso_aware=aware_dt.isoformat(),
                utc=utc_dt.isoformat(),
                timezone=timezone_hint,
                timezone_source="device_config",
                normalization_status=NORMALIZED,
                format_used=_detect_format(value),
            )
        except (ImportError, ValueError):
            pass

    return TimestampResult(
        original=original,
        iso_naive=naive_iso,
        normalization_status=TIMEZONE_UNKNOWN,
        format_used=_detect_format(value),
    )


def _detect_format(value: str) -> str:
    for fmt in DATETIME_FORMATS:
        try:
            datetime.strptime(value.strip(), fmt)  # noqa: DTZ007
            return fmt
        except ValueError:
            continue
    return "unknown"

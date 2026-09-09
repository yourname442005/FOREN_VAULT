import json
import re
from pathlib import Path

TIMEZONE_KEYS = {
    "timezone",
    "time_zone",
    "tz",
    "system_timezone",
    "device_timezone",
    "local_timezone",
}

KNOWN_TIMEZONE_ABBREVIATIONS = {
    "UTC": "UTC",
    "GMT": "UTC",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "IST": "Asia/Kolkata",
    "JST": "Asia/Tokyo",
    "CET": "Europe/Berlin",
    "CEST": "Europe/Berlin",
    "AEST": "Australia/Sydney",
    "AEDT": "Australia/Sydney",
}

IANA_TIMEZONE_PATTERN = re.compile(
    r"^[A-Z][a-zA-Z]+/[A-Z][a-zA-Z_]+$"
)


def extract_timezone_from_config(
    config_content: str,
) -> str | None:

    for line in config_content.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "[")):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

        if not value:
            continue

        if key in TIMEZONE_KEYS:
            resolved = _resolve_timezone(value)
            if resolved:
                return resolved

    return None


def extract_timezone_from_json(
    data: dict,
) -> str | None:

    for key in TIMEZONE_KEYS:
        if key in data:
            resolved = _resolve_timezone(str(data[key]))
            if resolved:
                return resolved

    for key, value in data.items():
        if isinstance(value, dict):
            found = extract_timezone_from_json(value)
            if found:
                return found

    return None


def extract_timezone_from_filesystem(
    filesystem_analysis: dict,
    get_file_content_fn,
    image_path: Path,
) -> str | None:

    timezone_files = {
        "device.conf",
        "system.conf",
        "config.conf",
        "settings.conf",
        "device.ini",
        "system.ini",
        "config.ini",
        "device.json",
        "system.json",
        "config.json",
        "settings.json",
        "time.conf",
        "timezone.conf",
    }

    for entry in filesystem_analysis.get("files", []):
        name = entry.get("name", "")
        basename = Path(name).name.lower()

        if basename not in timezone_files:
            continue

        content = get_file_content_fn(
            image_path,
            filesystem_analysis,
            entry,
        )

        if not content:
            continue

        if basename.endswith(".json"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    found = extract_timezone_from_json(data)
                    if found:
                        return found
            except json.JSONDecodeError:
                pass
        else:
            found = extract_timezone_from_config(content)
            if found:
                return found

    return None


def _resolve_timezone(value: str) -> str | None:

    if not value:
        return None

    value = value.strip().strip('"').strip("'")

    if IANA_TIMEZONE_PATTERN.match(value):
        return value

    upper = value.upper()
    if upper in KNOWN_TIMEZONE_ABBREVIATIONS:
        return KNOWN_TIMEZONE_ABBREVIATIONS[upper]

    offset_match = re.match(
        r"^UTC([+-])(\d{1,2})(?::(\d{2}))?$",
        upper,
    )
    if offset_match:
        sign = 1 if offset_match.group(1) == "+" else -1
        hours = int(offset_match.group(2))
        minutes = int(offset_match.group(3) or 0)
        total_minutes = sign * (hours * 60 + minutes)
        if total_minutes == 0:
            return "UTC"
        etc_hours = -total_minutes // 60
        etc_minutes = -total_minutes % 60
        if etc_minutes == 0:
            return f"Etc/GMT{'+' if etc_hours >= 0 else ''}{etc_hours}"
        return f"Etc/GMT{'+' if etc_hours >= 0 else ''}{etc_hours}:{etc_minutes:02d}"

    return None

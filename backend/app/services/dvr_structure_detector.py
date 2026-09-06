import re
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".h264",
    ".264",
    ".h265",
    ".265",
    ".dav",
    ".mp4",
    ".avi",
    ".mkv",
}

CAMERA_PATTERN = re.compile(
    r"^(cam|camera)[_-]?\d+$",
    re.IGNORECASE,
)

DVR_DIRECTORY_NAMES = {
    "recordings",
    "recording",
    "video",
    "videos",
    "camera",
    "cameras",
    "metadata",
    "config",
    "system",
    "logs",
}


def detect_dvr_structure(files: list[dict]) -> dict:
    camera_directories = []
    recording_files = []
    relevant_directories = []
    configuration_files = []
    log_files = []

    for entry in files:
        name = entry["name"]
        entry_type = entry["type"]

        if entry_type == "directory":
            normalized_name = name.lower()

            if normalized_name in DVR_DIRECTORY_NAMES:
                relevant_directories.append(name)

            if CAMERA_PATTERN.match(name):
                camera_directories.append(name)

        elif entry_type == "file":
            suffix = Path(name).suffix.lower()

            if suffix in VIDEO_EXTENSIONS:
                recording_files.append(name)

            if (
                suffix in {".conf", ".cfg", ".ini", ".json"}
                or "config" in name.lower()
            ):
                configuration_files.append(name)

            if (
                suffix in {".log"}
                or name.lower().startswith("log")
                or "_log" in name.lower()
                or "-log" in name.lower()
            ):
                log_files.append(name)

    clues = []

    if recording_files:
        clues.append(
            f"{len(recording_files)} surveillance-style video file(s) detected"
        )

    if camera_directories:
        clues.append(
            f"{len(camera_directories)} camera director{'y' if len(camera_directories) == 1 else 'ies'} detected"
        )

    if relevant_directories:
        clues.append(
            "DVR/NVR-related directory names detected"
        )

    if configuration_files:
        clues.append(
            f"{len(configuration_files)} configuration file(s) detected"
        )

    if log_files:
        clues.append(
            f"{len(log_files)} log file(s) detected"
        )

    score = 0.0

    if recording_files:
        score += 0.40

    if camera_directories:
        score += 0.30

    if relevant_directories:
        score += 0.15

    if configuration_files:
        score += 0.10

    if log_files:
        score += 0.05

    score = min(score, 1.0)

    if score >= 0.70:
        classification = "DVR_NVR_LIKELY"
    elif score >= 0.40:
        classification = "POSSIBLE_DVR_NVR"
    else:
        classification = "NOT_DVR_NVR"

    return {
        "classification": classification,
        "confidence": round(score, 2),
        "camera_directories": camera_directories,
        "recording_files": recording_files,
        "relevant_directories": relevant_directories,
        "configuration_files": configuration_files,
        "log_files": log_files,
        "clues": clues,
    }

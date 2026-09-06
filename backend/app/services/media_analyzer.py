import json
import subprocess
from pathlib import Path


def analyze_media(file_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    format_data = data.get("format", {})
    streams = data.get("streams", [])

    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )

    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )

    return {
        "container": format_data.get("format_long_name"),
        "format_name": format_data.get("format_name"),
        "duration": format_data.get("duration"),
        "size": format_data.get("size"),
        "bit_rate": format_data.get("bit_rate"),
        "creation_time": format_data.get("tags", {}).get("creation_time"),
        "make": format_data.get("tags", {}).get("com.apple.quicktime.make"),
        "model": format_data.get("tags", {}).get("com.apple.quicktime.model"),
        "video": {
            "codec": video_stream.get("codec_name") if video_stream else None,
            "width": video_stream.get("width") if video_stream else None,
            "height": video_stream.get("height") if video_stream else None,
            "frame_rate": (
                video_stream.get("avg_frame_rate")
                if video_stream
                else None
            ),
        },
        "audio": {
            "codec": audio_stream.get("codec_name") if audio_stream else None,
            "sample_rate": (
                audio_stream.get("sample_rate")
                if audio_stream
                else None
            ),
            "channels": (
                audio_stream.get("channels")
                if audio_stream
                else None
            ),
        },
    }
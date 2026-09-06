import re
from pathlib import Path

from app.models.dvr_evidence import Camera, DVREvidence, Recording
from app.parsers.vendor_base import VendorParser
from app.parsers.hikvision import (
    extract_file_content,
    parse_recording_filename,
)


class HoneywellParser(VendorParser):

    vendor_name = "Honeywell"

    MARKERS = (
        "honeywell",
        "honeywell security",
        "hrn",
        "hnb",
    )

    CAMERA_PATTERN = re.compile(
        r"^(cam|camera|ch|channel)[_-]?\d+$",
        re.IGNORECASE,
    )

    TEXT_EXTENSIONS = {
        ".conf",
        ".cfg",
        ".ini",
        ".json",
        ".xml",
        ".txt",
        ".log",
    }

    RECORDING_EXTENSIONS = {
        ".dav",
        ".h264",
        ".264",
        ".h265",
        ".265",
        ".mp4",
        ".ts",
    }

    def can_parse(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> bool:

        files = filesystem_analysis.get("files", [])

        for entry in files:

            name = entry.get(
                "name",
                "",
            ).lower()

            if any(
                marker in name
                for marker in self.MARKERS
            ):
                return True

        for entry in files:

            if entry.get("type") != "file":
                continue

            content = extract_file_content(
                image_path,
                filesystem_analysis,
                entry,
            )

            if not content:
                continue

            content_lower = content.lower()

            if any(
                marker in content_lower
                for marker in self.MARKERS
            ):
                return True

        return False

    def parse(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> DVREvidence:

        metadata = {}
        cameras = []
        recordings = []

        files = filesystem_analysis.get(
            "files",
            [],
        )

        for entry in files:

            if entry.get("type") != "file":
                continue

            filename = entry.get(
                "name",
                "",
            )

            extension = Path(
                filename
            ).suffix.lower()

            content = None

            if extension in self.TEXT_EXTENSIONS:

                content = extract_file_content(
                    image_path,
                    filesystem_analysis,
                    entry,
                )

            if content:

                for line in content.splitlines():

                    if "=" in line:

                        key, value = line.split(
                            "=",
                            1,
                        )

                        metadata[key.strip()] = (
                            value.strip()
                        )

            if extension in self.RECORDING_EXTENSIONS:

                parsed = parse_recording_filename(
                    filename
                )

                if parsed:

                    recordings.append(
                        Recording(
                            filename=filename,
                            camera_id=parsed.get(
                                "camera_id"
                            ),
                            start_time=parsed.get(
                                "start_time"
                            ),
                            end_time=parsed.get(
                                "end_time"
                            ),
                            format=extension.lstrip(
                                "."
                            ),
                            deleted=entry.get(
                                "deleted",
                                False,
                            ),
                        )
                    )

            camera_match = (
                self.CAMERA_PATTERN.match(
                    filename
                )
            )

            if camera_match:

                cameras.append(
                    Camera(
                        camera_id=filename,
                        source="filesystem",
                    )
                )

        return DVREvidence(
            vendor=self.vendor_name,
            model=metadata.get("model"),
            firmware=metadata.get("firmware"),
            cameras=cameras,
            recordings=recordings,
            metadata={
                "filesystem": (
                    filesystem_analysis.get(
                        "filesystem"
                    )
                ),
                "partition": (
                    filesystem_analysis.get(
                        "partition"
                    )
                ),
                "raw_device_metadata": metadata,
            },
            parser="honeywell",
        )
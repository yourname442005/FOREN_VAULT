import json
import re
from pathlib import Path

from app.models.dvr_evidence import Camera, DVREvidence, Recording
from app.models.vendor import (
    CAPABILITY_CAMERA_EXTRACTION,
    CAPABILITY_METADATA_EXTRACTION,
    CAPABILITY_RECORDING_EXTRACTION,
    CAPABILITY_TIMEZONE_EXTRACTION,
    CAPABILITY_TIMESTAMP_EXTRACTION,
    VendorCapability,
    VendorProfile,
)
from app.parsers.vendor_base import VendorParser
from app.services.timestamp_parser import (
    parse_recording_filename,
    parse_timestamp,
)
from app.services.timezone_extractor import (
    extract_timezone_from_filesystem,
)
from app.services.vendor_detector import (
    extract_file_content,
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

    def _get_file_content(
        self,
        image_path: Path,
        filesystem_analysis: dict,
        entry: dict,
    ) -> str | None:

        return extract_file_content(
            image_path,
            filesystem_analysis["filesystem"]["code"],
            filesystem_analysis["partition"][
                "start_sector"
            ],
            entry["inode"],
        )

    def get_capabilities(self) -> VendorProfile:
        return VendorProfile(
            vendor_name=self.vendor_name,
            capabilities=[
                VendorCapability(
                    name=CAPABILITY_METADATA_EXTRACTION,
                    supported=True,
                    detail="Extracts from .conf, .json, .xml, .ini, .txt, .log files",
                ),
                VendorCapability(
                    name=CAPABILITY_CAMERA_EXTRACTION,
                    supported=True,
                    detail="From directory names matching CAM/CAMERA/CH/CHANNEL pattern",
                ),
                VendorCapability(
                    name=CAPABILITY_RECORDING_EXTRACTION,
                    supported=True,
                    detail="Filename pattern YYYYMMDD_HHMMSS_HHMMSS_CAMx",
                ),
                VendorCapability(
                    name=CAPABILITY_TIMESTAMP_EXTRACTION,
                    supported=True,
                    detail="Phase 2 TimestampResult with normalization",
                ),
                VendorCapability(
                    name=CAPABILITY_TIMEZONE_EXTRACTION,
                    supported=True,
                    detail="From device config files",
                ),
            ],
            parser_class=self.__class__.__name__,
        )

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

            if (
                Path(entry.get("name", "")).suffix.lower()
                not in self.TEXT_EXTENSIONS
            ):
                continue

            content = self._get_file_content(
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

        timezone_hint = extract_timezone_from_filesystem(
            filesystem_analysis=filesystem_analysis,
            get_file_content_fn=self._get_file_content,
            image_path=image_path,
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

                content = self._get_file_content(
                    image_path,
                    filesystem_analysis,
                    entry,
                )

            if content:
                if extension == ".json":
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            metadata.update(data)
                    except json.JSONDecodeError:
                        pass
                else:
                    for line in content.splitlines():

                        if "=" in line:

                            key, value = line.split(
                                "=",
                                1,
                            )

                            metadata[
                                key.strip().lower()
                            ] = value.strip()

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
                            format=parsed.get(
                                "format"
                            ),
                            deleted=entry.get(
                                "deleted",
                                False,
                            ),
                            start_timestamp=parse_timestamp(
                                parsed["start_time"],
                                timezone_hint=timezone_hint,
                            ),
                            end_timestamp=parse_timestamp(
                                parsed["end_time"],
                                timezone_hint=timezone_hint,
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

        model = (
            metadata.get("model")
            or metadata.get("device_model")
            or metadata.get("devicemodel")
        )

        firmware = (
            metadata.get("firmware")
            or metadata.get("firmware_version")
            or metadata.get("version")
        )

        return DVREvidence(
            vendor=self.vendor_name,
            model=model,
            firmware=firmware,
            timezone=timezone_hint,
            timezone_source="device_config" if timezone_hint else None,
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

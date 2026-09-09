import json
import re
from pathlib import Path

from app.models.dvr_evidence import (
    Camera,
    DVREvidence,
    Recording,
)
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


class HikvisionParser(VendorParser):

    vendor_name = "Hikvision"

    MARKERS = [
        "hikvision",
        "hik",
        "ivms",
        "isapi",
        "ds-",
    ]

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
        ".h264",
        ".264",
        ".h265",
        ".265",
        ".dav",
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

    def _extract_metadata(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> dict:

        metadata = {}

        for entry in filesystem_analysis.get(
            "files",
            [],
        ):
            name = entry.get(
                "name",
                "",
            )

            suffix = Path(
                name
            ).suffix.lower()

            if suffix not in self.TEXT_EXTENSIONS:
                continue

            content = self._get_file_content(
                image_path,
                filesystem_analysis,
                entry,
            )

            if not content:
                continue

            if suffix == ".json":
                try:
                    data = json.loads(content)

                    if isinstance(data, dict):
                        metadata.update(data)

                except json.JSONDecodeError:
                    pass

            else:
                for line in content.splitlines():
                    line = line.strip()

                    if "=" not in line:
                        continue

                    key, value = line.split(
                        "=",
                        1,
                    )

                    metadata[
                        key.strip().lower()
                    ] = value.strip()

        return metadata

    def _extract_camera_metadata(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> dict:

        camera_metadata = {}

        for entry in filesystem_analysis.get(
            "files",
            [],
        ):
            name = entry.get(
                "name",
                "",
            )

            if (
                Path(name).name.lower()
                != "camera_config.json"
            ):
                continue

            content = self._get_file_content(
                image_path,
                filesystem_analysis,
                entry,
            )

            if not content:
                continue

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue

            for camera in data.get(
                "cameras",
                [],
            ):
                camera_id = camera.get(
                    "id"
                )

                if camera_id:
                    camera_metadata[
                        camera_id.upper()
                    ] = camera

        return camera_metadata

    def _extract_text_files(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> list[str]:

        contents = []

        for entry in filesystem_analysis.get(
            "files",
            [],
        ):
            name = entry.get(
                "name",
                "",
            )

            suffix = Path(
                name
            ).suffix.lower()

            if suffix not in self.TEXT_EXTENSIONS:
                continue

            content = self._get_file_content(
                image_path,
                filesystem_analysis,
                entry,
            )

            if content:
                contents.append(content)

        return contents

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
                    detail="From camera_config.json and CAM/CAMERA directory names",
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

        for content in self._extract_text_files(
            image_path,
            filesystem_analysis,
        ):
            content_lower = content.lower()

            if any(
                marker in content_lower
                for marker in self.MARKERS
            ):
                return True

        for entry in filesystem_analysis.get(
            "files",
            [],
        ):
            name = entry.get(
                "name",
                "",
            ).lower()

            if any(
                marker in name
                for marker in self.MARKERS
            ):
                return True

        return False

    def parse(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> DVREvidence:

        files = filesystem_analysis.get(
            "files",
            [],
        )

        metadata = self._extract_metadata(
            image_path,
            filesystem_analysis,
        )

        camera_metadata = (
            self._extract_camera_metadata(
                image_path,
                filesystem_analysis,
            )
        )

        timezone_hint = extract_timezone_from_filesystem(
            filesystem_analysis=filesystem_analysis,
            get_file_content_fn=self._get_file_content,
            image_path=image_path,
        )

        cameras = []
        recordings = []
        seen_cameras = set()

        for entry in files:

            name = entry.get(
                "name",
                "",
            )

            if entry.get("type") == "directory":

                if self.CAMERA_PATTERN.match(
                    name
                ):
                    camera_id = name.upper()

                    if camera_id in seen_cameras:
                        continue

                    seen_cameras.add(camera_id)

                    config = camera_metadata.get(
                        camera_id,
                        {},
                    )

                    cameras.append(
                        Camera(
                            camera_id=camera_id,
                            name=config.get(
                                "name"
                            ),
                            source="filesystem",
                        )
                    )

                continue

            if entry.get("type") != "file":
                continue

            suffix = Path(
                name
            ).suffix.lower()

            if suffix not in self.RECORDING_EXTENSIONS:
                continue

            parsed = parse_recording_filename(
                name
            )

            if parsed:
                recordings.append(
                    Recording(
                        filename=name,
                        camera_id=parsed[
                            "camera_id"
                        ],
                        start_time=parsed[
                            "start_time"
                        ],
                        end_time=parsed[
                            "end_time"
                        ],
                        format=parsed[
                            "format"
                        ],
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

            else:
                recordings.append(
                    Recording(
                        filename=name,
                        format=suffix.lstrip(
                            "."
                        ),
                        deleted=entry.get(
                            "deleted",
                            False,
                        ),
                    )
                )

        model = metadata.get(
            "model"
        )

        firmware = metadata.get(
            "firmware"
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
                "filesystem": filesystem_analysis[
                    "filesystem"
                ],
                "partition": filesystem_analysis[
                    "partition"
                ],
                "raw_device_metadata": metadata,
            },
            parser="hikvision",
        )

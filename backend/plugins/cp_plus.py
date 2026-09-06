import json
import re
from pathlib import Path

from app.models.dvr_evidence import (
    Camera,
    DVREvidence,
    Recording,
)
from app.parsers.vendor_base import VendorParser
from app.services.timestamp_parser import (
    parse_recording_filename,
)
from app.services.vendor_detector import (
    extract_file_content,
)


class CPPlusParser(VendorParser):

    vendor_name = "CP Plus"

    MARKERS = [
        "cp plus",
        "cpplus",
        "cp-plus",
        "cpplusworld",
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
        ".dav",
        ".h264",
        ".264",
        ".h265",
        ".265",
        ".mp4",
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

            suffix = Path(name).suffix.lower()

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

            suffix = Path(name).suffix.lower()

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
            ).lower()

            if name not in {
                "camera_config.json",
                "channel_config.json",
                "channels.json",
            }:
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

            cameras = (
                data.get("cameras")
                or data.get("channels")
                or []
            )

            if not isinstance(cameras, list):
                continue

            for camera in cameras:
                if not isinstance(camera, dict):
                    continue

                camera_id = (
                    camera.get("id")
                    or camera.get("channel")
                    or camera.get("camera_id")
                )

                if camera_id is None:
                    continue

                camera_id = str(
                    camera_id
                ).upper()

                camera_metadata[
                    camera_id
                ] = camera

        return camera_metadata

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

        cameras = []
        recordings = []
        seen_cameras = set()

        for entry in files:
            name = entry.get(
                "name",
                "",
            )

            if entry.get("type") == "directory":

                if self.CAMERA_PATTERN.match(name):
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

            suffix = Path(name).suffix.lower()

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
                    )
                )

            else:
                recordings.append(
                    Recording(
                        filename=name,
                        format=suffix.lstrip("."),
                        deleted=entry.get(
                            "deleted",
                            False,
                        ),
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
            parser="cp_plus",
        )
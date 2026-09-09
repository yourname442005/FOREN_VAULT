import json
import re
from pathlib import Path

from app.models.dvr_evidence import DVREvidence
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
from app.services.vendor_detector import (
    extract_file_content,
)


class MatrixParser(VendorParser):

    vendor_name = "Matrix"

    MARKERS = (
        "matrix",
        "matrix comsec",
        "visionpro",
        "sarv",
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
        ".h264",
        ".264",
        ".h265",
        ".265",
        ".mp4",
        ".avi",
        ".mkv",
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
                    detail="Extracts generic key/value and JSON metadata from config files",
                ),
                VendorCapability(
                    name=CAPABILITY_CAMERA_EXTRACTION,
                    supported=False,
                    detail="No deterministic Matrix camera structure identified",
                ),
                VendorCapability(
                    name=CAPABILITY_RECORDING_EXTRACTION,
                    supported=False,
                    detail="No deterministic Matrix recording filename pattern identified",
                ),
                VendorCapability(
                    name=CAPABILITY_TIMESTAMP_EXTRACTION,
                    supported=False,
                    detail="No deterministic Matrix timestamp format identified",
                ),
                VendorCapability(
                    name=CAPABILITY_TIMEZONE_EXTRACTION,
                    supported=False,
                    detail="No deterministic Matrix timezone structure identified",
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

        for entry in files:

            if entry.get("type") != "file":
                continue

            filename = entry.get(
                "name",
                "",
            )

            suffix = Path(filename).suffix.lower()

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
                    if "=" in line:
                        key, value = line.split(
                            "=",
                            1,
                        )
                        metadata[
                            key.strip().lower()
                        ] = value.strip()

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
            parser="matrix",
        )

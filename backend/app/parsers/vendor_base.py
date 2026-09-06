from abc import ABC, abstractmethod
from pathlib import Path

from app.models.dvr_evidence import DVREvidence


class VendorParser(ABC):

    vendor_name: str

    @abstractmethod
    def can_parse(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> bool:
        pass

    @abstractmethod
    def parse(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> DVREvidence:
        pass

from abc import ABC, abstractmethod
from pathlib import Path

from app.models.dvr_evidence import DVREvidence
from app.models.vendor import (
    CAPABILITY_VENDOR_DETECTION,
    CAPABILITY_METADATA_EXTRACTION,
    VendorCapability,
    VendorProfile,
)


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

    def get_capabilities(self) -> VendorProfile:
        return VendorProfile(
            vendor_name=self.vendor_name,
            capabilities=[
                VendorCapability(
                    name=CAPABILITY_VENDOR_DETECTION,
                    supported=True,
                ),
                VendorCapability(
                    name=CAPABILITY_METADATA_EXTRACTION,
                    supported=False,
                ),
            ],
            parser_class=self.__class__.__name__,
        )

from dataclasses import dataclass, field
from uuid import UUID

from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.detectors.schemas import (
    DetectorOutput,
)

from ._mask_manager import MaskManager
from ._part_properties import PartProperties
from ._schemas import PartNameType


@dataclass(slots=True)
class Part:
    # Input Variables
    detector_output: DetectorOutput
    config: Config
    file_uuid: UUID
    image_shape: tuple[int, int]

    # Properties
    properties: PartProperties = field(init=False)
    masks: MaskManager = field(init=False)

    def __post_init__(self):
        label = self.detector_output.label
        if label is None:
            msg = "Missing Name"
            raise TypeError(msg)

        # Part Properties
        self.properties = PartProperties(
            detector_output=self.detector_output,
            config=self.config,
        )

        # Mask Manager
        self.masks = MaskManager(
            mask_name=self.properties.settings.mask,
            protection_mask_name=self.properties.settings.protection_mask,
            image_shape=self.image_shape,
        )

    def get_name(self, output: PartNameType = PartNameType.NAME) -> str:
        name = self.properties.label
        id_part = self.properties.part_id
        merged = "merged" if self.properties.is_merged else "single"

        match output:
            case PartNameType.NAME:
                return name
            case PartNameType.ID_AND_NAME:
                return f"{id_part}_{name}"
            case PartNameType.ID_AND_NAME_AND_MERGED:
                return f"{id_part}_{name}_{merged}"
            case PartNameType.NAME_AND_MERGED:
                return f"{id_part}_{name}_{merged}"
            case _:
                return name

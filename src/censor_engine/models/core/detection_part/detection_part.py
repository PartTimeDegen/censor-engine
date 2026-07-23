from dataclasses import dataclass, field
from uuid import UUID

from censor_engine.models.libraries.configs.config import Config
from censor_engine.models.libraries.detectors.schemas import (
    DetectorOutput,
)
from censor_engine.models.libraries.masks.schemas import MaskContext

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
        mask_context = MaskContext(
            part_name=self.get_name(),
            part_properties=self.properties,
            mask=self.masks.current_mask,
            base_empty_mask=self.masks.create_empty_mask(),
        )
        self.masks = MaskManager(
            mask_name=self.properties.settings.mask,
            protection_mask_name=self.properties.settings.protection_mask,
            mask_context=mask_context,
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

from dataclasses import dataclass, field
from uuid import UUID

from censor_engine.models.models_library.configs.config import Config
from censor_engine.models.models_library.detectors.schemas import (
    DetectorOutput,
)

from ._config_shortcuts import ConfigShortcuts
from ._detection_properties import DetectionProperties
from ._mask_manager import MaskManager
from ._schemas import PartNameType


@dataclass(slots=True)
class Part:
    # Input Variables
    detector_output: DetectorOutput
    config: Config
    file_uuid: UUID
    image_shape: tuple[int, int]

    # Easy Access Classes
    _config_shortcuts: ConfigShortcuts = field(init=False)

    # Properties
    _properties: DetectionProperties = field(init=False)
    _masks: MaskManager = field(init=False)

    def __post_init__(self):
        label = self.detector_output.label
        if label is None:
            msg = "Missing Name"
            raise TypeError(msg)

        # Config Shortcuts
        part_config = self.config.detection.parts[label]
        self._config_shortcuts = ConfigShortcuts(
            detection_settings=self.config.detection,
            part_settings=part_config,
            groups=self.config.groups,
            image_settings=self.config.image,
        )

        # Part Properties
        self._properties = DetectionProperties(
            detector_output=self.detector_output,
            margins=self._config_shortcuts.margins,
            groups_merge=self._config_shortcuts.groups_merge,
            groups_persistance=self._config_shortcuts.groups_persistance,
        )

        # Mask Manager
        self._masks = MaskManager(
            mask_name=self._config_shortcuts.mask_name,
            protection_mask_name=self._config_shortcuts.protection_mask_name,
            image_shape=self.image_shape,
        )

    def get_name(self, output: PartNameType = PartNameType.NAME) -> str:
        name = self._properties.label
        id_part = self._properties.part_id
        merged = "merged" if self._properties.is_merged else "single"

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

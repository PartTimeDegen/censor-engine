from dataclasses import dataclass, field

from censor_engine.models.enums import MergeMethod
from censor_engine.models.libraries.configs.settings._detection import (
    DetectionSettings,
    _Margins,
    _PartSettings,
)
from censor_engine.models.libraries.configs.settings._groups import (
    GroupSettings,
)
from censor_engine.models.libraries.configs.settings._image import (
    ImageSettings,
)


@dataclass(slots=True)
class ConfigShortcuts:
    # Configs
    detection_settings: DetectionSettings
    part_settings: _PartSettings

    groups: GroupSettings

    image_settings: ImageSettings

    # Generated
    # # Part Changes
    minimum_score: float = field(init=False)
    margins: _Margins = field(init=False)

    # # Mask Settings
    mask_name: str = field(init=False)
    protection_mask_name: str | None = field(init=False)

    # # Merge and Persistance Groups
    groups_merge: list[list[str]] = field(init=False, default_factory=list)
    groups_persistance: list[list[str]] = field(
        init=False, default_factory=list
    )
    merge_method: MergeMethod = field(init=False)

    def __post_init__(self):
        # Part Changes
        self.minimum_score = self.part_settings.minimum_score
        self.margins = self.part_settings.margins

        # Mask Stuff
        self.mask_name = self.part_settings.mask
        self.protection_mask_name = self.part_settings.protection_mask

        # Groups
        self.groups_merge = self.groups.merging
        self.groups_persistance = self.groups.persistance

        # Merge Method Stuff
        self.merge_method = self.image_settings.merging.method
        if self.merge_method == MergeMethod.ALL:
            enabled_detections = self.detection_settings.enabled_parts
            if isinstance(enabled_detections, str):
                msg = "Need to Fix config"
                raise TypeError(msg)
            self.groups_merge = [enabled_detections]  # Type: ignore
            # TODO: This is due to an error in the config, need to fix this

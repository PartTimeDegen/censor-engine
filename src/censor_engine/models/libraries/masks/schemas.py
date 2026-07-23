from dataclasses import InitVar, dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from censor_engine._typing import MaskImage
from censor_engine.models.core.detection_part._part_properties import (
    PartProperties,
)

from .constants import FILLED, OBLIQUE


@dataclass(slots=True)
class GeneralParameters:
    colour: int = OBLIQUE
    thickness: float = FILLED

    fade_width: int = 0
    fade_gradient_mode: Literal["linear", "gaussian"] = "linear"


@dataclass(slots=True)
class MaskContext:
    # Tools
    part_name: str
    part_properties: PartProperties

    mask: MaskImage

    base_empty_mask: InitVar[MaskImage]
    file_uuid: UUID = field(default_factory=uuid4)
    _empty_mask: MaskImage = field(init=False)

    # Settings
    settings: GeneralParameters = field(default_factory=GeneralParameters)

    def __post_init__(self, base_empty_mask: MaskImage):
        self._empty_mask = base_empty_mask

    @property
    def is_merged(self) -> bool:
        return self.part_properties.is_merged

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._empty_mask.shape

    @property
    def cv2_standard_settings(self) -> dict[str, Any]:
        return {"color": self.settings.colour, "thickness": -1}

    @property
    def empty_mask(self) -> MaskImage:
        return self._empty_mask.copy()

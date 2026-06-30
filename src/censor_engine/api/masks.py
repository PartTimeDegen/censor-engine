from dataclasses import dataclass
from typing import TYPE_CHECKING

from censor_engine._typing import MaskImage

if TYPE_CHECKING:
    from censor_engine.models.detected_part import Part


@dataclass(slots=True)
class MaskContext:
    # Tools
    part: "Part"
    empty_mask: MaskImage

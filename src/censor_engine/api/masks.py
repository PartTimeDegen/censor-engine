from dataclasses import dataclass
from typing import TYPE_CHECKING

from censor_engine.typing import Mask

if TYPE_CHECKING:
    from censor_engine.detected_part import Part


@dataclass(slots=True)
class MaskContext:
    # Tools
    part: "Part"
    empty_mask: Mask

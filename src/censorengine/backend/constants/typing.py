from typing import TypeAlias, TypedDict

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censorengine.backend.models.detected_part import Part  # noqa: F401
    from censorengine.backend.models.config import Config  # noqa: F401


CVImage: TypeAlias = np.ndarray
Mask: TypeAlias = np.ndarray

NudeNetInfo = TypedDict(
    "NudeNetInfo",
    {
        "class": str,
        "score": float,
        "box": list[int],
    },
)

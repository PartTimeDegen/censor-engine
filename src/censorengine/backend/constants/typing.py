from typing import TypeAlias

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censorengine.backend.models.detected_part import Part  # noqa: F401
    from censorengine.backend.models.config import Config  # noqa: F401

# Information:
# You probably noticed there's three types for the same thing, it's to make
# understanding what type of image the package is processing. Mask is
# black/white, CVImage is a generic image, and ProcessedImage is the output
# for styles.
#


# This is a general Mask
Mask: TypeAlias = np.ndarray

# Type for a generic CVImage
CVImage: TypeAlias = np.ndarray

# Type for the end result of styles
ProcessedImage: TypeAlias = np.ndarray

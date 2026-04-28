from typing import Annotated

import numpy as np
from numpy.typing import NDArray

# Information:
# You probably noticed there's three types for the same thing, it's to make
# understanding what type of image the package is processing. Mask is
# black/white, Image is a generic image, and ProcessedImage is the output
# for effects.
#


type TypeMask = np.ndarray
type TypeEmptyMask = np.ndarray
type Image = np.ndarray
type ProcessedImage = np.ndarray
BBox = Annotated[NDArray[np.float32], (4,)]

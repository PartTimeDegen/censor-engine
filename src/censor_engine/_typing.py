from typing import Annotated

import numpy as np
from numpy.typing import NDArray

# Images
Image = Annotated[
    NDArray[np.uint8], ("H", "W", "C")
]  # generic color image (HWC)
GrayImage = Annotated[NDArray[np.uint8], ("H", "W")]  # grayscale
ProcessedImage = Annotated[NDArray[np.float32], ("H", "W", "C")]

# Masks
MaskImage = Annotated[NDArray[np.uint8], ("H", "W")]  # logical mask

# OpenCV-style mask (0/255)
OpenCVMask = Annotated[NDArray[np.uint32], ("H", "W")]

# Bounding box
BBox = tuple[int, int, int, int]  # [x1, y1, x2, y2]

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
Mask = Annotated[NDArray[np.bool_], ("H", "W")]  # logical mask
EmptyMask = Annotated[NDArray[np.bool_], ("H", "W")]

# OpenCV-style mask (0/255)
OpenCVMask = Annotated[NDArray[np.uint32], ("H", "W")]

# Bounding box
BBox = Annotated[NDArray[np.uint32], (4,)]  # [x1, y1, x2, y2]

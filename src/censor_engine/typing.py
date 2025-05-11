from typing import TypeAlias

import numpy as np


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

import numpy as np

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

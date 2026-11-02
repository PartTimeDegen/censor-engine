import cv2
import numpy as np

from censor_engine.typing import MaskImage


def fill_holes_within_mask(mask: MaskImage) -> MaskImage:
    # Get Slightly Bigger Mask
    # NOTE: The Mask is slightly bigger such that a flood fill hits all the
    #       mask without being stuck inside something at (0, 0).
    h, w = mask.shape[:2]
    flood_filled = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood Fill to find Holes as Black
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)  # type: ignore

    # Inverse, Showing Only Holes
    holes = cv2.bitwise_not(flood_filled)

    # Merge
    return cv2.bitwise_or(mask, holes)  # type: ignore

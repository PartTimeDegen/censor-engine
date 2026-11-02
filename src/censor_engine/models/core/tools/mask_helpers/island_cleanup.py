import cv2
import numpy as np

from censor_engine.typing import MaskImage


def filter_to_largest_connected_islands(
    mask: MaskImage,
    min_area: int = 200,
    relative_threshold: float = 0.1,
) -> MaskImage:
    # Get Binary Mask
    mask_binary = (mask > 0).astype(np.uint8)

    # Get Connected Components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )

    # Early Return if Only One Component
    if num_labels <= 1:
        return mask_binary * 255

    # Minimum Area/Relative Size Cutoff
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = areas.max()
    threshold = max(min_area, largest_area * relative_threshold)  # type: ignore

    mask_cleaned = np.zeros_like(mask_binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= threshold:
            mask_cleaned[labels == i] = 1

    return mask_cleaned * 255

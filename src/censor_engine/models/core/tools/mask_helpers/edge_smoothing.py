import cv2
import numpy as np

from censor_engine.typing import MaskImage


def fill_convex_hulls(mask: MaskImage) -> MaskImage:
    # Get Binary Mask
    binary = (mask > 0).astype(np.uint8) * 255

    # Find the Contours of the Mask
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the Convex Hull of the Mask
    result = np.zeros_like(binary)
    if not contours:
        return result
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)

    # Draw the Contours
    cv2.drawContours(result, [hull], -1, 255, thickness=cv2.FILLED)

    return result


def remove_stringy_parts_of_mask(
    mask: MaskImage,
    kernal_size: int = 7,
) -> MaskImage:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernal_size, kernal_size),
    )
    # Remove Strings
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # type: ignore
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # type: ignore

    return mask  # noqa: RET504


def smooth_edges(
    mask: MaskImage,
    kernal_size: int = 7,
) -> MaskImage:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernal_size, kernal_size),
    )
    # Use Morphology to Smooth Edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # type: ignore
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # type: ignore

    return mask  # noqa: RET504

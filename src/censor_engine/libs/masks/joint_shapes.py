from typing import TYPE_CHECKING

import cv2
import numpy as np

from censor_engine.libs.registries import MaskRegistry
from censor_engine.models.lib_models.masks import JointMask

if TYPE_CHECKING:
    from censor_engine.detected_part import Part
    from censor_engine.typing import TypeMask


@MaskRegistry.register()
class JointBox(JointMask):
    base_mask: str = "Ellipse"
    single_mask: str = "Box"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # Acquired from:
        # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
        rect = cv2.minAreaRect(np.vstack(cont_rect[0]).squeeze())
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        mask = cv2.drawContours(
            image=empty_mask,
            contours=[box],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if len(mask.shape) > 2:  # noqa: PLR2004
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return mask


@MaskRegistry.register()
class JointEllipse(JointMask):
    base_mask: str = "Ellipse"
    single_mask: str = "Ellipse"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # Find Minimum Area Ellipse
        cont_flat = np.vstack(cont_rect[0]).squeeze()
        mask_ellipse = cv2.fitEllipse(cont_flat)

        return cv2.ellipse(empty_mask, mask_ellipse, (255, 255, 255), -1)  # type: ignore


@MaskRegistry.register()
class RoundedJointBox(JointMask):
    base_mask: str = "Ellipse"
    single_mask: str = "RoundedBox"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
        mask = JointBox().generate(part, empty_mask)

        if len(mask.shape) > 2:  # noqa: PLR2004
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Rounding Part
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        iterations = 1
        mask_changed = cv2.erode(
            mask,
            kernel,
            iterations=iterations >> 1,
        )
        return cv2.dilate(
            mask_changed,
            kernel,
            iterations=iterations >> 1,
        )


@MaskRegistry.register()
class Block(JointMask):
    base_mask: str = "Ellipse"
    single_mask: str = "Box"

    def generate(self, part: "Part", empty_mask: "TypeMask") -> "TypeMask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        x, y, w, h = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

        # Define the box points (4 corners)
        box = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
        )

        mask = cv2.drawContours(
            image=empty_mask,
            contours=[box],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if len(mask.shape) > 2:  # noqa: PLR2004
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return mask

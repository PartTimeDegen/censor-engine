import cv2
import numpy as np

from censor_engine.libs.registries import ShapeRegistry
from censor_engine.models.lib_models.shapes import JointShape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censor_engine.typing import Mask
    from censor_engine.detected_part import Part


@ShapeRegistry.register()
class JointBox(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
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

        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return mask


@ShapeRegistry.register()
class JointEllipse(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "ellipse"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )

        # Find Minimum Area Ellipse
        cont_flat = np.vstack(cont_rect[0]).squeeze()
        shape_ellipse = cv2.fitEllipse(cont_flat)

        mask = cv2.ellipse(empty_mask, shape_ellipse, (255, 255, 255), -1)  # type: ignore

        return mask


@ShapeRegistry.register()
class RoundedJointBox(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "rounded_box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.imwrite("blah.jpg", part.mask)

        mask = JointBox().generate(part, empty_mask)

        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Rounding Part
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        iterations = 1
        mask_changed = cv2.erode(
            mask,
            kernel,
            iterations=iterations >> 1,
        )
        mask_changed = cv2.dilate(
            mask_changed,
            kernel,
            iterations=iterations >> 1,
        )

        return mask_changed

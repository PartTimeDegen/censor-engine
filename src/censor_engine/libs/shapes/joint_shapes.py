from typing import TYPE_CHECKING

import cv2
import numpy as np

from censor_engine.libs.registries import ShapeRegistry
from censor_engine.models.lib_models.shapes import JointShape

if TYPE_CHECKING:
    from censor_engine.detected_part import Part
    from censor_engine.typing import Mask


@ShapeRegistry.register()
class JointBox(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
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


@ShapeRegistry.register()
class JointEllipse(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "ellipse"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # Find Minimum Area Ellipse
        cont_flat = np.vstack(cont_rect[0]).squeeze()
        shape_ellipse = cv2.fitEllipse(cont_flat)

        return cv2.ellipse(empty_mask, shape_ellipse, (255, 255, 255), -1)  # type: ignore


@ShapeRegistry.register()
class RoundedJointBox(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "rounded_box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
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


@ShapeRegistry.register()
class Block(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
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


@ShapeRegistry.register()
class CoverBottom(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        width, height = empty_mask.shape[:2]
        _, y, _, _ = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

        box = np.array(
            [
                [0, y],
                [width, y],
                [width, height],
                [0, height],
            ],
            dtype=np.int32,
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


@ShapeRegistry.register()
class CoverTop(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        width, _ = empty_mask.shape[:2]
        _, y, _, h = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

        box = np.array(
            [
                [0, 0],
                [width, 0],
                [width, y + h],
                [0, y + h],
            ],
            dtype=np.int32,
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


@ShapeRegistry.register()
class CoverLeft(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        width, height = empty_mask.shape[:2]
        x, _, _, _ = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

        box = np.array(
            [
                [0, 0],
                [width - x, 0],
                [width - x, height],
                [0, height],
            ],
            dtype=np.int32,
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


@ShapeRegistry.register()
class CoverRight(JointShape):
    base_shape: str = "ellipse"
    single_shape: str = "box"

    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        width, height = empty_mask.shape[:2]
        x, _, _, _ = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

        box = np.array(
            [
                [x, 0],
                [width, 0],
                [width, height],
                [x, height],
            ],
            dtype=np.int32,
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

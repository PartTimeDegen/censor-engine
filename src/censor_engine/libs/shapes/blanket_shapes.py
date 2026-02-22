from typing import TYPE_CHECKING

import cv2
import numpy as np

from censor_engine.libs.registries import ShapeRegistry
from censor_engine.models.lib_models.shapes import BlanketShape

if TYPE_CHECKING:
    from censor_engine.detected_part import Part
    from censor_engine.typing import Mask


@ShapeRegistry.register()
class CoverLeft(BlanketShape):
    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        height, _ = empty_mask.shape[:2]
        x, _, w, _ = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

        box = np.array(
            [
                [0, 0],
                [0, height],
                [x + w, height],
                [x + w, 0],
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
class CoverRight(BlanketShape):
    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        height, width = empty_mask.shape[:2]
        x, _, w, _ = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

        box = np.array(
            [
                [width, 0],
                [width, height],
                [x, height],
                [x, 0],
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
class CoverBottom(BlanketShape):
    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        height, width = empty_mask.shape[:2]
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
class CoverTop(BlanketShape):
    def generate(self, part: "Part", empty_mask: "Mask") -> "Mask":
        cont_rect = cv2.findContours(
            image=part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        _, width = empty_mask.shape[:2]
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

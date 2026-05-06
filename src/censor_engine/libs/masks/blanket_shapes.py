from typing import TYPE_CHECKING

import cv2
import numpy as np

from censor_engine.api.masks import MaskContext
from censor_engine.libs.registries import MaskRegistry
from censor_engine.models.lib_models.masks import BlanketMask

if TYPE_CHECKING:
    from censor_engine.typing import Mask


@MaskRegistry.register()
class CoverLeft(BlanketMask):
    def generate(self, mask_context: MaskContext) -> "Mask":
        cont_rect = cv2.findContours(
            image=mask_context.part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        height, _ = mask_context.empty_mask.shape[:2]
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
            image=mask_context.empty_mask,
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
class CoverRight(BlanketMask):
    def generate(self, mask_context: MaskContext) -> "Mask":
        cont_rect = cv2.findContours(
            image=mask_context.part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        height, width = mask_context.empty_mask.shape[:2]
        x, _, _, _ = cv2.boundingRect(np.vstack(cont_rect[0]))  # type: ignore

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
            image=mask_context.empty_mask,
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
class CoverBottom(BlanketMask):
    def generate(self, mask_context: MaskContext) -> "Mask":
        cont_rect = cv2.findContours(
            image=mask_context.part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        height, width = mask_context.empty_mask.shape[:2]
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
            image=mask_context.empty_mask,
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
class CoverTop(BlanketMask):
    def generate(self, mask_context: MaskContext) -> "Mask":
        cont_rect = cv2.findContours(
            image=mask_context.part.mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        _, width = mask_context.empty_mask.shape[:2]
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
            image=mask_context.empty_mask,
            contours=[box],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if len(mask.shape) > 2:  # noqa: PLR2004
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return mask

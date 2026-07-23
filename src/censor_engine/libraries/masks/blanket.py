import cv2
import numpy as np

from censor_engine._typing import MaskImage
from censor_engine.libraries.registries import MaskRegistry
from censor_engine.models.libraries.masks.masks import BlanketMask
from censor_engine.models.libraries.masks.schemas import MaskContext


class _BaseCover(BlanketMask):
    def generate_mask(  # type: ignore
        self, mask_context: MaskContext, direction: str
    ) -> MaskImage:
        # Get Contours
        cont_rect = self.get_contours(mask_context)

        # Get Dimensions
        image_height, image_width = mask_context.empty_mask.shape[:2]
        left_side, top_side, mask_width, mask_height = cv2.boundingRect(
            np.vstack(cont_rect)
        )  # type: ignore

        # Aliases
        right_side = left_side + mask_width
        bottom_side = top_side + mask_height
        # Different Direction Algos
        cover_directions = {
            "top": [
                [0, 0],
                [image_width, 0],
                [image_width, bottom_side],
                [0, bottom_side],
            ],
            "bottom": [
                [0, top_side],
                [image_width, top_side],
                [image_width, image_height],
                [0, image_height],
            ],
            "left": [
                [0, 0],
                [0, image_height],
                [right_side, image_height],
                [right_side, 0],
            ],
            "right": [
                [image_width, 0],
                [image_width, image_height],
                [left_side, image_height],
                [left_side, 0],
            ],
        }

        # Generate Boxes
        box = np.array(
            cover_directions[direction],
            dtype=np.int32,
        )

        return cv2.drawContours(  # type: ignore
            image=mask_context.empty_mask,
            contours=[box],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=-1,
        )


@MaskRegistry.register()
class TopCover(_BaseCover):
    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return super().generate_mask(mask_context, direction="top")


@MaskRegistry.register()
class BottomCover(_BaseCover):
    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return super().generate_mask(mask_context, direction="bottom")


@MaskRegistry.register()
class LeftCover(_BaseCover):
    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return super().generate_mask(mask_context, direction="left")


@MaskRegistry.register()
class RightCover(_BaseCover):
    def generate_mask(self, mask_context: MaskContext) -> MaskImage:  # type: ignore
        return super().generate_mask(mask_context, direction="right")

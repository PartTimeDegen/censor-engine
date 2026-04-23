import cv2
import numpy as np

from censor_engine.models.structs.contours import Contour
from censor_engine.models.structs.meta_structs import Mixin
from censor_engine.typing import TypeMask


class MixinContourMasking(Mixin):
    def draw_mask(
        self,
        contours: list[Contour],
        image_mask: tuple[int, int],
        thickness: int,
        linetype: int,
    ) -> TypeMask:
        blank: TypeMask = np.zeros(image_mask, dtype=np.uint8)  # type: ignore
        return contours[0].draw_contour(blank, thickness, linetype)

    def apply_glow(self, mask: TypeMask, radius: int) -> TypeMask:
        blurred = cv2.GaussianBlur(mask, (0, 0), radius)
        _, glow_mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        return glow_mask.astype(np.uint8)  # type: ignore

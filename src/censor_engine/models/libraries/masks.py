import numpy as np

from censor_engine._typing import MaskImage
from censor_engine.api.masks import MaskContext
from censor_engine.models.enums import MaskType


class Mask:
    mask_name: str = "invalid_mask"
    base_mask: str = "invalid_mask"
    joint_mask: str = "invalid_mask"
    single_mask: str = "invalid_mask"

    mask_type: MaskType = MaskType.BASIC

    def __str__(self):
        return self.mask_name

    # Core Method
    def generate(
        self,
        mask_context: MaskContext,
    ) -> MaskImage:
        raise NotImplementedError

    # Static Methods
    @staticmethod
    def create_empty_mask(
        image_shape: tuple[int, int],
        *,
        inverse: bool = False,
    ) -> MaskImage:
        """
        This method is used to create an empty mask ie the base-image for the
        mask. The format is a greyscale (uint8) type image.

        :param tuple[int, int] image_shape: The image shape
        :param bool inverse: If the base mask is white instead of black,
        defaults to False
        :return MaskImage: Empty Mask
        """
        return (
            np.ones(image_shape, dtype=np.uint8) * 255
            if inverse
            else np.zeros(image_shape, dtype=np.uint8)
        )


class JointMask(Mask):
    mask_type: MaskType = MaskType.JOINT


class BlanketMask(Mask):
    single_mask: str = "Box"
    base_mask: str = "Box"
    mask_type: MaskType = MaskType.BLANKET


class BarMask(Mask):
    mask_type: MaskType = MaskType.BAR

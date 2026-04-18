from typing import TYPE_CHECKING

from censor_engine.api.masks import MaskContext
from censor_engine.models.enums import MaskType

if TYPE_CHECKING:
    from censor_engine.typing import TypeMask


class Mask:
    mask_name: str = "invalid_mask"
    base_mask: str = "invalid_mask"
    joint_mask: str = "invalid_mask"
    single_mask: str = "invalid_mask"

    mask_type: MaskType = MaskType.BASIC

    def __str__(self):
        return self.mask_name

    def generate(
        self,
        mask_context: MaskContext,
    ) -> "TypeMask":
        raise NotImplementedError


class JointMask(Mask):
    mask_type: MaskType = MaskType.JOINT


class BlanketMask(Mask):
    single_mask: str = "Box"
    base_mask: str = "Box"
    mask_type: MaskType = MaskType.BLANKET


class BarMask(Mask):
    mask_type: MaskType = MaskType.BAR

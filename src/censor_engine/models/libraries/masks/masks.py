from ._base_mask import Mask
from .enums import MaskType


class BasicMask(Mask):
    mask_type = MaskType.BASIC


class JointMask(Mask):
    mask_type = MaskType.JOINT
    single_mask = "Ellipse"


class BlanketMask(Mask):
    single_mask = "Box"
    base_mask = "Box"
    mask_type = MaskType.BLANKET


class BarMask(Mask):
    mask_type = MaskType.BAR


class PolygonMask(Mask):
    mask_type = MaskType.POLYGON

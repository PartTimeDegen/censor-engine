from censor_engine.models.enums import ShapeType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censor_engine.typing import Mask
    from censor_engine.detected_part import Part


class Shape:
    shape_name: str = "invalid_shape"
    base_shape: str = "invalid_shape"
    single_shape: str = "invalid_shape"

    shape_type: ShapeType = ShapeType.BASIC

    def __str__(self):
        return self.shape_name

    def generate(
        self,
        part: "Part",
        empty_mask: "Mask",
    ) -> "Mask":
        raise NotImplementedError


class JointShape(Shape):
    shape_type: ShapeType = ShapeType.JOINT


class BarShape(Shape):
    shape_type: ShapeType = ShapeType.BAR

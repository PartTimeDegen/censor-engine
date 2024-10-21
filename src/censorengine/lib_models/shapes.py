from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from censorengine.backend.constants.typing import Part, Mask


@dataclass
class Shape:
    shape_name: str = "invalid_shape"
    base_shape: str = "invalid_shape"
    single_shape: str = "invalid_shape"

    is_joint_shape: bool = False
    is_bar_shape: bool = False

    def generate(self, part: "Part") -> "Mask":
        raise NotImplementedError


class JointShape(Shape):
    is_joint_shape: bool = True


class BarShape(Shape):
    is_joint_shape: bool = True
    is_bar_shape: bool = True

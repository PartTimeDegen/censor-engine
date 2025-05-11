from typing import Type
from . import (
    bar_shapes,
    basic_shapes,
    joint_shapes,
)
from censor_engine.models.shapes import Shape

shape_catalogue: dict[str, Type[Shape]] = {
    **basic_shapes.shapes,
    **joint_shapes.shapes,
    **bar_shapes.shapes,
}

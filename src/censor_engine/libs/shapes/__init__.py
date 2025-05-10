from typing import Type
from censor_engine.libs.shape_library.shapes import (
    bar_shapes,
    basic_shapes,
    joint_shapes,
)
from censor_engine.lib_models.shapes import Shape

shape_catalogue: dict[str, Type[Shape]] = {
    **basic_shapes.shapes,
    **joint_shapes.shapes,
    **bar_shapes.shapes,
}

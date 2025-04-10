from typing import Type
from censorengine.libs.shape_library.shapes.core import (
    bar_shapes,
    basic_shapes,
    joint_shapes,
)
from censorengine.lib_models.shapes import Shape

shape_catalogue: dict[str, Type[Shape]] = {
    **basic_shapes.shapes,
    **joint_shapes.shapes,
    **bar_shapes.shapes,
}

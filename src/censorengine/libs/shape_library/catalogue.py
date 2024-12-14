from censorengine.libs.shape_library.shapes.core import (
    bar_shapes,
    basic_shapes,
    joint_shapes,
)

shape_catalogue = {
    **basic_shapes.shapes,
    **joint_shapes.shapes,
    **bar_shapes.shapes,
}

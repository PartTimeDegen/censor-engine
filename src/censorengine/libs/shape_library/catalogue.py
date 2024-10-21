from censorengine.libs.shape_library.shapes import (
    bar_shapes,
    basic_shapes,
    joint_shapes,
)

shape_catalogue = {
    shape().shape_name: shape
    for shape in [
        *bar_shapes.shapes,
        *basic_shapes.shapes,
        *joint_shapes.shapes,
    ]
}

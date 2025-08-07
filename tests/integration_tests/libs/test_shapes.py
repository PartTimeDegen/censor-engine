import inspect
import os
from censor_engine.models.enums import ShapeType
from tests.utils import run_image_test
import pytest
from censor_engine.libs.registries import ShapeRegistry

shapes = ShapeRegistry.get_all()
shapes_basic = [
    shape
    for shape, class_shape in shapes.items()
    if class_shape.shape_type == ShapeType.BASIC
]
shapes_joint = [
    shape
    for shape, class_shape in shapes.items()
    if class_shape.shape_type == ShapeType.JOINT
]
shapes_bar = [
    shape
    for shape, class_shape in shapes.items()
    if class_shape.shape_type == ShapeType.BAR
]


def run_shape_tests(shape, dummy_input_image_data):
    config_data = {
        "render_settings": {"smoothing": True},
        "censor_settings": {
            "enabled_parts": [
                "FEMALE_BREAST_EXPOSED",
            ],
            "merge_settings": {"merge_groups": [["FEMALE_BREAST_EXPOSED"]]},
            "default_part_settings": {
                "censors": [{"function": "overlay", "args": {"colour": "BLACK"}}],
                "shape": shape,
            },
        },
    }

    dummy_input_image_data.update_parts("FEMALE_BREAST_EXPOSED")
    run_image_test(
        dummy_input_image_data,
        config_data,
        subfolder=shape,
        batch_tests=True,
        group_name=os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0],
    )


@pytest.mark.parametrize("shape", shapes_basic)
def test_basic_shapes(shape, dummy_input_image_data):
    run_shape_tests(shape, dummy_input_image_data)


@pytest.mark.parametrize("shape", shapes_joint)
def test_joint_shapes(shape, dummy_input_image_data):
    run_shape_tests(shape, dummy_input_image_data)


@pytest.mark.parametrize("shape", shapes_bar)
def test_bar_shapes(shape, dummy_input_image_data):
    run_shape_tests(shape, dummy_input_image_data)


# Exceptions
def test_bar_single_object(dummy_input_image_data):
    config_data = {
        "render_settings": {"smoothing": True},
        "censor_settings": {
            "enabled_parts": [
                "FEMALE_BREAST_EXPOSED",
            ],
            "default_part_settings": {
                "censors": [{"function": "overlay", "args": {"colour": "BLACK"}}],
                "shape": "bar",
            },
        },
    }

    dummy_input_image_data.update_parts("FEMALE_BREAST_EXPOSED")
    run_image_test(
        dummy_input_image_data,
        config_data,
        group_name=os.path.splitext(os.path.basename(inspect.stack()[0].filename))[0],
        edge_case=True,
    )

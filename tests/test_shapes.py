import os

import pytest

from censorengine.backend._dev import (
    assert_files_are_intended,
    dev_compare_before_after_if_different,
)
from censorengine.backend.handlers.file_handlers.config_handler import (
    CONFIG,
    load_config,
)
from main import perform_image_files

# TODO: Check if shapes are different


@pytest.fixture()
def setup_parts():
    # Get Config
    file_path = os.path.dirname(__file__)
    list_path = file_path.split(os.sep)
    list_path = list_path[: list_path.index("tests")]
    root_path = os.sep.join(list_path)
    config_path = os.path.join(
        root_path, "tests", "configs", "test_shapes.yml"
    )
    load_config(config_path)
    yield root_path


basic_shape = [
    "box",
    "circle",
    "ellipse",
    "rounded_box",
]
joint_shape = [
    "joint_box",
    "joint_ellipse",
    "rounded_joint_box",
]
bar_shape = [
    "horizontal_bar",
    "bar",
    "vertical_bar",
]

shapes = ["invalid_shape"] + basic_shape + joint_shape + bar_shape


@pytest.mark.parametrize("shape_name", shapes)
def test_shape(setup_config, shape_name):
    results_path = f"shapes/{shape_name}"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="test_shapes.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["defaults"]["shape"] = shape_name

    # Perform Test
    perform_image_files(
        main_file_path=dict_locations["root_path"],
        test_mode=True,
    )

    # Assertion
    assert_files_are_intended(
        root_path=dict_locations["root_path"],
        results_path=results_path,
    )

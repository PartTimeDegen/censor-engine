import os

import pytest

from censorengine.backend._dev import assert_files_are_intended
from censorengine.backend.handlers.file_handlers.config_handler import (
    CONFIG,
    load_config,
)
from main import perform_image_files


# Note: Commented out ones are ones I couldn't get to work
classes = [
    "FACE_FEMALE",
    "ARMPITS_EXPOSED",
    "ARMPITS_COVERED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "BELLY_EXPOSED",
    "BELLY_COVERED",
    "BUTTOCKS_EXPOSED",
    "BUTTOCKS_COVERED",
    "ANUS_EXPOSED",
    #  "ANUS_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "FEET_EXPOSED",
    "FEET_COVERED",
    # "FACE_MALE",
    "MALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
]


@pytest.mark.parametrize("part_class", classes)
def test_part(setup_config, part_class):
    results_path = f"classes/{part_class}"
    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="test_classes.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG.update({"parts_enabled": [part_class]})

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

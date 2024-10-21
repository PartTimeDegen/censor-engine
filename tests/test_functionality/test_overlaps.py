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


# tests_left = [
#     "overlap_reveal_double",
#     "overlap_forced_double",
#     "overlap_reverse",
#     "overlap_reverse_double",
#     "overlap_reverse_reveal",
#     "overlap_reverse_forced",
# ]


def test_overlap(setup_config):
    results_path = "functionality/overlaps/overlap"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="functionality/test_functionality.yml",
        results_path=results_path,
    )

    # Config Updates

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


def test_overlap_reveal(setup_config):
    results_path = "functionality/overlaps/overlap_reveal"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="functionality/test_functionality.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["FACE_FEMALE"].update({"revealed": True})

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


def test_overlap_reveal_protected(setup_config):
    results_path = "functionality/overlaps/overlap_reveal_protected"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="functionality/test_functionality.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["FACE_FEMALE"].update({"revealed": True})
    CONFIG["information"]["MALE_GENITALIA_EXPOSED"].update({"protected": True})

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


def test_overlap_protected_demotion(setup_config):
    results_path = "functionality/overlaps/overlap_protected_demotion"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="functionality/test_demotion.yml",
        results_path=results_path,
    )

    # Config Updates

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


def test_overlap_protected(setup_config):
    results_path = "functionality/overlaps/overlap_protected/base"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="functionality/test_functionality.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["MALE_GENITALIA_EXPOSED"].update({"revealed": True})
    CONFIG["information"]["FEMALE_BREAST_EXPOSED"].update({"protected": True})

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


@pytest.mark.parametrize(
    "shape_name",
    [
        "protected_shape",
        "inherit_shape",
        "default_shape",
        "no_shape",
    ],
)
def test_overlap_protected_custom_shapes(setup_config, shape_name):
    results_path = (
        f"functionality/overlaps/overlap_protected/custom_shapes/{shape_name}"
    )

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="functionality/test_functionality.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["MALE_GENITALIA_EXPOSED"].update({"revealed": True})
    CONFIG["information"]["FEMALE_BREAST_EXPOSED"].update({"protected": True})

    match shape_name:
        case "protected_shape":
            CONFIG["information"]["FEMALE_BREAST_EXPOSED"].update(
                {"protected_shape": "circle"}
            )
        case "inherit_shape":
            CONFIG["information"]["FEMALE_BREAST_EXPOSED"].update(
                {"shape": "circle"}
            )
        case "default_shape":
            CONFIG["information"]["defaults"].update(
                {"protected_shape": "circle"}
            )
        case "no_shape":
            CONFIG["information"]["defaults"].pop("shape")

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

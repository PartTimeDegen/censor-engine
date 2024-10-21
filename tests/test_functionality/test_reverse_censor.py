# TODO: Reversed, protected, smoothing
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


"""
This file tests:
-   
-   

"""
reverse_censor = ["normal", "overlap", "revealed", "protected"]


def test_reverse_censor_normal(setup_config):
    results_path = "functionality/features/reverse_censor/normal"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="test_reverse_censor.yml",
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


def test_reverse_censor_revealed(setup_config):
    results_path = "functionality/features/reverse_censor/revealed"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="test_reverse_censor.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["FEMALE_GENITALIA_COVERED"] = {
        "revealed": True,
    }
    CONFIG["information"]["FEMALE_BREAST_EXPOSED"] = {
        "revealed": True,
    }
    CONFIG["information"]["FEMALE_BREAST_COVERED"] = {
        "revealed": True,
    }

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

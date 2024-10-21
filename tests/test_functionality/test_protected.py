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


def test_protected(setup_config):
    results_path = "functionality/features/protected"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="test_revealed_and_protected.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["FEMALE_GENITALIA_EXPOSED"] = {
        "protected": True,
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

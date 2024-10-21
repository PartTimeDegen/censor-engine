
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

issues = [
    "not_connected_overlap",
    "bar_width_not_properly_sized",
]


@pytest.mark.parametrize("issue", issues)
def test_shape(setup_config, issue):
    results_path = f"misc_issues/{issue}"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path=f"misc_issues/{issue}.yml",
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

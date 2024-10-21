import os

import pytest

from censorengine.backend._dev import (
    dev_compare_before_after_if_different,
    assert_files_are_intended,
)
from censorengine.backend.handlers.file_handlers.config_handler import (
    CONFIG,
    load_config,
)
from main import perform_image_files


BLUR_STRENGTH = 15
styles = {
    "greyscale": {},
}


@pytest.mark.parametrize("style", styles.keys())
def test_colour_effects(setup_config, style):
    results_path = f"styles/{style}"

    # Load Config and get Locs
    dict_locations = setup_config(
        config_path="test_styles.yml",
        results_path=results_path,
    )

    # Config Updates
    CONFIG["information"]["defaults"]["censors"] = [
        {"function": style, "args": styles[style]}
    ]

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

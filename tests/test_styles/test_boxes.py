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


BLUR_STRENGTH = 15
styles = {
    "box": {
        "rgb_colour": "BLACK",
    },
    # "caption": {"rgb_colour": "BLACK"},
    "dev_debug": {
        "rgb_colour": "BLACK",
    },
    "outline": {
        "rgb_colour": "BLACK",
        "pxl_thickness": 4,
    },
    "outlined_box": {
        "rgb_colour_box": "BLACK",
        "pxl_thickness": 4,
        "rgb_colour_outline": "HOT_PINK",
    },
    "overlay": {
        "rgb_colour": "BLACK",
        "alpha": 0.5,
    },
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

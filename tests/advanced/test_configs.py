import os
from censorengine import CensorEngine  # type: ignore
from censorengine.backend._dev import assert_files_are_intended  # type: ignore

import pytest

CONFIGS = [
    "00_default",
    "01_cooler_default",
    "02_ellipses",
    "03_joint_ellipses",
    "04_black_bars",
    "05_blurred_lines",
    "06_cutouts",
    "07_only_blur",
    "08_pixel_paradise",
    "09_red_tape",
    "10_state_approved_triggers",
    "11_trigger_focus",
    "12_white_bars",
]


@pytest.mark.parametrize("config", CONFIGS)
def test_configs(config, root_path):
    # Test Type
    test_loc = os.path.join("advanced/configs", config)

    # Initiate
    ce = CensorEngine(
        root_path,
        config_data=f"{config}.yml",
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    ce._config.file_settings.uncensored_folder = folder_uncen
    ce._config.file_settings.censored_folder = folder_cen

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)

import os
from censorengine import CensorEngine  # type: ignore
from censorengine.backend._dev import assert_files_are_intended  # type: ignore
from censorengine.libs.shape_library.catalogue import shape_catalogue  # type: ignore
from censorengine.libs.style_library.catalogue import style_catalogue  # type: ignore
import pytest

CONFIGS = [
    "00_default",
    "01_cooler_default",
    "02_ellipses",
    "03_joint_ellipses",
    "black_bars",
    "blurred_lines",
    "cutouts",
    "only_blur",
    "pixel_paradise",
    "red_tape",
    "state_approved_triggers",
    "trigger_focus",
    "white_bars",
]


@pytest.mark.parametrize("config", CONFIGS)
def test_configs(config, root_path):
    # Test Type
    test_loc = os.path.join("advanced/configs", config)

    # Initiate
    ce = CensorEngine(
        root_path,
        config=f"{config}.yml",
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    ce.config.uncensored_folder = folder_uncen
    ce.config.censored_folder = folder_cen

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)

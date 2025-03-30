import os

from censorengine import CensorEngine  # type: ignore
from censorengine.backend._dev import assert_files_are_intended  # type: ignore
from censorengine.backend.models.schemas import Censor

COLOURS = [
    "WHITE",
    "BLACK",
    "PINK",
    "YELLOW",
    "RED",
    "GREEN",
    "BLUE",
]


def test_basic(root_path):
    # Test Type
    test_loc = os.path.join("advanced/overlaps/basic")

    # Initiate
    ce = CensorEngine(
        root_path,
        config_data="000_tests/03_configs/advanced/overlaps.yml",
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    ce.config.file_settings.uncensored_folder = folder_uncen
    ce.config.file_settings.censored_folder = folder_cen
    ce.config.censor_settings.enabled_parts = [
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "BUTTOCKS_EXPOSED",
    ]
    for part in ce.config.censor_settings.enabled_parts:
        ce.config.censor_settings.parts_settings[part].shape = "bar"

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)


def test_overlapping_parts(root_path):
    # Test Type
    test_loc = os.path.join("advanced/overlaps/parts")

    # Initiate
    ce = CensorEngine(
        root_path,
        config_data="000_tests/03_configs/advanced/overlaps.yml",
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    parts = [
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "BUTTOCKS_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
    ]
    ce.config.file_settings.uncensored_folder = folder_uncen
    ce.config.file_settings.censored_folder = folder_cen
    ce.config.censor_settings.enabled_parts = parts

    for part, colour in zip(parts, COLOURS):
        ce.config.censor_settings.parts_settings[part].censors = [
            Censor("outline", {"colour": (colour)}),
            Censor("blur", {}),
        ]
        ce.config.censor_settings.parts_settings[part].margin = 5

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)

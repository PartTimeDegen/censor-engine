import os
from censorengine import CensorEngine  # type: ignore
from censorengine.backend._dev import assert_files_are_intended  # type: ignore

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
        config="000_tests/03_configs/advanced/overlaps.yml",
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    ce.config.uncensored_folder = folder_uncen
    ce.config.censored_folder = folder_cen
    ce.config.parts_enabled = [
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "BUTTOCKS_EXPOSED",
    ]
    for part in ce.config.parts_enabled:
        ce.config.part_settings[part].shape = "bar"
        ce.config.dev_load_censors(
            part,
            [
                {
                    "function": "outline",
                    "args": {"colour": "BLACK"},
                },
                {
                    "function": "blur",
                    "args": {},
                },
            ],
        )

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)


def test_overlapping_parts(root_path):
    # Test Type
    test_loc = os.path.join("advanced/overlaps/parts")

    # Initiate
    ce = CensorEngine(
        root_path,
        config="000_tests/03_configs/advanced/overlaps.yml",
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
    ce.config.uncensored_folder = folder_uncen
    ce.config.censored_folder = folder_cen
    ce.config.parts_enabled = parts
    ce.config.dev_generate_new_parts()

    for part, colour in zip(parts, COLOURS):
        ce.config.dev_load_censors(
            part,
            [
                {
                    "function": "outline",
                    "args": {"colour": (colour)},
                },
                {
                    "function": "blur",
                    "args": {},
                },
            ],
        )
        ce.config.part_settings[part].margin = 5

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)

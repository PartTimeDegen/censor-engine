import os
from censorengine import CensorEngine  # type: ignore
from censorengine.backend._dev import assert_files_are_intended  # type: ignore
from censorengine.libs.shape_library.catalogue import shape_catalogue  # type: ignore
from censorengine.libs.style_library.catalogue import style_catalogue  # type: ignore
import pytest

# Note: Commented out ones are ones I couldn't get to work
classes = [
    "FACE_FEMALE",
    "ARMPITS_EXPOSED",
    # "ARMPITS_COVERED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "BELLY_EXPOSED",
    "BELLY_COVERED",
    "BUTTOCKS_EXPOSED",
    "BUTTOCKS_COVERED",
    "ANUS_EXPOSED",
    #  "ANUS_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "FEET_EXPOSED",
    "FEET_COVERED",
    "FACE_MALE",
    "MALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
]

shapes = shape_catalogue.keys()
styles = style_catalogue.keys()


@pytest.mark.parametrize("part_name", classes)
# @pytest.mark.timeout(5)
def test_classes(part_name, root_path):
    # Test Type
    test_loc = os.path.join("core/classes", part_name)

    # Initiate
    ce = CensorEngine(
        root_path,
        config="000_tests/03_configs/test_core.yml",
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    ce.config.uncensored_folder = folder_uncen
    ce.config.censored_folder = folder_cen

    ce.config.parts_enabled = [part_name]

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)


@pytest.mark.parametrize("shape", shapes)
# @pytest.mark.timeout(5)
def test_shapes(shape, root_path):
    # Test Type
    test_loc = os.path.join("core/shapes", shape)

    # Initiate
    ce = CensorEngine(
        root_path,
        config="000_tests/03_configs/test_core.yml",
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
    ]

    for part in ce.config.parts_enabled:
        ce.config.part_settings[part].shape = shape

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)


@pytest.mark.parametrize("style", styles)
# @pytest.mark.timeout(5)
def test_styles(style, root_path):
    # Test Type
    test_loc = os.path.join("core/styles", style)

    # Initiate
    ce = CensorEngine(
        root_path,
        config="000_tests/03_configs/test_core.yml",
        test_mode=True,
    )

    folder_uncen = os.path.join("000_tests/00_uncensored", test_loc)
    folder_cen = "000_tests/01_censored"

    # Set Config
    ce.config.uncensored_folder = folder_uncen
    ce.config.censored_folder = folder_cen

    colours = [
        "BLACK",
        "RED",
        "BLUE",
        "YELLOW",
        "GREEN",
        "GREY",
        "PURPLE",
        "BROWN",
        "ORANGE",
        "WHITE",
        "PINK",
        "HOT_GREEN",
        "HOT_BLUE",
        "HOT_RED",
        "HOT_PINK",
        "HOT_ORANGE",
        "HOT_YELLOW",
        "WHITE",
        "DARK_PINK",
    ]

    for index, part in enumerate(ce.config.parts_enabled):
        ce.config.dev_load_censors(
            part,
            [
                {
                    "function": "outline",
                    "args": {
                        "colour": "BLACK" if style != "dev_debug" else colours[index]
                    },
                },
                {
                    "function": style,
                    "args": {},
                },
            ],
        )

    # Start CensorEngine
    ce.start()

    assert_files_are_intended(root_path, test_loc)


pytest.mark.parametrize("shape", shapes)


def test_reverse_censor(root_path):
    test_loc = os.path.join("core", "reverse_censor")

    # Initiate
    ce = CensorEngine(
        root_path,
        config="000_tests/03_configs/test_reverse_censor.yml",
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

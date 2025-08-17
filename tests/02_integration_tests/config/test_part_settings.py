import inspect
import os
from pathlib import Path

import pytest

from censor_engine.libs.detectors.box_based_detectors.multi_detectors import (
    NudeNetDetector,
)

# from censor_engine.models.enums import PartState
from tests.utils import run_image_test

part_settings = {
    "minimum_score": [
        0.0,
        0,
        -10,
        100,
        0.2,
        1.2,
    ],
    "censors": [
        # Good
        [],
        [{"style": "blur"}],
        [{"style": "blur", "parameters": {"factor": 20}}],
        "blur",
        # Bad # TODO: Consider how to check for bad inputs, also if should include in tests
        # "fake censor",
        # [{"style": "fake censor"}],
        # [{"style": "fake censor", "parameters": {"factor": 20}}],
        # [{"style": "blur", "parameters": {"factor": 20}}],
    ],
    "shape": [
        # Good
        "circle",
        # Base
        # "dinosaur",  # TODO: Add to "raises error"
    ],
    # "protected_shape": [ # TODO: Needs own tests
    #     PartState.UNPROTECTED,
    #     PartState.PROTECTED,
    #     PartState.REVEALED,
    # ],
    "fade_percent": [
        # Good
        0,
        50,
        100,
        # Bad
        0.0,
        -50,
        150,
    ],
    # "video_part_search_region": [
    #     # Good
    #     0.2,
    #     0.5,
    #     1.0,
    #     # Bad
    #     1,
    #     -1.0,
    #     0.0,
    # ],
}


def run_base(field, value, dummy_input_image_data, use_part: bool):
    all_parts = list(NudeNetDetector.model_classifiers)

    backup_info = {
        "censors": [{"style": "outlined_overlay"}],
        "shape": "joint_box",
    }

    config_data = {
        "censor_settings": {
            "enabled_parts": all_parts,
            "merge_settings": {
                "merge_groups": [
                    [
                        "FEMALE_BREAST_EXPOSED",
                        "FEMALE_BREAST_COVERED",
                    ]
                ]
            },
        },
    }
    censor_data = config_data["censor_settings"]
    if use_part:
        censor_data["default_part_settings"] = dict(backup_info)
        censor_data["FEMALE_BREAST_EXPOSED"] = {field: value}
        if field != "shape":
            censor_data["FEMALE_BREAST_EXPOSED"]["shape"] = "joint_box"
    else:
        censor_data["default_part_settings"] = dict(backup_info)
        censor_data["default_part_settings"][field] = value
        if field != "shape":
            censor_data["default_part_settings"]["shape"] = "joint_box"

    run_image_test(
        dummy_input_image_data,
        config_data,
        subfolder=str(Path(field) / str(value)),
        batch_tests=True,
        group_name=os.path.splitext(
            os.path.basename(inspect.stack()[1].filename)
        )[0],
    )


@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field, values in part_settings.items()
        for value in values
    ],
)
def test_default_parts(field, value, dummy_input_image_data):
    run_base(field, value, dummy_input_image_data, use_part=False)


@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field, values in part_settings.items()
        for value in values
    ],
)
def test_parts(field, value, dummy_input_image_data):
    run_base(field, value, dummy_input_image_data, use_part=True)

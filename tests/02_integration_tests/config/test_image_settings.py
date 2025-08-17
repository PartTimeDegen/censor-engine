import inspect
import os

import pytest

from censor_engine.libs.detectors.box_based_detectors.multi_detectors import (
    NudeNetDetector,
)
from tests.utils import run_image_test

rendering_merge_method = [
    "none",
    "parts",
    "groups",
    "full",
    "GrOuPs",  # Case test
]


def run_merge_method(merge_method, dummy_input_image_data):
    all_parts = list(NudeNetDetector.model_classifiers)
    config_data = {
        "render_settings": {"merge_method": merge_method},
        "censor_settings": {
            "enabled_parts": all_parts,
            "merge_settings": {
                "merge_groups": [
                    [
                        "FEMALE_BREAST_EXPOSED",
                        "FEMALE_BREAST_COVERED",
                    ]
                ]
                if merge_method == "groups" or merge_method == "GrOuPs"
                else [all_parts]
            },
            "default_part_settings": {
                "censors": [{"style": "outlined_overlay"}],
                "shape": "joint_box",
            },
        },
    }

    dummy_input_image_data.update_parts("FEMALE_BREAST_EXPOSED")
    run_image_test(
        dummy_input_image_data,
        config_data,
        subfolder=merge_method,
        batch_tests=True,
        group_name=os.path.splitext(
            os.path.basename(inspect.stack()[1].filename)
        )[0],
    )


def run_reverse_censor(dummy_input_image_data):
    all_parts = list(NudeNetDetector.model_classifiers)
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
            "default_part_settings": {
                "censors": [{"style": "no_censor"}],
                "shape": "joint_box",
            },
            "reverse_censor_settings": [{"style": "outlined_overlay"}],
        },
    }

    dummy_input_image_data.update_parts("FEMALE_BREAST_EXPOSED")
    run_image_test(
        dummy_input_image_data,
        config_data,
        batch_tests=True,
        group_name=os.path.splitext(
            os.path.basename(inspect.stack()[1].filename)
        )[0],
    )


@pytest.mark.parametrize("merge_method", rendering_merge_method)
def test_merge_methods(merge_method, dummy_input_image_data):
    run_merge_method(merge_method, dummy_input_image_data)


def test_reverse_censor(dummy_input_image_data):
    run_reverse_censor(dummy_input_image_data)

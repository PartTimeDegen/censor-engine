import inspect
import os

import pytest

from censor_engine.libs.registries import MaskRegistry
from censor_engine.models.enums import MaskType
from tests.utils import run_image_test

masks = MaskRegistry.get_all()
masks_basic = [
    mask
    for mask, class_mask in masks.items()
    if class_mask.mask_type == MaskType.BASIC
]
masks_joint = [
    mask
    for mask, class_mask in masks.items()
    if class_mask.mask_type == MaskType.JOINT
]
masks_bar = [
    mask
    for mask, class_mask in masks.items()
    if class_mask.mask_type == MaskType.BAR
]
masks_blanket = [
    mask
    for mask, class_mask in masks.items()
    if class_mask.mask_type == MaskType.BLANKET
]


def run_mask_tests(mask, dummy_input_image_data) -> None:
    config_data = {
        "censor_settings": {
            "enabled_parts": [
                "FEMALE_BREAST_EXPOSED",
            ],
            "merge_settings": {"merge_groups": [["FEMALE_BREAST_EXPOSED"]]},
            "default_part_settings": {
                "censors": [
                    {"effect": "Overlay", "parameters": {"colour": "BLACK"}},
                ],
                "mask": mask,
            },
        },
    }

    dummy_input_image_data.update_parts("FEMALE_BREAST_EXPOSED")
    run_image_test(
        dummy_input_image_data,
        config=config_data,
        subfolder=mask,
        batch_tests=True,
        group_name=os.path.splitext(
            os.path.basename(inspect.stack()[1].filename),
        )[0],
    )


@pytest.mark.parametrize("mask", masks_basic)
def test_basic_masks(mask, dummy_input_image_data) -> None:
    run_mask_tests(mask, dummy_input_image_data)


@pytest.mark.parametrize("mask", masks_joint)
def test_joint_masks(mask, dummy_input_image_data) -> None:
    run_mask_tests(mask, dummy_input_image_data)


@pytest.mark.parametrize("mask", masks_bar)
def test_bar_masks(mask, dummy_input_image_data) -> None:
    run_mask_tests(mask, dummy_input_image_data)


@pytest.mark.parametrize("mask", masks_blanket)
def test_blanket_masks(mask, dummy_input_image_data) -> None:
    run_mask_tests(mask, dummy_input_image_data)


# Exceptions
def test_bar_single_object(dummy_input_image_data) -> None:
    config_data = {
        "censor_settings": {
            "enabled_parts": [
                "FEMALE_BREAST_EXPOSED",
            ],
            "default_part_settings": {
                "censors": [
                    {"effect": "Overlay", "parameters": {"colour": "BLACK"}},
                ],
                "mask": "Bar",
            },
        },
    }

    dummy_input_image_data.update_parts("FEMALE_BREAST_EXPOSED")
    run_image_test(
        dummy_input_image_data,
        config=config_data,
        group_name=os.path.splitext(
            os.path.basename(inspect.stack()[0].filename),
        )[0],
        edge_case=True,
    )

import inspect
import os
import platform

import pytest

from censor_engine.libs.detectors.box_based_detectors.nude_net import (
    NudeNetDetector,
)
from censor_engine.libs.registries import EffectRegistry
from censor_engine.models.enums import EffectType
from tests.utils import run_image_test

effects = EffectRegistry.get_all()
effects_blur = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.BLUR
]
effects_overlay = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.OVERLAY
]
effects_colour = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.COLOUR
]
effects_dev = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.DEV
]
effects_edge_detection = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.EDGE_DETECTION
]
effects_noise = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.NOISE
]
effects_pixelation = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.PIXELATION
]
effects_stylisation = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.STYLISATION
]
effects_text = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.TEXT
]
effects_transparency = [
    effect
    for effect, class_effect in effects.items()
    if class_effect.effect_type == EffectType.TRANSPARENCY
]


def run_tests(
    effect,
    dummy_input_image_data,
    expect_png: bool = False,
    mean_absolute_error: float = 6,
    is_text: bool = False,
) -> None:
    all_parts = list(NudeNetDetector.model_classifiers)

    merge = "none" if effect == "Debug" else "groups"
    censor_params = {}
    if is_text and platform.system() == "Linux":
        censor_params = {"parameters": {"font": "DejaVuSans"}}
    config_data = {
        "censor_settings": {
            "enabled_parts": all_parts,
            "merge_settings": {"merge_groups": [all_parts]},
            "default_part_settings": {
                "censors": [{"effect": effect, **censor_params}],
            },
        },
        "render_settings": {"merge_method": merge},
    }

    run_image_test(
        dummy_input_image_data,
        config=config_data,
        subfolder=effect,
        batch_tests=True,
        group_name=os.path.splitext(
            os.path.basename(inspect.stack()[1].filename),
        )[0],
        expect_png=expect_png,
        mean_absolute_error=mean_absolute_error,
    )


@pytest.mark.parametrize("effect", effects_blur)
def test_blur_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data)


@pytest.mark.parametrize("effect", effects_overlay)
def test_overlay_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data)


@pytest.mark.parametrize("effect", effects_colour)
def test_colour_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data)


@pytest.mark.parametrize("effect", effects_dev)
def test_dev_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data)


@pytest.mark.parametrize("effect", effects_edge_detection)
def test_edge_detection_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data)


@pytest.mark.parametrize("effect", effects_noise)
def test_noise_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data, mean_absolute_error=8.5)


@pytest.mark.parametrize("effect", effects_pixelation)
def test_pixelation_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data)


@pytest.mark.parametrize("effect", effects_stylisation)
def test_stylisation_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data)


@pytest.mark.parametrize("effect", effects_text)
def test_text_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data, is_text=True)


@pytest.mark.parametrize("effect", effects_transparency)
def test_transparency_effects(effect, dummy_input_image_data) -> None:
    run_tests(effect, dummy_input_image_data, expect_png=True)

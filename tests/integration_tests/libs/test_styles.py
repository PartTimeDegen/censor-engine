import inspect
import os
from censor_engine.models.enums import StyleType
from tests.utils import run_image_test
from censor_engine.libs.detectors.multi_detectors import NudeNetDetector
import pytest
from censor_engine.libs.registries import StyleRegistry

styles = StyleRegistry.get_all()
styles_blur = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.BLUR
]
styles_overlay = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.OVERLAY
]
styles_colour = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.COLOUR
]
styles_dev = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.DEV
]
styles_edge_detection = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.EDGE_DETECTION
]
styles_noise = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.NOISE
]
styles_pixelation = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.PIXELATION
]
styles_stylisation = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.STYLISATION
]
styles_text = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.TEXT
]
styles_transparency = [
    style
    for style, class_style in styles.items()
    if class_style.style_type == StyleType.TRANSPARENCY
]


def run_tests(
    style,
    dummy_input_image_data,
    expect_png: bool = False,
    mean_absolute_error: float = 6,
):
    all_parts = list(NudeNetDetector.model_classifiers)
    config_data = {
        "render_settings": {"smoothing": True},
        "censor_settings": {
            "enabled_parts": all_parts,
            "merge_settings": {"merge_groups": [all_parts]},
            "default_part_settings": {
                "censors": [{"function": style}],
            },
        },
    }

    dummy_input_image_data.update_parts("FEMALE_BREAST_EXPOSED")
    run_image_test(
        dummy_input_image_data,
        config_data,
        subfolder=style,
        batch_tests=True,
        group_name=os.path.splitext(
            os.path.basename(inspect.stack()[1].filename)
        )[0],
        expect_png=expect_png,
        mean_absolute_error=mean_absolute_error,
    )


@pytest.mark.parametrize("style", styles_blur)
def test_blur_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_overlay)
def test_overlay_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_colour)
def test_colour_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_dev)
def test_dev_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_edge_detection)
def test_edge_detection_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_noise)
def test_noise_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data, mean_absolute_error=7.5)


@pytest.mark.parametrize("style", styles_pixelation)
def test_pixelation_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_stylisation)
def test_stylisation_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_text)
def test_text_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data)


@pytest.mark.parametrize("style", styles_transparency)
def test_transparency_styles(style, dummy_input_image_data):
    run_tests(style, dummy_input_image_data, expect_png=True)

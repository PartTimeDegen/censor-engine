from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from censor_engine.libraries.effects.colours import (
    ColourMask,
    Contrast,
    DuoTone,
    Greyscale,
    HeatMap,
    Negative,
    Posterise,
    Sepia,
)
from censor_engine.models.libraries.effects.schemas import EffectContext
from tests.helpers.effect_handler import run_effect_tester

file_path = Path(__file__)


class TestEffects:
    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
        ],
    )
    def test_greyscale(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Greyscale, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("colour_one", "GREY"),
            ("colour_two", "PINK"),
        ],
    )
    def test_duotone(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, DuoTone, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("colour_map", "COLORMAP_AUTUMN"),
            ("colour_map", "COLORMAP_OCEAN"),
            ("colour_map", "COLORMAP_CIVIDIS"),
        ],
    )
    def test_heatmap(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        value_fixed = {
            "COLORMAP_AUTUMN": cv2.COLORMAP_AUTUMN,
            "COLORMAP_OCEAN": cv2.COLORMAP_OCEAN,
            "COLORMAP_CIVIDIS": cv2.COLORMAP_CIVIDIS,
        }.get(value)
        run_effect_tester(
            file_path, effect_context, HeatMap, param, value_fixed
        )

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("contrast_alpha", 0.5),
            ("contrast_alpha", 3),
            ("contrast_beta", 0.5),
            ("contrast_beta", 3),
        ],
    )
    def test_contrast(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Contrast, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("hsv_lower_limit", "(120, 120, 120)"),
            ("hsv_lower_limit", "(160, 160, 160)"),
            ("hsv_upper_limit", "(70, 70, 70)"),
            ("hsv_upper_limit", "(120, 120, 120)"),
            ("use_greyscale", False),
        ],
    )
    def test_colour_mask(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, ColourMask, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("levels", 2),
            ("levels", 8),
        ],
    )
    def test_posterise(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Posterise, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
        ],
    )
    def test_negative(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Negative, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
        ],
    )
    def test_sepia(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Sepia, param, value)

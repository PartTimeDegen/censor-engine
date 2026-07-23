from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from censor_engine.libraries.effects.blurs import (
    BilateralBlur,
    Blur,
    GaussianBlur,
    MedianBlur,
    MotionBlur,
)
from censor_engine.models.libraries.effects.schemas import EffectContext
from tests.helpers.effect_handler import run_effect_tester

file_path = Path(__file__)


class TestEffects:
    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("factor", 5),
            ("factor", 100),
        ],
    )
    def test_blur(self, param: str, value: Any, effect_context: EffectContext):
        run_effect_tester(file_path, effect_context, Blur, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("factor", 5),
            ("factor", 100),
        ],
    )
    def test_gaussian_blur(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, GaussianBlur, param, value
        )

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("factor", 5),
            ("factor", 100),
        ],
    )
    def test_median_blur(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, MedianBlur, param, value)

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("distance", 2),
            ("distance", 100),
            ("sigma_colour", 10),
            ("sigma_colour", 300),
            ("sigma_space", 10),
            ("sigma_space", 30),
        ],
    )
    def test_bilateral_blur(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, BilateralBlur, param, value
        )

    @pytest.mark.parametrize(
        "param, value",
        [
            (None, None),
            ("offset", 2),
            ("offset", 100),
            ("angle", 0),
            ("angle", 90),
        ],
    )
    def test_motion_blur(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, MotionBlur, param, value)

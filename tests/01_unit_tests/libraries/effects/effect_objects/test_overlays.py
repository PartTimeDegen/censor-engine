from pathlib import Path
from typing import Any

import cv2
import pytest

from censor_engine.libraries.effects.overlays import (
    MissingEffect,
    Outline,
    Overlay,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from tests.helpers.effect_handler import run_effect_tester

file_path = Path(__file__)


class TestEffects:
    def test_missing(self, effect_context: EffectContext):
        run_effect_tester(file_path, effect_context, MissingEffect, None, None)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("colour", "PINK"),
            ("colour", "black"),
        ],
    )
    def test_overlay(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Overlay, param, value)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("colour", "PINK"),
            ("colour", "black"),
            ("thickness", 1),
            ("thickness", 5),
            ("linetype", cv2.LINE_8),
        ],
    )
    def test_outline(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Outline, param, value)


class TestAlternativeCases:
    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("include_image_borders", True),
        ],
    )
    def test_outline_image_edges(
        self, param: str, value: Any, effect_context_blanket: EffectContext
    ):

        run_effect_tester(
            file_path,
            effect_context_blanket,
            Outline,
            param,
            value,
            secondary="_image_borders",
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("include_image_borders", True),
        ],
    )
    def test_outline_image_all_edges(
        self, param: str, value: Any, effect_context_fully: EffectContext
    ):

        run_effect_tester(
            file_path,
            effect_context_fully,
            Outline,
            param,
            value,
            secondary="_image_borders_all",
        )

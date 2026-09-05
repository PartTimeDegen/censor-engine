from pathlib import Path
from typing import Any

import pytest

from censor_engine.libraries.effects.stylisation import (
    Painting,
    Pencil,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)
from tests.helpers.effect_handler import run_effect_tester

file_path = Path(__file__)


class TestEffects:
    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("sigma_s", 10),
            ("sigma_s", 100),
            ("sigma_r", 0.10),
            ("sigma_r", 0.90),
        ],
    )
    def test_overlay(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Painting, param, value)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("sigma_s", 10),
            ("sigma_s", 100),
            ("sigma_r", 0.10),
            ("sigma_r", 0.90),
            ("shade_factor", 0.1),
            ("shade_factor", 0.3),
            ("_use_grey", None),
        ],
    )
    def test_outline(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        secondary = None
        if param == "_use_grey":
            effect_context.general_settings.pre_processing.greyscale = True
            secondary = "grey_test"
            param = None

        run_effect_tester(
            file_path,
            effect_context,
            Pencil,
            param,
            value,
            secondary=secondary,
        )

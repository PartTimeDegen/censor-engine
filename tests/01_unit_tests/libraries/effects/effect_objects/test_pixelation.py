from pathlib import Path
from typing import Any

import pytest

from censor_engine.libraries.effects.pixelation import (
    HexagonPixelate,
    Pixelate,
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
            ("factor", 6),
            ("factor", 24),
        ],
    )
    def test_hexagon_pixelate(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, HexagonPixelate, param, value
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("factor", 6),
            ("factor", 24),
        ],
    )
    def test_pixelate(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Pixelate, param, value)

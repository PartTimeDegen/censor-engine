from pathlib import Path
from typing import Any

import pytest

from censor_engine.libraries.effects.crystallisation import (
    GridCrystallise,
    RandomCrystallise,
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
            ("point_density", 10),
            ("point_density", 200),
            ("jitter", 0),
            ("jitter", 1),
            ("outline_width", 1),
            ("outline_width", 5),
            ("outline_colour", "PINK"),
        ],
    )
    def test_grid_crystallise(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        extra_args = {"outline_width": 2} if param == "outline_colour" else {}

        run_effect_tester(
            file_path,
            effect_context,
            GridCrystallise,
            param,
            value,
            extra_params=extra_args,
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("point_density", 10),
            ("point_density", 200),
            ("outline_width", 1),
            ("outline_width", 5),
            ("outline_colour", "PINK"),
        ],
    )
    def test_random_crystallise(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        extra_args = {"outline_width": 2} if param == "outline_colour" else {}

        run_effect_tester(
            file_path,
            effect_context,
            RandomCrystallise,
            param,
            value,
            extra_params=extra_args,
        )

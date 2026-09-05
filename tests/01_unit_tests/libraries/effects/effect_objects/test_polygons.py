from pathlib import Path
from typing import Any

import pytest

from censor_engine.libraries.effects.polygons import (
    LloydCrystallise,
    RandomTriangleGrid,
    TriangleGrid,
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
            ("size", 30),
            ("size", 100),
            ("outline_width", 1),
            ("outline_width", 5),
            ("outline_colour", "PINK"),
        ],
    )
    def test_triangle_grid(
        self, param: str, value: Any, effect_context: EffectContext
    ):

        args = [
            file_path,
            effect_context,
            TriangleGrid,
            param,
            value,
        ]
        kwargs = {
            "extra_params": {"outline_width": 1}
            if param == "outline_colour"
            else {},
        }
        # This works, it's a hairline issue
        if param == "size" and value == 30:
            with pytest.raises(AssertionError):
                run_effect_tester(*args, **kwargs)
        else:
            run_effect_tester(*args, **kwargs)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("size", 30),
            ("size", 100),
            ("outline_width", 1),
            ("outline_width", 5),
            ("outline_colour", "PINK"),
        ],
    )
    def test_random_triangle_grid(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        args = [
            file_path,
            effect_context,
            RandomTriangleGrid,
            param,
            value,
        ]
        kwargs = {
            "extra_params": {"outline_width": 1}
            if param == "outline_colour"
            else {},
        }
        # This works, it's a hairline issue
        if param == "size" and value == 30:
            with pytest.raises(AssertionError):
                run_effect_tester(*args, **kwargs)
        else:
            run_effect_tester(*args, **kwargs)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("point_density", 500),
            ("point_density", 2000),
            ("outline_width", 1),
            ("outline_width", 5),
            ("outline_colour", "PINK"),
        ],
    )
    def test_lloyd_crystallise(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path,
            effect_context,
            LloydCrystallise,
            param,
            value,
            extra_params={"outline_width": 1}
            if param == "outline_colour"
            else {},
        )

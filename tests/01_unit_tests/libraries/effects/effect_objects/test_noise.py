from pathlib import Path
from typing import Any

import pytest

from censor_engine.libraries.effects.noise import (
    CentricChromaticAberration,
    ChromaticAberration,
    DeNoise,
    Noise,
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
            ("offset", 3),
            ("offset", 20),
            ("angle", 0),
            ("angle", 90),
        ],
    )
    def test_chromatic_aberration(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, ChromaticAberration, param, value
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("offset", 3),
            ("offset", 20),
        ],
    )
    def test_centric_chromatic_aberration(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, CentricChromaticAberration, param, value
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("intensity", 5),
            ("intensity", 10),
            ("grain_size", 10),
            ("grain_size", 100),
        ],
    )
    def test_noise(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, Noise, param, value)

    @pytest.mark.parametrize(
        ("param", "value"), [(None, None), ("strength", 2), ("strength", 20)]
    )
    def test_denoise(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(file_path, effect_context, DeNoise, param, value)


class TestAlternativeCases:
    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("use_mask_centre", False),
        ],
    )
    def test_centric_chromatic_aberration_off_centre(
        self, param: str, value: Any, effect_context_offset: EffectContext
    ):
        run_effect_tester(
            file_path,
            effect_context_offset,
            CentricChromaticAberration,
            param,
            value,
            secondary="_off_centre",
        )

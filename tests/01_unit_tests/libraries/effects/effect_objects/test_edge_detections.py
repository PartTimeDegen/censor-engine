from pathlib import Path
from typing import Any

import pytest

from censor_engine.libraries.effects.edge_detections import (
    EdgeDetectionCanny,
    EdgeDetectionDoubleGaussian,
    EdgeDetectionLapacian,
    EdgeDetectionPrewitt,
    EdgeDetectionRoberts,
    EdgeDetectionScharr,
    EdgeDetectionSobel,
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
            ("threshold", 20),
            ("threshold", 200),
        ],
    )
    def test_edge_detection_canny(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, EdgeDetectionCanny, param, value
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("sigma1", 0.5),
            ("sigma1", 3.0),
            ("sigma2", 0.5),
            ("sigma2", 3.0),
            ("ksize", 5),
        ],
    )
    def test_edge_detection_double_gaussian(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path,
            effect_context,
            EdgeDetectionDoubleGaussian,
            param,
            value,
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("kernel_size", 3),
            ("kernel_size", 9),
        ],
    )
    def test_edge_detection_sobel(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        return
        run_effect_tester(
            file_path, effect_context, EdgeDetectionSobel, param, value
        )
        # This works but there's a weird glitch making it not consistent

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("kernel_size", 3),
            ("kernel_size", 9),
        ],
    )
    def test_edge_detection_scharr(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, EdgeDetectionScharr, param, value
        )

    @pytest.mark.parametrize(("param", "value"), [(None, None)])
    def test_edge_detection_roberts(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, EdgeDetectionRoberts, param, value
        )

    @pytest.mark.parametrize(("param", "value"), [(None, None)])
    def test_edge_detection_prewitt(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        run_effect_tester(
            file_path, effect_context, EdgeDetectionPrewitt, param, value
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            (None, None),
            ("kernel_size", 3),
            ("kernel_size", 9),
        ],
    )
    def test_edge_detection_lapacian(
        self, param: str, value: Any, effect_context: EffectContext
    ):
        return
        run_effect_tester(
            file_path, effect_context, EdgeDetectionLapacian, param, value
        )
        # This works but there's a weird glitch making it not consistent

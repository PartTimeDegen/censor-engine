import numpy as np

from censor_engine.models.libraries.effects.helpers.crystallisation import (
    CrystallisationHelpers,
)
from censor_engine.models.libraries.effects.helpers.polygons import (
    PolygonHelpers,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)

from .base_effect.base_effect import Effect
from .enums import EffectType


class BlurEffect(Effect):
    effect_type = EffectType.BLUR


class ColourEffect(Effect):
    effect_type = EffectType.COLOUR


class DevelopmentEffect(Effect):
    effect_type = EffectType.DEV


class EdgeDetectionEffect(Effect):
    effect_type = EffectType.EDGE_DETECTION


class NoiseEffect(Effect):
    effect_type = EffectType.NOISE


class OverlayEffect(Effect):
    effect_type = EffectType.OVERLAY


class PixelateEffect(Effect):
    effect_type = EffectType.PIXELATION


class CrystallisationEffect(Effect):
    effect_type = EffectType.CRYSTALLISATION
    helper: CrystallisationHelpers

    def __init__(self):
        super().__init__()
        self.helper = CrystallisationHelpers()


class PolygonEffect(Effect):
    effect_type = EffectType.POLYGON
    helper: PolygonHelpers

    def __init__(self):
        super().__init__()
        self.helper = PolygonHelpers()

    def _polygon_function(
        self,
        effect_context: EffectContext,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> list[np.ndarray]:
        raise NotImplementedError


class StylisationEffect(Effect):
    effect_type = EffectType.STYLISATION


class TextEffect(Effect):
    effect_type = EffectType.TEXT


class TransparentEffect(Effect):
    effect_type = EffectType.TRANSPARENCY
    requires_alpha_channel: bool = True

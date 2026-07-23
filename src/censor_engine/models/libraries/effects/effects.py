from ._base_effect import Effect
from .enums import EffectType


class BlurEffect(Effect):
    effect_type = EffectType.BLUR


class ColourEffect(Effect):
    effect_type = EffectType.COLOUR

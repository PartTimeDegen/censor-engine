from typing import ClassVar

import cv2
import numpy as np

from censor_engine._typing import Image

from .enums import EffectType
from .helpers.general_effects import GeneralHelpers
from .schemas import EffectContext


class Effect:
    effect_type: ClassVar[EffectType]

    # Mechanisms
    helpers: GeneralHelpers

    # Supporting Information
    force_png: bool = False
    default_linetype: int = cv2.LINE_AA

    def __init__(self):
        self.helpers = GeneralHelpers()

    # Core Methods
    def apply_effect_to_image(
        self,
        effect_context: EffectContext,
        image_output: Image,
    ) -> Image:
        return np.where(
            effect_context.mask_bool[..., None],
            image_output,
            effect_context.image,
        )

    # Pipeline
    def _pre_process(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> tuple[EffectContext, dict]:
        return (effect_context, kwargs)

    def _internal_generate_effect(
        self,
        effect_context: EffectContext,
        **kwargs: dict,
    ) -> Image:
        new_effect_context, new_kwargs = self._pre_process(
            effect_context, **kwargs
        )
        return self.generate_effect(new_effect_context, **new_kwargs)

    def generate_effect(
        self,
        effect_context: EffectContext,
        **kwargs: dict,
    ) -> Image:
        msg = f"{self.__class__.__name__} does not implement generate_effect()"
        raise NotImplementedError(msg)

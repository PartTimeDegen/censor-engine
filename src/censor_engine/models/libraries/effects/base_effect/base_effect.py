from typing import ClassVar

import numpy as np

from censor_engine._typing import Image
from censor_engine.models.libraries.effects.enums import EffectType
from censor_engine.models.libraries.effects.helpers.general_effects import (
    GeneralHelpers,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


class Effect:
    effect_type: ClassVar[EffectType]

    # Mechanisms
    helpers: GeneralHelpers

    # Supporting Information
    force_png: ClassVar[bool] = False

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
            effect_context.original_image,
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

    # Public Methods
    def generate_effect_pre_process(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return effect_context.image

    def generate_effect(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        msg = f"{self.__class__.__name__} does not implement generate_effect()"
        raise NotImplementedError(msg)

    def generate_effect_post_process(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return effect_context.image

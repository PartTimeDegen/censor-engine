import cv2

from censor_engine._typing import Image
from censor_engine.models.libraries.effects.base_effect._common_methods import (  # type: ignore
    CommonProcessingMethods,
)
from censor_engine.models.libraries.effects.schemas.schemas import (
    EffectContext,
)


class PostProcessingPipeline(CommonProcessingMethods):
    def _create_black_and_white(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return effect_context.image

    def _create_inverse(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return cv2.bitwise_not(effect_context.image)  # type: ignore

    def _create_fade(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return effect_context.image

    def _create_glow(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return effect_context.image

    def _create_blur(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return effect_context.image

    def _create_alpha(
        self, effect_context: EffectContext, **kwargs: dict
    ) -> Image:
        return effect_context.image

    def post_process_image(
        self,
        effect_context: EffectContext,
        **kwargs: dict,
    ):
        settings = effect_context.general_settings.post_processing
        image = effect_context.image

        effects = [
            ("greyscale", self._create_greyscale),
            ("black_and_white", self._create_black_and_white),  # TODO
            ("inverse", self._create_inverse),
            ("fade", self._create_fade),  # TODO
            ("glow", self._create_glow),  # TODO
            ("blur", self._create_blur),  # TODO
            ("alpha", self._create_alpha),  # TODO
        ]

        for setting, effect in effects:
            if getattr(settings, setting):
                effect_context.image = image
                image = effect(effect_context, **kwargs)

        return image
